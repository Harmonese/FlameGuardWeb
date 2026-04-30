
"""
sweep_kappa_closed_loop.py

用途
----
1. 调用工作目录中的 solve_inner_preheater.py
2. 进行闭环滚动仿真
3. 对候选 kappa 做粗筛
4. 导出汇总结果和完整时间序列

重要约定
--------
- 本脚本默认与 solve_inner_preheater.py 放在同一工作目录
- 会从该文件导入：
    Config
    solve_inner_lexicographic
    validate_composition
    T_avg_proxy
    T_min_proxy
    T_max_proxy
    sigma_proxy

运行前
------
pip install numpy pandas

运行
----
python sweep_kappa_closed_loop.py

输出
----
- kappa_sweep_summary.csv
- kappa_sweep_trajectories.csv
- kappa_sweep_failures.csv
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import math
import traceback

import numpy as np
import pandas as pd

# =========================================================
# 0) 导入固定命名的内层求解器
# =========================================================

# 重要：用户已说明工作目录中的文件名固定为 solve_inner_preheater.py
from solve_inner_preheater import (
    Config,
    solve_inner_lexicographic,
    validate_composition,
    T_avg_proxy,
    T_min_proxy,
    T_max_proxy,
    sigma_proxy,
)


# =========================================================
# 1) 你需要先改的地方
# =========================================================

# 候选 kappa 粗筛集合
KAPPA_CANDIDATES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# 闭环仿真步数
N_STEPS = 12

# 收敛判据：认为已经回到稳态附近
SETTLE_BAND_DEGC = 3.0

# 结果输出文件名
SUMMARY_CSV = "kappa_sweep_summary.csv"
TRAJ_CSV = "kappa_sweep_trajectories.csv"
FAIL_CSV = "kappa_sweep_failures.csv"

# ===== 场景模板 =====
# 你后续只需要把这里改成你希望评估的场景
# x 的顺序固定为：[菜叶, 西瓜皮, 橙子皮, 肉, 杂项混合, 米饭]
SCENARIOS = [
    {
        "name": "typical_mid_cold",
        "x": [0.20, 0.15, 0.15, 0.10, 0.20, 0.20],
        "T_avg0": 870.0,
    },
    {
        "name": "typical_mid_hot",
        "x": [0.20, 0.15, 0.15, 0.10, 0.20, 0.20],
        "T_avg0": 900.0,
    },
    {
        "name": "wet_mix_cold",
        "x": [0.25, 0.25, 0.20, 0.05, 0.15, 0.10],
        "T_avg0": 880.0,
    },
    {
        "name": "dry_mix_hot",
        "x": [0.05, 0.05, 0.10, 0.30, 0.20, 0.30],
        "T_avg0": 892.0,
    },
]


# =========================================================
# 2) 数据结构
# =========================================================

@dataclass
class StepRecord:
    kappa: float
    scenario: str
    step: int
    T_avg_prev: float
    T_set: float
    error_prev: float
    w_hat_prev: float
    T_tar_k: float
    w_tar_k: float
    Tg: float
    vg: float
    Tm: float
    w_opt: float
    T_avg_next: float
    T_min_next: float
    T_max_next: float
    sigma_next: float
    power_kW: float
    d_w_minus: float
    d_w_plus: float
    stage1_success: bool
    stage2_success: bool
    stage3_success: bool
    numerically_feasible: bool
    max_violation: float
    hard_violation: bool
    error_next: float


@dataclass
class ScenarioSummary:
    kappa: float
    scenario: str
    completed_steps: int
    solver_fail: bool
    any_numerical_infeasible: bool
    any_hard_violation: bool
    settle_step: Optional[int]
    final_error_abs: float
    max_abs_error: float
    max_overshoot_hot: float
    max_overshoot_cold: float
    cumulative_power_kW: float
    avg_sigma: float
    total_abs_dTg: float
    total_abs_dvg: float
    score: float
    fail_message: str = ""


# =========================================================
# 3) 工具函数
# =========================================================

def hard_violation_from_outputs(Tavg: float, Tmin: float, Tmax: float, cfg: Config) -> bool:
    return bool(
        (Tavg < cfg.TAVG_BURN_MIN - 1e-6)
        or (Tmin < cfg.TMIN_BURN_MIN - 1e-6)
        or (Tmax > cfg.TMAX_BURN_MAX + 1e-6)
    )


def settling_step(records: List[StepRecord], settle_band: float) -> Optional[int]:
    """
    返回第一次进入稳态带并在之后一直保持的步号。
    若从未满足，则返回 None。
    """
    if not records:
        return None
    errors = [abs(r.error_next) for r in records]
    n = len(errors)
    for i in range(n):
        if all(e <= settle_band for e in errors[i:]):
            return i + 1  # 步号从 1 开始
    return None


def compute_score(summary: ScenarioSummary) -> float:
    """
    粗筛评分函数，值越小越好。
    设计原则：
    - 求解失败 / 硬约束违背：重罚
    - 收敛慢：罚
    - 过冲大：罚
    - 动作不平滑：罚
    - 能耗高：轻罚
    """
    score = 0.0
    if summary.solver_fail:
        score += 1e8
    if summary.any_hard_violation:
        score += 1e7
    if summary.any_numerical_infeasible:
        score += 1e6

    settle = summary.settle_step if summary.settle_step is not None else 100
    score += 1000.0 * settle
    score += 200.0 * summary.max_abs_error
    score += 150.0 * max(summary.max_overshoot_hot, summary.max_overshoot_cold)
    score += 5.0 * summary.total_abs_dTg
    score += 20.0 * summary.total_abs_dvg
    score += 0.2 * summary.cumulative_power_kW
    score += 50.0 * summary.final_error_abs
    return score


# =========================================================
# 4) 单场景闭环仿真
# =========================================================

def simulate_closed_loop_for_scenario(
    scenario_name: str,
    x: List[float],
    T_avg0: float,
    kappa: float,
    cfg: Config,
    n_steps: int = N_STEPS,
) -> tuple[List[StepRecord], ScenarioSummary]:
    x_arr = validate_composition(np.array(x, dtype=float))

    traj: List[StepRecord] = []
    T_prev = float(T_avg0)
    prev_Tg = None
    prev_vg = None

    solver_fail = False
    fail_message = ""

    for step in range(1, n_steps + 1):
        try:
            result = solve_inner_lexicographic(
                x=x_arr,
                T_avg_prev=T_prev,
                cfg=cfg,
                kappa=kappa,
                use_outer_feedback=True,
            )
        except Exception as e:
            solver_fail = True
            fail_message = f"step={step}: {type(e).__name__}: {e}"
            break

        outer = result["outer"]
        opt = result["optimal"]
        overall = result.get("overall", {})
        stg = result["stages"]

        T_next = float(opt["Tavg_burn"])
        Tmin_next = float(opt["Tmin_burn"])
        Tmax_next = float(opt["Tmax_burn"])
        sigma_next = float(opt["sigma_burn"])
        Tg = float(opt["Tg"])
        vg = float(opt["vg"])

        numerical_feasible = bool(overall.get("numerically_feasible", True))
        max_violation = float(overall.get("max_constraint_violation", 0.0))
        hard_violation = hard_violation_from_outputs(T_next, Tmin_next, Tmax_next, cfg)

        rec = StepRecord(
            kappa=kappa,
            scenario=scenario_name,
            step=step,
            T_avg_prev=T_prev,
            T_set=cfg.T_SET,
            error_prev=T_prev - cfg.T_SET,
            w_hat_prev=float(outer["w_hat_prev"]),
            T_tar_k=float(outer["T_tar_k"]),
            w_tar_k=float(outer["w_tar_k"]),
            Tg=Tg,
            vg=vg,
            Tm=float(opt["Tm"]),
            w_opt=float(opt["w_opt"]),
            T_avg_next=T_next,
            T_min_next=Tmin_next,
            T_max_next=Tmax_next,
            sigma_next=sigma_next,
            power_kW=float(opt["Power_kW"]),
            d_w_minus=float(opt["d_w_minus"]),
            d_w_plus=float(opt["d_w_plus"]),
            stage1_success=bool(stg["stage1_success"]),
            stage2_success=bool(stg["stage2_success"]),
            stage3_success=bool(stg["stage3_success"]),
            numerically_feasible=numerical_feasible,
            max_violation=max_violation,
            hard_violation=hard_violation,
            error_next=T_next - cfg.T_SET,
        )
        traj.append(rec)

        T_prev = T_next
        prev_Tg = Tg
        prev_vg = vg

    if traj:
        settle = settling_step(traj, SETTLE_BAND_DEGC)
        final_error_abs = abs(traj[-1].error_next)
        max_abs_error = max(abs(r.error_next) for r in traj)

        max_overshoot_hot = max(max(r.error_next, 0.0) for r in traj)   # 高于稳态点
        max_overshoot_cold = max(max(-r.error_next, 0.0) for r in traj) # 低于稳态点

        cumulative_power = sum(r.power_kW for r in traj)
        avg_sigma = float(np.mean([r.sigma_next for r in traj]))

        total_abs_dTg = 0.0
        total_abs_dvg = 0.0
        for i in range(1, len(traj)):
            total_abs_dTg += abs(traj[i].Tg - traj[i-1].Tg)
            total_abs_dvg += abs(traj[i].vg - traj[i-1].vg)

        any_numerical_infeasible = any(not r.numerically_feasible for r in traj)
        any_hard_violation = any(r.hard_violation for r in traj)

        summary = ScenarioSummary(
            kappa=kappa,
            scenario=scenario_name,
            completed_steps=len(traj),
            solver_fail=solver_fail,
            any_numerical_infeasible=any_numerical_infeasible,
            any_hard_violation=any_hard_violation,
            settle_step=settle,
            final_error_abs=final_error_abs,
            max_abs_error=max_abs_error,
            max_overshoot_hot=max_overshoot_hot,
            max_overshoot_cold=max_overshoot_cold,
            cumulative_power_kW=cumulative_power,
            avg_sigma=avg_sigma,
            total_abs_dTg=total_abs_dTg,
            total_abs_dvg=total_abs_dvg,
            score=0.0,
            fail_message=fail_message,
        )
        summary.score = compute_score(summary)
        return traj, summary

    # 完全没跑起来
    summary = ScenarioSummary(
        kappa=kappa,
        scenario=scenario_name,
        completed_steps=0,
        solver_fail=True,
        any_numerical_infeasible=True,
        any_hard_violation=True,
        settle_step=None,
        final_error_abs=9999.0,
        max_abs_error=9999.0,
        max_overshoot_hot=9999.0,
        max_overshoot_cold=9999.0,
        cumulative_power_kW=9999.0,
        avg_sigma=9999.0,
        total_abs_dTg=9999.0,
        total_abs_dvg=9999.0,
        score=1e9,
        fail_message=fail_message if fail_message else "no trajectory generated",
    )
    return traj, summary


# =========================================================
# 5) 主程序：kappa 粗筛
# =========================================================

def main() -> None:
    cfg = Config()

    # 先验证场景输入
    validated_scenarios = []
    for sc in SCENARIOS:
        x = validate_composition(np.array(sc["x"], dtype=float))
        validated_scenarios.append({
            "name": sc["name"],
            "x": x.tolist(),
            "T_avg0": float(sc["T_avg0"]),
        })

    all_traj_rows = []
    all_summary_rows = []
    fail_rows = []

    print("=" * 78)
    print("开始进行 kappa 粗筛闭环仿真")
    print(f"候选 kappa: {KAPPA_CANDIDATES}")
    print(f"场景数: {len(validated_scenarios)}")
    print(f"每条场景仿真步数: {N_STEPS}")
    print("脚本将调用当前工作目录中的 solve_inner_preheater.py")
    print("=" * 78)

    for kappa in KAPPA_CANDIDATES:
        print(f"\n>>> 正在评估 kappa = {kappa:.3f}")
        for sc in validated_scenarios:
            print(f"    - 场景 {sc['name']} ...", end="")
            traj, summary = simulate_closed_loop_for_scenario(
                scenario_name=sc["name"],
                x=sc["x"],
                T_avg0=sc["T_avg0"],
                kappa=kappa,
                cfg=cfg,
                n_steps=N_STEPS,
            )

            all_traj_rows.extend([asdict(r) for r in traj])
            all_summary_rows.append(asdict(summary))

            if summary.solver_fail or summary.any_hard_violation or summary.any_numerical_infeasible:
                fail_rows.append(asdict(summary))

            print(
                f" 完成 | steps={summary.completed_steps}, "
                f"settle={summary.settle_step}, "
                f"final_err={summary.final_error_abs:.3f}, "
                f"score={summary.score:.2f}"
            )

    # 导出 CSV
    traj_df = pd.DataFrame(all_traj_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    fail_df = pd.DataFrame(fail_rows)

    # 每个 kappa 的聚合汇总（平均）
    agg_rows = []
    for kappa, grp in summary_df.groupby("kappa", sort=True):
        agg = {
            "kappa": kappa,
            "n_scenarios": int(len(grp)),
            "solver_fail_count": int(grp["solver_fail"].sum()),
            "hard_violation_count": int(grp["any_hard_violation"].sum()),
            "numerical_infeasible_count": int(grp["any_numerical_infeasible"].sum()),
            "avg_settle_step": float(grp["settle_step"].fillna(100).mean()),
            "avg_final_error_abs": float(grp["final_error_abs"].mean()),
            "max_final_error_abs": float(grp["final_error_abs"].max()),
            "avg_max_abs_error": float(grp["max_abs_error"].mean()),
            "avg_overshoot_hot": float(grp["max_overshoot_hot"].mean()),
            "avg_overshoot_cold": float(grp["max_overshoot_cold"].mean()),
            "avg_cumulative_power_kW": float(grp["cumulative_power_kW"].mean()),
            "avg_sigma": float(grp["avg_sigma"].mean()),
            "avg_total_abs_dTg": float(grp["total_abs_dTg"].mean()),
            "avg_total_abs_dvg": float(grp["total_abs_dvg"].mean()),
            "avg_score": float(grp["score"].mean()),
        }
        agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows).sort_values(
        by=["solver_fail_count", "hard_violation_count", "numerical_infeasible_count", "avg_score", "avg_final_error_abs"],
        ascending=[True, True, True, True, True],
    )

    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    traj_df.to_csv(TRAJ_CSV, index=False, encoding="utf-8-sig")
    fail_df.to_csv(FAIL_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("单场景汇总结果（按 score 从好到差排序）")
    print(summary_df.sort_values(
        by=["solver_fail", "any_hard_violation", "any_numerical_infeasible", "score"],
        ascending=[True, True, True, True]
    ).head(20).to_string(index=False))

    print("\n" + "=" * 78)
    print("按 kappa 聚合后的推荐排序")
    print(agg_df.to_string(index=False))

    print("\n输出文件：")
    print(f"- 单场景汇总: {SUMMARY_CSV}")
    print(f"- 完整时间序列: {TRAJ_CSV}")
    print(f"- 失败/不安全场景: {FAIL_CSV}")

    if not agg_df.empty:
        best = agg_df.iloc[0]
        print("\n当前粗筛下的最佳 kappa 候选：")
        print(f"kappa = {best['kappa']:.3f}")
        print("说明：这只是粗筛结果。下一步建议在它附近做局部细扫，例如 0.45 / 0.50 / 0.55。")


if __name__ == "__main__":
    main()
