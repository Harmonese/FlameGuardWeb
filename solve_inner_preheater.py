
"""
solve_inner_preheater_v2.py

单点求解脚本：外层平均炉温反馈修正 + 内层多目标非线性规划

设计原则
--------
1. 用户输入仍然使用 6 类垃圾组分比例 x1..x6
2. 用户输入上一时刻平均炉温 T_avg_prev
3. 默认由外层反馈公式自动生成 w_tar^k
4. 内层优化严格跟踪动态目标含水率 w_tar^k
5. 本脚本只做“单点求解”，不做批量扫描制表

说明
----
- 当前实现的外层反馈公式，采用更稳健的凸组合形式：
    T_tar^k = clip((1-kappa) * T_avg_prev + kappa * T_set,
                   [T_avg_min_feas, T_avg_max_feas])
  其中 e_{k-1} = T_avg_prev - T_set，kappa 表示“每一步朝稳态点靠近的比例”

依赖
----
pip install numpy scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize


@dataclass
class Config:
    T_REF: float = 175.0
    T0: float = 20.0
    T_AMB: float = 20.0

    MDOT_W: float = 20000.0 / 86400.0   # kg/s

    D: float = 1.2
    L: float = 6.8
    PHI: float = 0.14
    DTUBE: float = 0.168
    N_TUBES: int = 6
    R_DUCT: float = 0.4
    R_STACK: float = 0.3

    RHO_BULK: float = 450.0

    U0: float = 18.97
    K_U: float = 20.09
    N_U: float = 0.65

    CPG: float = 1.05
    CS: float = 1.70
    CW: float = 4.1844
    LAMBDA: float = 2257.9

    RHO_G_REF: float = 0.78
    T_G_REF_K: float = 450.0

    VG_MIN: float = 3.0
    VG_MAX: float = 12.0
    TG_MIN: float = 100.0
    TG_SAFE_MAX: float = 2000.0          # 不作为实际安全约束，仅保留为极大上界
    TM_MAX: float = 250.0
    DELTA_T_MIN: float = 0.0

    TMIN_BURN_MIN: float = 850.0
    TAVG_BURN_MIN: float = 850.0
    TMAX_BURN_MAX: float = 1100.0

    W_REF: float = 31.06
    OMEGA_REF: float = 0.3106
    T_SET: float = 886.28
    TAVG_FEAS_MIN: float = 876.57
    TAVG_FEAS_MAX: float = 895.99
    W_FEAS_MIN: float = 30.32
    W_FEAS_MAX: float = 31.80

    MAXITER: int = 500
    FEAS_TOL: float = 1e-6
    STAGE_TOL: float = 1e-6

    def __post_init__(self) -> None:
        self.A_D = np.pi * self.R_DUCT ** 2
        self.A_S = np.pi * self.R_STACK ** 2
        self.A = np.pi * self.D * self.L + self.N_TUBES * np.pi * self.DTUBE * self.L
        self.M_H = self.RHO_BULK * self.PHI * (np.pi * self.D ** 2 / 4.0) * self.L
        self.TAU_R_MIN = self.M_H / self.MDOT_W / 60.0


OMEGA = np.array([0.948, 0.948, 0.817, 0.442, 0.773, 0.611], dtype=float)
TREF_TIME = np.array([12.1, 17.7, 15.3, 11.5, 16.3, 15.8], dtype=float)
SLOPE = np.array([-0.132, -0.251, -0.189, -0.216, -0.210, -0.243], dtype=float)
NAMES = ["菜叶", "西瓜皮", "橙子皮", "肉", "杂项混合", "米饭"]

IDX_TG = 0
IDX_VG = 1
IDX_TM = 2
IDX_W = 3
IDX_DWM = 4
IDX_DWP = 5


def validate_composition(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float).reshape(-1)
    if len(x) != 6:
        raise ValueError("x 必须是长度为 6 的数组，对应 [菜叶, 西瓜皮, 橙子皮, 肉, 杂项混合, 米饭]")
    if np.any(x < 0):
        raise ValueError("组分比例不能为负数。")
    total = x.sum()
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"组分比例和为 {total:.10f}，不等于 1。请重新输入。")
    return x


def mixed_properties(x: np.ndarray, cfg: Config) -> Dict[str, float]:
    x = validate_composition(x)
    omega0 = float(np.dot(x, OMEGA))
    tref = float(np.dot(x, TREF_TIME))
    slope = float(np.dot(x, SLOPE))
    ceq = (1.0 - omega0) * cfg.CS + omega0 * cfg.CW
    return {"omega0": omega0, "tref": tref, "slope": slope, "ceq": ceq}


def T_stack(w: float) -> float:
    return -14.6746 * w + 1446.1956


def v_stack(w: float) -> float:
    return -0.2206 * w + 25.8437


def T_avg_proxy(w: float) -> float:
    return -13.1210 * w + 1293.8225


def sigma_proxy(w: float) -> float:
    return -0.267 * w + 24.8868


def T_min_proxy(w: float) -> float:
    return -12.514 * w + 1247.89


def T_max_proxy(w: float) -> float:
    return -16.1051 * w + 1588.3027


def rho_g(T_degC: float, cfg: Config) -> float:
    return cfg.RHO_G_REF * cfg.T_G_REF_K / (T_degC + 273.15)


def rho_stack_from_w(w: float, cfg: Config) -> float:
    return rho_g(T_stack(w), cfg)


def rho_pre_from_Tg(Tg: float, cfg: Config) -> float:
    return rho_g(Tg, cfg)


def project(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def estimate_prev_w(T_avg_prev: float) -> float:
    return (1293.8225 - T_avg_prev) / 13.1210


def outer_feedback(T_avg_prev: float, kappa: float, cfg: Config) -> Dict[str, float]:
    """
    推荐采用的凸组合形式：
        e_{k-1} = T_avg_prev - T_set
        T_tar^k = clip((1-kappa) * T_avg_prev + kappa * T_set, feasible_band)
    等价地，它表示“下一步目标温度是当前温度与稳态温度的凸组合”，
    kappa 越大，纠偏越积极；kappa 越小，纠偏越平缓。
    """
    if not (0.0 < kappa <= 1.0):
        raise ValueError("kappa 必须满足 0 < kappa <= 1。")
    w_hat_prev = estimate_prev_w(T_avg_prev)
    e_prev = T_avg_prev - cfg.T_SET
    T_tar_raw = (1.0 - kappa) * T_avg_prev + kappa * cfg.T_SET
    T_tar_k = project(T_tar_raw, cfg.TAVG_FEAS_MIN, cfg.TAVG_FEAS_MAX)
    w_tar_k = (1293.8225 - T_tar_k) / 13.1210
    w_tar_k = project(w_tar_k, cfg.W_FEAS_MIN, cfg.W_FEAS_MAX)
    omega_tar_k = w_tar_k / 100.0
    return {
        "w_hat_prev": w_hat_prev,
        "e_prev": e_prev,
        "T_tar_raw": T_tar_raw,
        "T_tar_k": T_tar_k,
        "w_tar_k": w_tar_k,
        "omega_tar_k": omega_tar_k,
    }


def tau20(Tm: float, x: np.ndarray, cfg: Config) -> float:
    props = mixed_properties(x, cfg)
    return props["tref"] + props["slope"] * (Tm - cfg.T_REF)


def tau_target(Tm: float, x: np.ndarray, w_percent: float, cfg: Config) -> float:
    props = mixed_properties(x, cfg)
    omega0 = props["omega0"]
    omega = w_percent / 100.0
    denom = omega0 - 0.20
    if denom <= 0:
        raise ValueError("omega0 <= 0.20，当前模型的线性缩放动力学不适用。")
    return tau20(Tm, x, cfg) * (omega0 - omega) / denom


def q_sup_kW(Tg: float, vg: float, Tm: float, cfg: Config) -> float:
    U = cfg.U0 + cfg.K_U * (vg ** cfg.N_U)
    return U * cfg.A * (Tg - Tm) / 1000.0


def evap_water_per_kg_wet(omega0: float, omega_target: float) -> float:
    if omega_target >= 1.0:
        raise ValueError("目标湿基含水率必须小于 1。")
    if omega0 <= omega_target:
        return 0.0
    return (omega0 - omega_target) / (1.0 - omega_target)


def q_req_kW(Tm: float, x: np.ndarray, w_percent: float, cfg: Config) -> float:
    props = mixed_properties(x, cfg)
    omega0 = props["omega0"]
    ceq = props["ceq"]
    omega_target = w_percent / 100.0
    m_evap = evap_water_per_kg_wet(omega0, omega_target)
    return cfg.MDOT_W * (ceq * (Tm - cfg.T0) + cfg.LAMBDA * m_evap)


def power_kW(Tg: float, vg: float, cfg: Config) -> float:
    return rho_pre_from_Tg(Tg, cfg) * cfg.A_D * vg * cfg.CPG * (Tg - cfg.T_AMB)


def stack_mass_flow_cap(T_avg_prev: float, cfg: Config) -> Dict[str, float]:
    w_hat_prev = estimate_prev_w(T_avg_prev)
    Tstack = T_stack(w_hat_prev)
    vstack = v_stack(w_hat_prev)
    rho_stack_val = rho_stack_from_w(w_hat_prev, cfg)
    mdot_cap = rho_stack_val * cfg.A_S * vstack
    Tg_cap = Tstack
    return {
        "w_hat_prev": w_hat_prev,
        "T_stack_cap": Tstack,
        "v_stack_cap": vstack,
        "rho_stack_cap": rho_stack_val,
        "mdot_stack_cap": mdot_cap,
        "Tg_cap": Tg_cap,
    }


def bounds(cfg: Config, T_avg_prev: float) -> List[Tuple[float, float]]:
    caps = stack_mass_flow_cap(T_avg_prev, cfg)
    Tg_hi = caps["Tg_cap"]
    if Tg_hi < cfg.TG_MIN:
        raise ValueError(
            f"动态烟气温度上限仅为 {Tg_hi:.4f}°C，低于预热炉最小可运行温度 {cfg.TG_MIN:.1f}°C，当前工况下不可行。"
        )
    big = 1e6
    return [
        (cfg.TG_MIN, Tg_hi),
        (cfg.VG_MIN, cfg.VG_MAX),
        (cfg.T0, cfg.TM_MAX),
        (cfg.W_FEAS_MIN, cfg.W_FEAS_MAX),
        (0.0, big),
        (0.0, big),
    ]


def eq_constraints(z: np.ndarray, w_tar_k: float) -> np.ndarray:
    w = z[IDX_W]
    dwm = z[IDX_DWM]
    dwp = z[IDX_DWP]
    g_track = w + dwm - dwp - w_tar_k
    return np.array([g_track], dtype=float)


def ineq_constraints(z: np.ndarray, x: np.ndarray, T_avg_prev: float, cfg: Config) -> np.ndarray:
    Tg, vg, Tm, w = z[IDX_TG], z[IDX_VG], z[IDX_TM], z[IDX_W]
    caps = stack_mass_flow_cap(T_avg_prev, cfg)

    g_tmin = T_min_proxy(w) - cfg.TMIN_BURN_MIN
    g_tavg = T_avg_proxy(w) - cfg.TAVG_BURN_MIN
    g_tmax = cfg.TMAX_BURN_MAX - T_max_proxy(w)

    g_tau = cfg.TAU_R_MIN - tau_target(Tm, x, w, cfg)
    g_heat = q_sup_kW(Tg, vg, Tm, cfg) - q_req_kW(Tm, x, w, cfg)

    g_resource_mdot = caps["mdot_stack_cap"] - rho_pre_from_Tg(Tg, cfg) * cfg.A_D * vg
    g_tm_tg = Tg - Tm - cfg.DELTA_T_MIN

    return np.array(
        [g_tmin, g_tavg, g_tmax, g_tau, g_heat, g_resource_mdot, g_tm_tg],
        dtype=float,
    )


def objective_stage1(z: np.ndarray) -> float:
    return z[IDX_DWM] + z[IDX_DWP] + 1e-8 * (z[IDX_TG] + z[IDX_VG] + z[IDX_TM] + z[IDX_W])


def objective_stage2(z: np.ndarray) -> float:
    return sigma_proxy(z[IDX_W])


def objective_stage3(z: np.ndarray, cfg: Config) -> float:
    return power_kW(z[IDX_TG], z[IDX_VG], cfg)


def max_constraint_violation(
    z: np.ndarray,
    x: np.ndarray,
    T_avg_prev: float,
    w_tar_k: float,
    cfg: Config,
    extra_ineqs=None,
) -> float:
    extra_ineqs = extra_ineqs or []
    eqv = np.max(np.abs(eq_constraints(z, w_tar_k)))
    ineqv = np.max(np.maximum(-ineq_constraints(z, x, T_avg_prev, cfg), 0.0))
    extra = 0.0
    if extra_ineqs:
        vals = np.array([f(z) for f in extra_ineqs], dtype=float)
        extra = np.max(np.maximum(-vals, 0.0))
    return max(eqv, ineqv, extra)


def build_initial_guess(Tg: float, vg: float, Tm: float, w: float, w_tar_k: float) -> np.ndarray:
    if w > w_tar_k:
        dwm, dwp = 0.0, w - w_tar_k
    else:
        dwm, dwp = w_tar_k - w, 0.0
    return np.array([Tg, vg, Tm, w, dwm, dwp], dtype=float)


def initial_guesses(T_avg_prev: float, w_tar_k: float, cfg: Config) -> List[np.ndarray]:
    caps = stack_mass_flow_cap(T_avg_prev, cfg)
    Tg_hi = max(cfg.TG_MIN, min(caps["Tg_cap"], cfg.TG_SAFE_MAX))
    Tg_candidates = sorted(set([
        cfg.TG_MIN,
        min(180.0, Tg_hi),
        min(300.0, Tg_hi),
        min(500.0, Tg_hi),
        min(800.0, Tg_hi),
        Tg_hi,
    ]))
    vg_candidates = [3.5, 5.0, 8.0, 11.0]
    Tm_candidates = [45.0, 65.0, 85.0, 100.0]
    w_candidates = sorted(set([cfg.W_FEAS_MIN, w_tar_k, 31.06, cfg.W_FEAS_MAX]))

    seeds = []
    for Tg in Tg_candidates:
        for vg in vg_candidates:
            for Tm in Tm_candidates:
                if Tm <= cfg.TM_MAX and Tg - Tm >= cfg.DELTA_T_MIN:
                    for w in w_candidates:
                        seeds.append(build_initial_guess(Tg, vg, Tm, w, w_tar_k))
    return seeds


def solve_stage(
    x: np.ndarray,
    T_avg_prev: float,
    w_tar_k: float,
    cfg: Config,
    objective,
    extra_ineqs=None,
):
    extra_ineqs = extra_ineqs or []

    example_seed = initial_guesses(T_avg_prev, w_tar_k, cfg)[0]
    n_ineq = ineq_constraints(example_seed, x, T_avg_prev, cfg).shape[0]

    cons = [{"type": "eq", "fun": lambda z: eq_constraints(z, w_tar_k)[0]}]
    for i in range(n_ineq):
        cons.append({"type": "ineq", "fun": lambda z, idx=i: ineq_constraints(z, x, T_avg_prev, cfg)[idx]})
    for f in extra_ineqs:
        cons.append({"type": "ineq", "fun": f})

    best_res = None
    best_obj = np.inf

    for x0 in initial_guesses(T_avg_prev, w_tar_k, cfg):
        try:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds(cfg, T_avg_prev),
                constraints=cons,
                options={"maxiter": cfg.MAXITER, "ftol": 1e-10, "disp": False},
            )
        except Exception:
            continue

        if res.x is None:
            continue

        vio = max_constraint_violation(res.x, x, T_avg_prev, w_tar_k, cfg, extra_ineqs)
        obj = objective(res.x)

        if vio <= 1e-5 and obj < best_obj:
            best_obj = obj
            best_res = res
        elif best_res is None:
            if res.success or vio < 1e-2:
                best_obj = obj
                best_res = res

    if best_res is None:
        raise RuntimeError("求解失败：未找到可行解。建议检查输入或放宽边界。")

    return best_res


def solve_inner_lexicographic(
    x: np.ndarray,
    T_avg_prev: float,
    cfg: Optional[Config] = None,
    *,
    kappa: float = 0.5,
    w_tar_k: Optional[float] = None,
    use_outer_feedback: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or Config()
    x = validate_composition(x)
    props = mixed_properties(x, cfg)

    if use_outer_feedback:
        outer = outer_feedback(T_avg_prev, kappa, cfg)
        target_w = outer["w_tar_k"]
    else:
        if w_tar_k is None:
            raise ValueError("当 use_outer_feedback=False 时，必须显式提供 w_tar_k。")
        target_w = float(w_tar_k)
        outer = {
            "w_hat_prev": estimate_prev_w(T_avg_prev),
            "e_prev": T_avg_prev - cfg.T_SET,
            "T_tar_k": T_avg_proxy(target_w),
            "w_tar_k": target_w,
            "omega_tar_k": target_w / 100.0,
        }

    caps = stack_mass_flow_cap(T_avg_prev, cfg)

    res1 = solve_stage(x, T_avg_prev, target_w, cfg, objective_stage1)
    stage1_star = objective_stage1(res1.x)

    extra2 = [lambda z, cap=stage1_star + cfg.STAGE_TOL: cap - objective_stage1(z)]
    res2 = solve_stage(x, T_avg_prev, target_w, cfg, objective_stage2, extra_ineqs=extra2)
    stage2_star = objective_stage2(res2.x)

    extra3 = [
        lambda z, cap=stage1_star + cfg.STAGE_TOL: cap - objective_stage1(z),
        lambda z, cap=stage2_star + cfg.STAGE_TOL: cap - objective_stage2(z),
    ]
    res3 = solve_stage(
        x,
        T_avg_prev,
        target_w,
        cfg,
        objective=lambda z: objective_stage3(z, cfg),
        extra_ineqs=extra3,
    )

    z = res3.x
    Tg, vg, Tm, w = z[IDX_TG], z[IDX_VG], z[IDX_TM], z[IDX_W]

    final_vio = max_constraint_violation(res3.x, x, T_avg_prev, target_w, cfg)

    return {
        "config": cfg,
        "x": x,
        "mixed_props": props,
        "outer": outer,
        "resource_caps": caps,
        "overall": {
            "max_constraint_violation": float(final_vio),
            "numerically_feasible": bool(final_vio <= 1e-5),
        },
        "optimal": {
            "Tg": float(Tg),
            "vg": float(vg),
            "Tm": float(Tm),
            "w_opt": float(w),
            "omega_opt": float(w / 100.0),
            "d_w_minus": float(z[IDX_DWM]),
            "d_w_plus": float(z[IDX_DWP]),
            "tau20_min": float(tau20(Tm, x, cfg)),
            "tau_target_min": float(tau_target(Tm, x, w, cfg)),
            "tau_r_min": float(cfg.TAU_R_MIN),
            "Qsup_kW": float(q_sup_kW(Tg, vg, Tm, cfg)),
            "Qreq_kW": float(q_req_kW(Tm, x, w, cfg)),
            "Power_kW": float(power_kW(Tg, vg, cfg)),
            "T_stack_cap": float(caps["T_stack_cap"]),
            "v_stack_cap": float(caps["v_stack_cap"]),
            "mdot_stack_cap": float(caps["mdot_stack_cap"]),
            "mdot_preheater": float(rho_pre_from_Tg(Tg, cfg) * cfg.A_D * vg),
            "Tmin_burn": float(T_min_proxy(w)),
            "Tavg_burn": float(T_avg_proxy(w)),
            "Tmax_burn": float(T_max_proxy(w)),
            "sigma_burn": float(sigma_proxy(w)),
            "deltaT": float(Tg - Tm),
        },
        "stages": {
            "stage1_success": bool(res1.success),
            "stage2_success": bool(res2.success),
            "stage3_success": bool(res3.success),
            "stage1_objective": float(stage1_star),
            "stage2_objective": float(stage2_star),
            "stage3_objective": float(objective_stage3(res3.x, cfg)),
        },
    }


def input_composition_strict() -> np.ndarray:
    vals = []
    print("=" * 72)
    print("请输入六种垃圾的质量分数（小数形式），例如 0.2")
    print("输入顺序：菜叶、西瓜皮、橙子皮、肉、杂项混合、米饭")
    print("注意：6 项相加必须等于 1，否则程序会报错退出。")
    print("=" * 72)

    for name in NAMES:
        raw = input(f"请输入【{name}】的质量分数: ").strip()
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(f"{name} 的输入 '{raw}' 不是合法数字。")
        if v < 0:
            raise ValueError(f"{name} 的质量分数不能为负数。")
        vals.append(v)

    x = np.array(vals, dtype=float)
    total = x.sum()
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"六项质量分数之和为 {total:.10f}，不等于 1。")
    return x


def print_result(result: Dict[str, Any], T_avg_prev: float) -> None:
    x = result["x"]
    props = result["mixed_props"]
    outer = result["outer"]
    caps = result["resource_caps"]
    overall = result["overall"]
    opt = result["optimal"]
    stg = result["stages"]
    cfg = result["config"]

    print("\n" + "=" * 72)
    print("一、输入组成比例")
    for n, xi in zip(NAMES, x):
        print(f"{n:<8s}: {xi:.4f}")
    print(f"比例和: {x.sum():.4f}")

    print("\n" + "=" * 72)
    print("二、混合垃圾等效性质")
    print(f"omega0            = {props['omega0']:.4f} ({props['omega0']*100:.2f}%)")
    print(f"t_ref             = {props['tref']:.4f} min")
    print(f"slope             = {props['slope']:.4f} min/°C")
    print(f"c_eq              = {props['ceq']:.4f} kJ/(kg·K)")

    print("\n" + "=" * 72)
    print("三、外层反馈修正结果")
    print(f"上一时刻 T_avg_prev = {T_avg_prev:.4f} °C")
    print(f"稳态设定 T_set      = {cfg.T_SET:.4f} °C")
    print(f"误差 e_prev         = {outer['e_prev']:.4f} °C")
    print(f"估计 w_hat_prev     = {outer['w_hat_prev']:.4f} %")
    if 'T_tar_raw' in outer:
        print(f"未投影 T_tar_raw    = {outer['T_tar_raw']:.4f} °C")
    print(f"目标 T_tar^k        = {outer['T_tar_k']:.4f} °C")
    print(f"目标 w_tar^k        = {outer['w_tar_k']:.4f} %")

    print("\n" + "=" * 72)
    print("四、动态烟气资源上限")
    print(f"T_stack_cap        = {caps['T_stack_cap']:.4f} °C")
    print(f"v_stack_cap        = {caps['v_stack_cap']:.4f} m/s")
    print(f"mdot_stack_cap     = {caps['mdot_stack_cap']:.4f} kg/s")
    print(f"Tg 动态上界            = {caps['Tg_cap']:.4f} °C")

    print("\n" + "=" * 72)
    print("五、内层优化最优结果")
    print(f"Tg*                = {opt['Tg']:.4f} °C")
    print(f"vg*                = {opt['vg']:.4f} m/s")
    print(f"Tm*                = {opt['Tm']:.4f} °C")
    print(f"w_opt              = {opt['w_opt']:.4f} %")
    print(f"omega_opt          = {opt['omega_opt']:.4f}")

    print("\n" + "=" * 72)
    print("六、约束核验")
    print(f"tau20              = {opt['tau20_min']:.4f} min")
    print(f"tau_target         = {opt['tau_target_min']:.4f} min")
    print(f"tau_r              = {opt['tau_r_min']:.4f} min")
    print(f"Qsup               = {opt['Qsup_kW']:.4f} kW")
    print(f"Qreq               = {opt['Qreq_kW']:.4f} kW")
    print(f"Power              = {opt['Power_kW']:.4f} kW")
    print(f"mdot_preheater     = {opt['mdot_preheater']:.4f} kg/s")
    print(f"Tmin_burn          = {opt['Tmin_burn']:.4f} °C")
    print(f"Tavg_burn          = {opt['Tavg_burn']:.4f} °C")
    print(f"Tmax_burn          = {opt['Tmax_burn']:.4f} °C")
    print(f"sigma_burn         = {opt['sigma_burn']:.4f}")
    print(f"Tg - Tm            = {opt['deltaT']:.4f} °C")

    print("\n" + "=" * 72)
    print("七、目标跟踪偏差")
    print(f"d_w^-              = {opt['d_w_minus']:.6f}")
    print(f"d_w^+              = {opt['d_w_plus']:.6f}")

    print("\n" + "=" * 72)
    print("八、阶段求解信息")
    print(f"Stage1 success     = {stg['stage1_success']}")
    print(f"Stage2 success     = {stg['stage2_success']}")
    print(f"Stage3 success     = {stg['stage3_success']}")
    print(f"Stage1 obj         = {stg['stage1_objective']:.8f}")
    print(f"Stage2 obj         = {stg['stage2_objective']:.8f}")
    print(f"Stage3 obj         = {stg['stage3_objective']:.8f}")
    print(f"Max violation      = {overall['max_constraint_violation']:.8e}")
    print(f"Numerically feasible = {overall['numerically_feasible']}")


def main() -> None:
    cfg = Config()

    print("是否使用外层反馈自动生成 w_tar^k？")
    print("1 = 是（输入上一时刻平均炉温和 kappa）")
    print("2 = 否（手动输入 w_tar^k）")
    mode = input("请选择 [1/2]: ").strip()

    x = input_composition_strict()

    if mode == "1":
        raw_t = input("请输入上一时刻平均炉温 T_avg_prev (°C): ").strip()
        raw_k = input("请输入反馈增益 kappa（建议 0.2~0.8，例如 0.5）: ").strip()
        T_avg_prev = float(raw_t)
        kappa = float(raw_k)
        result = solve_inner_lexicographic(
            x=x,
            T_avg_prev=T_avg_prev,
            cfg=cfg,
            kappa=kappa,
            use_outer_feedback=True,
        )
    elif mode == "2":
        raw_t = input("请输入上一时刻平均炉温 T_avg_prev (°C)（用于烟气资源约束）: ").strip()
        raw_w = input("请输入外层给定的目标含水率 w_tar^k (%)：").strip()
        T_avg_prev = float(raw_t)
        w_tar_k = float(raw_w)
        result = solve_inner_lexicographic(
            x=x,
            T_avg_prev=T_avg_prev,
            cfg=cfg,
            w_tar_k=w_tar_k,
            use_outer_feedback=False,
        )
    else:
        raise ValueError("模式选择错误，请输入 1 或 2。")

    print_result(result, T_avg_prev)


if __name__ == "__main__":
    main()
