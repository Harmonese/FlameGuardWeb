import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize


# =========================
# 1) 基础配置
# =========================

@dataclass
class Config:
    # ===== 动力学与目标 =====
    T_REF: float = 175.0           # 参考温度 (degC)
    T0: float = 20.0               # 进炉垃圾温度 (degC)
    T_AMB: float = 20.0            # 环境温度 (degC)
    OMEGA_F: float = 0.20          # 目标终含水率 (湿基)
    P_TARGET: float = 307.7        # 能耗软目标 (kW)
    TG_TARGET: float = 175.0       # 目标烟气温度 (degC)

    # ===== 结构设计 =====
    D: float = 1.2                 # 主筒体内径 (m)
    L: float = 6.8                 # 有效长度 (m)
    PHI: float = 0.14              # 工作充填率
    DTUBE: float = 0.168           # 导烟管外径 (m)
    N_TUBES: int = 6               # 导烟管数量
    R_DUCT: float = 0.4            # 主烟道半径 (m)

    # ===== 滞留与处理量 =====
    RHO_BULK: float = 450.0        # 高湿垃圾堆积密度 (kg/m^3)
    MDOT_W: float = 20000.0 / 86400.0   # 湿垃圾连续进料流量 (kg/s)

    # ===== 传热模型 =====
    U0: float = 18.97              # U(v)=U0 + k*v^n, 单位与整体U相容，输出为 W/(m^2*K)
    K_U: float = 20.09
    N_U: float = 0.65

    # ===== 烟气物性 =====
    CPG: float = 1.05              # 烟气定压比热 kJ/(kg*K)
    RHO_G: float = 0.78            # 烟气密度 kg/m^3

    # ===== 垃圾热物性 =====
    CS: float = 1.70               # 干基垃圾比热 kJ/(kg*K)
    CW: float = 4.1844             # 水比热 kJ/(kg*K)
    LAMBDA: float = 2257.9         # 汽化潜热 kJ/kg

    # ===== 边界条件 =====
    VG_MIN: float = 3.0
    VG_MAX: float = 12.0
    TG_MIN: float = 100.0
    TG_MAX: float = 250.0
    TM_MAX: float = 110.0          
    DELTA_T_MIN: float = 0

    # ===== 数值求解参数 =====
    TOL: float = 1e-7
    MAXITER: int = 500

    def __post_init__(self):
        self.A_D = np.pi * self.R_DUCT**2
        self.A = np.pi * self.D * self.L + self.N_TUBES * np.pi * self.DTUBE * self.L
        self.M_H = self.RHO_BULK * self.PHI * (np.pi * self.D**2 / 4.0) * self.L
        self.TAU_R_MIN = self.M_H / self.MDOT_W / 60.0  # 转成 min


# 六类垃圾原始数据
OMEGA = np.array([0.948, 0.948, 0.817, 0.442, 0.773, 0.611], dtype=float)
TREF_TIME = np.array([12.1, 17.7, 15.3, 11.5, 16.3, 15.8], dtype=float)  # min
SLOPE = np.array([-0.132, -0.251, -0.189, -0.216, -0.210, -0.243], dtype=float)  # min/degC

# 变量索引
IDX_TG = 0
IDX_VG = 1
IDX_TM = 2
IDX_D1M = 3
IDX_D1P = 4
IDX_D2M = 5
IDX_D2P = 6
IDX_D3M = 7
IDX_D3P = 8


# =========================
# 2) 前处理与物理模型
# =========================

def validate_composition(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float).reshape(-1)
    if len(x) != 6:
        raise ValueError("x 必须是长度为 6 的数组，对应 [菜叶, 西瓜皮, 橙子皮, 肉, 杂项混合, 米饭]")
    if np.any(x < 0):
        raise ValueError("x_i 不能为负数")
    s = x.sum()
    if s <= 0:
        raise ValueError("x 的和必须大于 0")
    if abs(s - 1.0) > 1e-10:
        x = x / s
        print(f"[提示] 输入比例和不为 1，已自动归一化。归一化后 x = {x}")
    return x


def mixed_properties(x: np.ndarray, cfg: Config) -> Dict[str, float]:
    x = validate_composition(x)
    omega0 = float(np.dot(x, OMEGA))
    tref = float(np.dot(x, TREF_TIME))
    slope = float(np.dot(x, SLOPE))
    ceq = (1.0 - omega0) * cfg.CS + omega0 * cfg.CW
    return {
        "omega0": omega0,
        "tref": tref,
        "slope": slope,
        "ceq": ceq
    }


def tau20(Tm: float, x: np.ndarray, cfg: Config) -> float:
    props = mixed_properties(x, cfg)
    return props["tref"] + props["slope"] * (Tm - cfg.T_REF)


def q_sup_kW(Tg: float, vg: float, Tm: float, cfg: Config) -> float:
    """
    供热能力:
    Q_sup = (U0 + k*vg^n) * A * (Tg - Tm)
    U 的单位是 W/(m^2*K)，最终转 kW
    """
    U = cfg.U0 + cfg.K_U * (vg ** cfg.N_U)
    return U * cfg.A * (Tg - Tm) / 1000.0


def evap_water_per_kg_wet(omega0: float, omega_f: float) -> float:
    """
    每 kg 初始湿垃圾最终真正蒸发掉的水量
    基于湿基含水率定义推导:
    m_evap = (omega0 - omega_f) / (1 - omega_f)
    """
    if omega0 <= omega_f:
        return 0.0
    return (omega0 - omega_f) / (1.0 - omega_f)


def q_req_kW(Tm: float, x: np.ndarray, cfg: Config) -> float:
    """
    所需热负荷:
    mdot * [ ceq*(Tm - T0) + lambda*m_evap ]
    单位: kW = kJ/s
    """
    props = mixed_properties(x, cfg)
    omega0 = props["omega0"]
    ceq = props["ceq"]
    m_evap = evap_water_per_kg_wet(omega0, cfg.OMEGA_F)
    return cfg.MDOT_W * (ceq * (Tm - cfg.T0) + cfg.LAMBDA * m_evap)


def power_kW(Tg: float, vg: float, cfg: Config) -> float:
    """
    烟气侧功率:
    P = rho_g * c_pg * A_d * vg * (Tg - T_amb)
    这里 cpg 用 kJ/(kg*K)，因此结果单位是 kW
    """
    return cfg.RHO_G * cfg.CPG * cfg.A_D * vg * (Tg - cfg.T_AMB)


# =========================
# 3) 目标规划约束
# =========================

def eq_constraints(z: np.ndarray, x: np.ndarray, cfg: Config) -> np.ndarray:
    Tg, vg, Tm = z[IDX_TG], z[IDX_VG], z[IDX_TM]
    d1m, d1p = z[IDX_D1M], z[IDX_D1P]
    d2m, d2p = z[IDX_D2M], z[IDX_D2P]
    d3m, d3p = z[IDX_D3M], z[IDX_D3P]

    g1 = tau20(Tm, x, cfg) - cfg.TAU_R_MIN + d1m - d1p
    g2 = power_kW(Tg, vg, cfg) - cfg.P_TARGET + d2m - d2p
    g3 = Tg - cfg.TG_TARGET + d3m - d3p
    return np.array([g1, g2, g3], dtype=float)


def ineq_constraints(z: np.ndarray, x: np.ndarray, cfg: Config) -> np.ndarray:
    Tg, vg, Tm = z[IDX_TG], z[IDX_VG], z[IDX_TM]

    g_heat = q_sup_kW(Tg, vg, Tm, cfg) - q_req_kW(Tm, x, cfg)
    g_dt = Tg - Tm - cfg.DELTA_T_MIN
    return np.array([g_heat, g_dt], dtype=float)


def bounds(cfg: Config) -> List[Tuple[float, float]]:
    big = 1e6
    return [
        (cfg.TG_MIN, cfg.TG_MAX),  # Tg
        (cfg.VG_MIN, cfg.VG_MAX),  # vg
        (cfg.T0, cfg.TM_MAX),      # Tm
        (0.0, big),                # d1-
        (0.0, big),                # d1+
        (0.0, big),                # d2-
        (0.0, big),                # d2+
        (0.0, big),                # d3-
        (0.0, big),                # d3+
    ]


def split_for_goal_equation(lhs_minus_target: float) -> Tuple[float, float]:
    """
    对 y + d^- - d^+ = 0
    若 y > 0 -> d^+ = y
    若 y < 0 -> d^- = -y
    """
    if lhs_minus_target > 0:
        return 0.0, lhs_minus_target
    else:
        return -lhs_minus_target, 0.0


def build_initial_guess(Tg: float, vg: float, Tm: float, x: np.ndarray, cfg: Config) -> np.ndarray:
    y1 = tau20(Tm, x, cfg) - cfg.TAU_R_MIN
    d1m, d1p = split_for_goal_equation(y1)

    y2 = power_kW(Tg, vg, cfg) - cfg.P_TARGET
    d2m, d2p = split_for_goal_equation(y2)

    y3 = Tg - cfg.TG_TARGET
    d3m, d3p = split_for_goal_equation(y3)

    return np.array([Tg, vg, Tm, d1m, d1p, d2m, d2p, d3m, d3p], dtype=float)


def initial_guesses(x: np.ndarray, cfg: Config) -> List[np.ndarray]:
    seeds = []
    Tg_list = [120.0, 150.0, 175.0, 210.0, 240.0]
    vg_list = [3.2, 5.0, 8.0, 11.0]
    Tm_list = [50.0, 70.0, 85.0, 95.0]

    for Tg in Tg_list:
        for vg in vg_list:
            for Tm in Tm_list:
                if Tm <= cfg.TM_MAX and Tg - Tm >= cfg.DELTA_T_MIN:
                    seeds.append(build_initial_guess(Tg, vg, Tm, x, cfg))

    return seeds


def max_constraint_violation(z: np.ndarray, x: np.ndarray, cfg: Config, extra_ineqs=None) -> float:
    eqv = np.max(np.abs(eq_constraints(z, x, cfg)))
    ineqv = np.max(np.maximum(-ineq_constraints(z, x, cfg), 0.0))
    extra = 0.0
    if extra_ineqs is not None:
        vals = np.array([f(z) for f in extra_ineqs], dtype=float)
        if len(vals) > 0:
            extra = np.max(np.maximum(-vals, 0.0))
    return max(eqv, ineqv, extra)


def solve_stage(
    x: np.ndarray,
    cfg: Config,
    objective,
    extra_ineqs=None,
) -> Any:
    extra_ineqs = extra_ineqs or []

    cons = [
        {"type": "eq", "fun": lambda z: eq_constraints(z, x, cfg)[0]},
        {"type": "eq", "fun": lambda z: eq_constraints(z, x, cfg)[1]},
        {"type": "eq", "fun": lambda z: eq_constraints(z, x, cfg)[2]},
        {"type": "ineq", "fun": lambda z: ineq_constraints(z, x, cfg)[0]},
        {"type": "ineq", "fun": lambda z: ineq_constraints(z, x, cfg)[1]},
    ]

    for f in extra_ineqs:
        cons.append({"type": "ineq", "fun": f})

    best_res = None
    best_obj = np.inf

    for x0 in initial_guesses(x, cfg):
        try:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds(cfg),
                constraints=cons,
                options={"maxiter": cfg.MAXITER, "ftol": 1e-10, "disp": False},
            )
        except Exception:
            continue

        if res.x is None:
            continue

        vio = max_constraint_violation(res.x, x, cfg, extra_ineqs=extra_ineqs)
        obj = objective(res.x)

        if vio <= 1e-5 and obj < best_obj:
            best_obj = obj
            best_res = res
        elif best_res is None:
            # 先保留一个最好的近似解
            if res.success or vio < 1e-2:
                best_obj = obj
                best_res = res

    if best_res is None:
        raise RuntimeError("求解失败：未找到可行解。建议检查组成比例或放宽约束。")

    return best_res


def solve_lexicographic(x: np.ndarray, cfg: Config) -> Dict[str, Any]:
    x = validate_composition(x)

    # ---- 阶段 1：最小化 d1+
    obj1 = lambda z: z[IDX_D1P] + 1e-8 * (z[IDX_TG] + z[IDX_VG] + z[IDX_TM])
    res1 = solve_stage(x, cfg, obj1)
    d1p_star = max(0.0, res1.x[IDX_D1P])

    # ---- 阶段 2：在 d1+ 最优基础上最小化 d2+
    extra2 = [
        lambda z, cap=d1p_star + 1e-6: cap - z[IDX_D1P],
    ]
    obj2 = lambda z: z[IDX_D2P] + 1e-8 * (z[IDX_TG] + z[IDX_VG] + z[IDX_TM])
    res2 = solve_stage(x, cfg, obj2, extra_ineqs=extra2)
    d2p_star = max(0.0, res2.x[IDX_D2P])

    # ---- 阶段 3：在 d1+, d2+ 最优基础上最小化 d3+
    extra3 = [
        lambda z, cap=d1p_star + 1e-6: cap - z[IDX_D1P],
        lambda z, cap=d2p_star + 1e-6: cap - z[IDX_D2P],
    ]
    obj3 = lambda z: z[IDX_D3P]
    res3 = solve_stage(x, cfg, obj3, extra_ineqs=extra3)

    z = res3.x
    props = mixed_properties(x, cfg)

    result = {
        "x": x,
        "props": props,
        "stage1": res1,
        "stage2": res2,
        "stage3": res3,
        "optimal": {
            "Tg": z[IDX_TG],
            "vg": z[IDX_VG],
            "Tm": z[IDX_TM],
            "d1_minus": z[IDX_D1M],
            "d1_plus": z[IDX_D1P],
            "d2_minus": z[IDX_D2M],
            "d2_plus": z[IDX_D2P],
            "d3_minus": z[IDX_D3M],
            "d3_plus": z[IDX_D3P],
            "tau20_min": tau20(z[IDX_TM], x, cfg),
            "tau_r_min": cfg.TAU_R_MIN,
            "Qsup_kW": q_sup_kW(z[IDX_TG], z[IDX_VG], z[IDX_TM], cfg),
            "Qreq_kW": q_req_kW(z[IDX_TM], x, cfg),
            "Power_kW": power_kW(z[IDX_TG], z[IDX_VG], cfg),
        }
    }
    return result


# =========================
# 4) 输出
# =========================

def print_result(result: Dict[str, Any], cfg: Config) -> None:
    x = result["x"]
    props = result["props"]
    opt = result["optimal"]

    names = ["菜叶", "西瓜皮", "橙子皮", "肉", "杂项混合", "米饭"]

    print("=" * 70)
    print("一、输入组成比例")
    for n, xi in zip(names, x):
        print(f"{n:<8s}: {xi:.4f}")
    print(f"比例和: {x.sum():.4f}")

    print("\n" + "=" * 70)
    print("二、混合垃圾等效性质")
    print(f"混合初始含水率 omega0      = {props['omega0']:.4f} ({props['omega0']*100:.2f}%)")
    print(f"混合参考干燥时间 tref      = {props['tref']:.4f} min")
    print(f"混合温度敏感性 slope       = {props['slope']:.4f} min/°C")
    print(f"混合等效比热 ceq           = {props['ceq']:.4f} kJ/(kg·K)")

    print("\n" + "=" * 70)
    print("三、结构与固定参数")
    print(f"A_d  = {cfg.A_D:.4f} m^2")
    print(f"A    = {cfg.A:.4f} m^2")
    print(f"M_h  = {cfg.M_H:.4f} kg")
    print(f"tau_r= {cfg.TAU_R_MIN:.4f} min")
    print(f"P*   = {cfg.P_TARGET:.4f} kW")
    print(f"Tg*  = {cfg.TG_TARGET:.4f} °C")

    print("\n" + "=" * 70)
    print("四、最优工况")
    print(f"Tg 最优烟气入口温度        = {opt['Tg']:.4f} °C")
    print(f"vg 最优烟气流速            = {opt['vg']:.4f} m/s")
    print(f"Tm 最优垃圾平均温度        = {opt['Tm']:.4f} °C")

    print("\n" + "=" * 70)
    print("五、约束核验")
    print(f"tau20(Tm,x)                = {opt['tau20_min']:.4f} min")
    print(f"tau_r                      = {opt['tau_r_min']:.4f} min")
    print(f"Qsup                       = {opt['Qsup_kW']:.4f} kW")
    print(f"Qreq                       = {opt['Qreq_kW']:.4f} kW")
    print(f"Power                      = {opt['Power_kW']:.4f} kW")
    print(f"Tg - Tm                    = {opt['Tg'] - opt['Tm']:.4f} °C")

    print("\n" + "=" * 70)
    print("六、目标规划偏差变量")
    print(f"d1- = {opt['d1_minus']:.6f}, d1+ = {opt['d1_plus']:.6f}")
    print(f"d2- = {opt['d2_minus']:.6f}, d2+ = {opt['d2_plus']:.6f}")
    print(f"d3- = {opt['d3_minus']:.6f}, d3+ = {opt['d3_plus']:.6f}")

    print("\n" + "=" * 70)
    if opt["d1_plus"] <= 1e-5:
        print("结论：在当前模型与参数下，出口含水率达到 20% 的动力学约束可满足。")
    else:
        print("结论：在当前模型与参数下，出口含水率 20% 约束未完全满足，需要调整结构或放宽边界。")

def input_composition_strict() -> np.ndarray:
    """
    逐项输入六组分比例。
    要求:
    1. 每项都必须是数字
    2. 每项都不能为负
    3. 六项之和必须等于 1（允许极小浮点误差）
    """
    names = ["菜叶", "西瓜皮", "橙子皮", "肉", "杂项混合", "米饭"]
    values = []

    print("=" * 70)
    print("请输入六种垃圾的质量分数（小数形式），例如 0.2")
    print("输入顺序：菜叶、西瓜皮、橙子皮、肉、杂项混合、米饭")
    print("注意：6 项相加必须等于 1，否则程序会报错退出。")
    print("=" * 70)

    for name in names:
        raw = input(f"请输入【{name}】的质量分数: ").strip()
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"输入错误：{name} 的值 '{raw}' 不是合法数字。")

        if value < 0:
            raise ValueError(f"输入错误：{name} 的质量分数不能为负数。")

        values.append(value)

    x = np.array(values, dtype=float)
    total = x.sum()

    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"输入错误：六项质量分数之和为 {total:.10f}，不等于 1。"
            "请重新运行程序并修改输入。"
        )

    return x

# =========================
# 5) 主程序
# =========================

if __name__ == "__main__":
    cfg = Config()

    try:
        x = input_composition_strict()
        result = solve_lexicographic(x, cfg)
        print_result(result, cfg)

    except Exception as e:
        print("\n程序运行失败：")
        print(e)