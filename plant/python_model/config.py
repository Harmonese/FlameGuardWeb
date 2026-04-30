from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class Config:
    T_REF: float = 175.0
    T0: float = 20.0
    T_AMB: float = 20.0
    MDOT_W: float = 20000.0 / 86400.0
    D: float = 1.2
    L: float = 3.2
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
    TG_SAFE_MAX: float = 2000.0
    TM_MAX: float = 250.0
    DELTA_T_MIN: float = 0.0
    OMEGA_MODEL_MIN: float = 0.05
    OMEGA_MODEL_MAX: float = 0.98
    TAVG_A: float = -13.109632
    TAVG_B: float = 1294.871365
    TMIN_A: float = -13.422320
    TMIN_B: float = 1330.692379
    TMAX_A: float = -16.072025
    TMAX_B: float = 1589.019616
    SIGMA_A: float = -0.189997
    SIGMA_B: float = 18.331239
    TMIN_BURN_MIN: float = 850.0
    TAVG_BURN_MIN: float = 850.0
    TAVG_BURN_MAX: float = 1100.0
    TMAX_BURN_MAX: float = 1100.0
    OMEGA_STEADY_MIN: float = 0.3042675804697915
    OMEGA_STEADY_MAX: float = 0.3393469511577442

    def __post_init__(self) -> None:
        # A_FLOW_EQ is the equivalent gas-flow area used by the one-channel
        # preheater model for mass flow. It is not itself a heat-transfer
        # surface. A_HEAT_EQ is the equivalent wall area available for heat
        # transfer from tubes/jacket to waste.
        self.A_D = math.pi * self.R_DUCT ** 2
        self.A_FLOW_EQ = self.A_D
        self.A_S = math.pi * self.R_STACK ** 2
        self.A = math.pi * self.D * self.L + self.N_TUBES * math.pi * self.DTUBE * self.L
        self.A_HEAT_EQ = self.A
        self.M_H = self.RHO_BULK * self.PHI * (math.pi * self.D ** 2 / 4.0) * self.L
        self.TAU_R_SEC = self.M_H / self.MDOT_W
        self.TAU_R_MIN = self.TAU_R_SEC / 60.0
