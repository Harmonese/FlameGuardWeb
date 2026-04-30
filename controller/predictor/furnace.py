from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Mapping, Any
import copy
import math
import numpy as np

from domain.types import FurnaceFeed


# COMSOL steady-state surrogate fitted from scripts/data/furnace_static_comsol.
# The COMSOL CSV temperature outputs are in K; these coefficients emit degC.
# Inputs are:
#   dry_basis_ratio: rd column in the COMSOL sweep, clipped to [0.5, 3.5]
#   omega_b: furnace-inlet wet-basis moisture fraction, i.e. w_b / 100,
#            clipped to [0.0, 0.5]
_SURROGATE_RD_MIN = 0.5
_SURROGATE_RD_MAX = 3.5
_SURROGATE_OMEGA_B_MIN = 0.0
_SURROGATE_OMEGA_B_MAX = 0.5
_SURROGATE_RD_CENTER = 3.0
_SURROGATE_OMEGA_B_CENTER = 0.3218
_SURROGATE_DEFAULT_RD = 1.0
_SURROGATE_MDOT_D_REF_KGPS = 0.052
_SURROGATE_POWERS = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)]
_SURROGATE_COEFFS = {
    'T_avg_C': [852.4426190424261, -849.4807268334719, -1166.6124634869277, -1126.948691749353, 2.467298604268791, -1.5042698840226745, -8.830479773611199, 2.3082232875178192, 4.211853360097635, 2.041410086216956],
    'T_stack_C': [932.4401791755217, -919.2539761151407, -1269.3426543588164, -1196.7957386216133, 23.53533605134413, -22.746424874931943, -56.042554428368184, 4.588004315773162, 16.950941326187657, 7.240567036126322],
    'v_stack_mps': [17.104236410296128, -9.067246251499252, -14.051240553414893, -11.19617148189401, 5.990995227668757, -3.4615809888105913, -4.076329409518369, 0.00519281497203572, -0.06868452314472803, -0.02451640588719073],
    'T_surface_min_C': [864.4663301993795, -834.1399029598682, -1116.9735301755027, -960.4952330470502, 66.61338787813057, -51.52293625332101, -115.75260644234278, 0.4820353378259483, 25.24842316470899, 10.417026818380236],
    'T_surface_max_C': [1047.3816985917608, -1039.730409136574, -1429.0460920781084, -1384.7156372570716, 1.3857227579539995, -0.4582054444157997, -4.564802259658642, 0.4667162180268747, 1.1082051767737298, 0.6397070367426068],
    'T_surface_std_C': [13.539449481579085, -13.988511947088902, -22.009564639461686, -34.88327876362528, -5.138188647226297, 3.6585610709271394, 15.64318911475531, -2.324137716188889, -5.757876875969796, -2.6891746287676472],
}


@dataclass(frozen=True)
class FurnaceStaticOutputs:
    T_avg_C: float
    T_stack_C: float
    v_stack_mps: float
    T_surface_min_C: float
    T_surface_max_C: float
    T_surface_std_C: float


def _finite_or_default(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) else float(default)


def dry_basis_ratio_from_feedstock(feedstock: Any, *, default: float = _SURROGATE_DEFAULT_RD) -> float:
    """Extract COMSOL rd/dry-basis-ratio metadata from a FeedstockObservation.

    Existing scenarios do not yet provide rd, so this intentionally falls back to
    the physical baseline dry-basis load rd=1.0.  New feed characterization code can pass one
    of these keys in feedstock.raw: dry_basis_ratio, rd, r_d, dry_ratio.
    """
    raw = getattr(feedstock, "raw", None)
    if isinstance(raw, Mapping):
        for key in ("dry_basis_ratio", "rd", "r_d", "dry_ratio"):
            if key in raw:
                return _finite_or_default(raw.get(key), default)
    return float(default)



def dry_basis_ratio_from_mdot_d(mdot_d_kgps: float | None, *, mdot_d_ref_kgps: float = _SURROGATE_MDOT_D_REF_KGPS, default: float = _SURROGATE_DEFAULT_RD) -> float:
    """Convert engineering dry-solid load to COMSOL rd coordinate."""
    if mdot_d_kgps is None:
        return float(default)
    ref = max(_finite_or_default(mdot_d_ref_kgps, _SURROGATE_MDOT_D_REF_KGPS), 1e-12)
    return _finite_or_default(mdot_d_kgps, ref * default) / ref


def furnace_feed_from_preheater_output(
    *,
    time_s: float,
    omega_b: float,
    mdot_d_kgps: float | None,
    mdot_water_kgps: float | None = None,
    mdot_wet_kgps: float | None = None,
    mdot_d_ref_kgps: float = _SURROGATE_MDOT_D_REF_KGPS,
) -> FurnaceFeed:
    """Build a mass-flow based furnace feed and derived rd diagnostic."""
    omega = float(np.clip(_finite_or_default(omega_b, _SURROGATE_OMEGA_B_CENTER), 0.0, 0.999999))
    if mdot_d_kgps is None:
        if mdot_wet_kgps is not None:
            mdot_d = max(0.0, float(mdot_wet_kgps) * (1.0 - omega))
        elif mdot_water_kgps is not None:
            mdot_d = max(0.0, float(mdot_water_kgps) * (1.0 - omega) / max(omega, 1e-12))
        else:
            mdot_d = _SURROGATE_MDOT_D_REF_KGPS
    else:
        mdot_d = max(0.0, float(mdot_d_kgps))
    if mdot_water_kgps is None:
        mdot_water = mdot_d * omega / max(1.0 - omega, 1e-12)
    else:
        mdot_water = max(0.0, float(mdot_water_kgps))
    if mdot_wet_kgps is None:
        mdot_wet = mdot_d + mdot_water
    else:
        mdot_wet = max(0.0, float(mdot_wet_kgps))
    rd = dry_basis_ratio_from_mdot_d(mdot_d, mdot_d_ref_kgps=mdot_d_ref_kgps)
    return FurnaceFeed(
        time_s=float(time_s),
        omega_b=omega,
        mdot_d_kgps=float(mdot_d),
        mdot_water_kgps=float(mdot_water),
        mdot_wet_kgps=float(mdot_wet),
        rd=float(rd),
    )


def _clip_inputs(omega_b: float, dry_basis_ratio: float) -> tuple[float, float]:
    omega_b_c = float(np.clip(_finite_or_default(omega_b, _SURROGATE_OMEGA_B_CENTER), _SURROGATE_OMEGA_B_MIN, _SURROGATE_OMEGA_B_MAX))
    rd_c = float(np.clip(_finite_or_default(dry_basis_ratio, _SURROGATE_DEFAULT_RD), _SURROGATE_RD_MIN, _SURROGATE_RD_MAX))
    return omega_b_c, rd_c


def _surrogate_value(name: str, omega_b: float, dry_basis_ratio: float) -> float:
    omega_b_c, rd_c = _clip_inputs(omega_b, dry_basis_ratio)
    x = rd_c - _SURROGATE_RD_CENTER
    z = omega_b_c - _SURROGATE_OMEGA_B_CENTER
    coeffs = _SURROGATE_COEFFS[name]
    value = 0.0
    for coeff, (i, j) in zip(coeffs, _SURROGATE_POWERS):
        value += coeff * (x ** i) * (z ** j)
    return float(value)


def furnace_static_outputs_from_inputs(
    omega_b: float,
    dry_basis_ratio: float = _SURROGATE_DEFAULT_RD,
) -> FurnaceStaticOutputs:
    """Return steady furnace outputs from the COMSOL static surrogate.

    Parameters
    ----------
    omega_b:
        Furnace-inlet wet-basis moisture fraction, i.e. COMSOL w_b / 100.
    dry_basis_ratio:
        COMSOL rd sweep variable.  Values outside the fitted sweep are clipped.
    """
    return FurnaceStaticOutputs(
        T_avg_C=_surrogate_value('T_avg_C', omega_b, dry_basis_ratio),
        T_stack_C=_surrogate_value('T_stack_C', omega_b, dry_basis_ratio),
        v_stack_mps=_surrogate_value('v_stack_mps', omega_b, dry_basis_ratio),
        T_surface_min_C=_surrogate_value('T_surface_min_C', omega_b, dry_basis_ratio),
        T_surface_max_C=_surrogate_value('T_surface_max_C', omega_b, dry_basis_ratio),
        T_surface_std_C=max(0.0, _surrogate_value('T_surface_std_C', omega_b, dry_basis_ratio)),
    )


def furnace_outputs_from_omega_b(omega_b: float, dry_basis_ratio: float = _SURROGATE_DEFAULT_RD):
    """Return (T_avg_C, T_stack_C, v_stack_mps) for furnace-inlet omega_b."""
    out = furnace_static_outputs_from_inputs(omega_b, dry_basis_ratio)
    return out.T_avg_C, out.T_stack_C, out.v_stack_mps


def furnace_outputs_from_omega_b_and_mdot_d(omega_b: float, mdot_d_kgps: float, *, mdot_d_ref_kgps: float = _SURROGATE_MDOT_D_REF_KGPS):
    """Return outputs from engineering dry-solid mass flow and wet-basis moisture."""
    rd = dry_basis_ratio_from_mdot_d(mdot_d_kgps, mdot_d_ref_kgps=mdot_d_ref_kgps)
    return furnace_outputs_from_omega_b(omega_b, rd)


def furnace_outputs_from_omega(omega: float, dry_basis_ratio: float = _SURROGATE_DEFAULT_RD):
    """Backward-compatible alias; omega is furnace-inlet wet-basis omega_b."""
    return furnace_outputs_from_omega_b(omega, dry_basis_ratio)


@dataclass
class FurnaceDynConfig:
    # Backward-compatible name; this is furnace-inlet wet-basis omega_b.
    omega_ref: float = 0.3218
    dry_basis_ratio_ref: float = _SURROGATE_DEFAULT_RD
    mdot_d_ref_kgps: float = _SURROGATE_MDOT_D_REF_KGPS
    dead_s: float = 5.0
    tau1_s: float = 0.223
    tau2_s: float = 75.412
    dt_s: float = 0.1


class FurnaceDyn:
    """Two-lag + dead-time furnace dynamic proxy around a COMSOL static map.

    The steady target comes from the fitted COMSOL surrogate as a function of
    moisture and dry-basis ratio.  The two-lag dynamics are retained from the
    previous low-order proxy so plant and controller behavior remains dynamic
    rather than instantly jumping to the static COMSOL surface.
    """

    def __init__(self, cfg: FurnaceDynConfig | None = None):
        self.cfg = cfg or FurnaceDynConfig()
        self.delay_steps = max(1, round(self.cfg.dead_s / self.cfg.dt_s))
        self.queue: Deque[float] = deque([self.cfg.omega_ref] * self.delay_steps, maxlen=self.delay_steps)
        t_ref, ts_ref, vs_ref = furnace_outputs_from_omega_b(self.cfg.omega_ref, self.cfg.dry_basis_ratio_ref)
        self.refs = {'T_avg': t_ref, 'T_stack': ts_ref, 'v_stack': vs_ref}
        self.states = {'T_avg': (0.0, 0.0), 'T_stack': (0.0, 0.0), 'v_stack': (0.0, 0.0)}
        self._last_dry_basis_ratio = float(self.cfg.dry_basis_ratio_ref)

    def clone(self) -> "FurnaceDyn":
        return copy.deepcopy(self)

    def initialize_outputs(self, T_avg_init_C: float, T_stack_init_C: float, v_stack_init_mps: float):
        targets = {'T_avg': float(T_avg_init_C), 'T_stack': float(T_stack_init_C), 'v_stack': float(v_stack_init_mps)}
        for key in ['T_avg', 'T_stack', 'v_stack']:
            delta = targets[key] - self.refs[key]
            self.states[key] = (delta, delta)

    def initialize_from_omega(self, omega_init: float, dry_basis_ratio: float | None = None, mdot_d_kgps: float | None = None) -> None:
        """Initialize from furnace-inlet wet-basis omega_b.

        The method name is kept for compatibility with existing runtime code;
        omega_init is not rd and is not a dry-basis quantity.
        """
        omega_b = float(omega_init)
        rd = (dry_basis_ratio_from_mdot_d(mdot_d_kgps, mdot_d_ref_kgps=self.cfg.mdot_d_ref_kgps, default=self.cfg.dry_basis_ratio_ref)
              if dry_basis_ratio is None else float(dry_basis_ratio))
        self._last_dry_basis_ratio = rd
        self.queue = deque([omega_b] * self.delay_steps, maxlen=self.delay_steps)
        T_avg, T_stack, v_stack = furnace_outputs_from_omega_b(omega_b, rd)
        self.initialize_outputs(T_avg, T_stack, v_stack)

    def _delayed_omega_b(self, omega_b_in: float, dt: float) -> float:
        """Return delayed furnace-inlet wet-basis moisture for this step.

        For plant integration (dt == cfg.dt_s), use the persistent queue. For
        coarse MPC rollout steps, approximate the physical dead time in seconds,
        not in the plant number of 0.1 s samples.
        """
        omega_b_in = float(omega_b_in)
        if abs(dt - self.cfg.dt_s) <= 1e-12:
            self.queue.append(omega_b_in)
            return float(self.queue[0])

        if dt >= self.cfg.dead_s:
            return omega_b_in

        desired_len = max(2, int(np.ceil(self.cfg.dead_s / max(dt, 1e-9))) + 1)
        recent = list(self.queue)[-desired_len:] if self.queue else [self.cfg.omega_ref]
        while len(recent) < desired_len:
            recent.insert(0, recent[0])
        local_queue: Deque[float] = deque(recent, maxlen=desired_len)
        local_queue.append(omega_b_in)
        omega_b_d = float(local_queue[0])

        preserved = list(local_queue)[-self.delay_steps:]
        if len(preserved) < self.delay_steps:
            preserved = [preserved[0]] * (self.delay_steps - len(preserved)) + preserved
        self.queue = deque(preserved, maxlen=self.delay_steps)
        return omega_b_d

    def step(
        self,
        omega_b_in: float,
        *,
        dry_basis_ratio: float | None = None,
        mdot_d_kgps: float | None = None,
        dt_s: float | None = None,
        disturbance=None,
    ):
        dt = float(dt_s if dt_s is not None else self.cfg.dt_s)
        rd = (dry_basis_ratio_from_mdot_d(mdot_d_kgps, mdot_d_ref_kgps=self.cfg.mdot_d_ref_kgps, default=self._last_dry_basis_ratio)
              if dry_basis_ratio is None else float(dry_basis_ratio))
        self._last_dry_basis_ratio = rd
        omega_b_d = self._delayed_omega_b(float(omega_b_in), dt)
        steady = furnace_static_outputs_from_inputs(omega_b_d, rd)
        targets = {
            'T_avg': steady.T_avg_C,
            'T_stack': steady.T_stack_C,
            'v_stack': steady.v_stack_mps,
        }
        outputs = {}
        for key in ['T_avg', 'T_stack', 'v_stack']:
            x1, x2 = self.states[key]
            u = targets[key] - self.refs[key]
            a1 = np.exp(-dt / max(self.cfg.tau1_s, 1e-9))
            x1 = a1 * x1 + (1.0 - a1) * u
            a2 = np.exp(-dt / max(self.cfg.tau2_s, 1e-9))
            x2 = a2 * x2 + (1.0 - a2) * x1
            self.states[key] = (x1, x2)
            outputs[key] = self.refs[key] + x2
        if disturbance is None:
            d_avg, d_stack, d_v = 0.0, 0.0, 0.0
        elif isinstance(disturbance, (tuple, list)) and len(disturbance) == 3:
            d_avg, d_stack, d_v = float(disturbance[0]), float(disturbance[1]), float(disturbance[2])
        else:
            d_avg, d_stack, d_v = float(disturbance), 0.0, 0.0
        return outputs['T_avg'] + d_avg, outputs['T_stack'] + d_stack, outputs['v_stack'] + d_v
