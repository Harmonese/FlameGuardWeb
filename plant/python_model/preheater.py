from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import numpy as np

from .material_model import composition_to_equivalent_properties, properties_from_feedstock
from domain.types import FeedstockObservation, PreheaterCellState, PreheaterState, PreheaterDiagnostics
from .config import Config
from domain.types import EquivalentProperties
from .physics import rho_g


@dataclass
class PreheaterForwardConfig:
    n_cells: int = 20
    length_m: float = 3.2
    tau_residence_s: float = 985.0
    feed_delay_s: float = 5.0
    omega_min: float = 0.05
    omega_max: float = 0.98
    T_amb_C: float = 20.0
    T_evap_C: float = 100.0
    T_solid_max_C: float = 250.0
    opt_cfg: Config | None = None

    # Single equivalent gas channel.  The flow area is the equivalent area used
    # for mass flow, while the heat area is the wall area available for transfer.
    gas_flow_direction: str = "counter_current"  # counter_current or co_current
    gas_flow_area_m2: float | None = None
    gas_heat_area_m2: float | None = None

    # Transfer/drying tuning.
    heat_transfer_factor: float = 1.0
    min_tau20_min: float = 1.0
    evaporation_heat_fraction: float = 0.85


class PreheaterForwardModel:
    """Distributed preheater forward model.

    This is a one-channel equivalent gas model.  Waste moves from cell 0 to
    cell N-1.  By default the gas is counter-current: it enters at cell N-1,
    exchanges heat cell-by-cell, and exits near cell 0.  Moisture is represented
    by dry/water masses, so evaporation reduces cell mass and wet-basis moisture
    is derived from those masses.
    """

    def __init__(self, cfg: PreheaterForwardConfig | None = None):
        self.cfg = cfg or PreheaterForwardConfig()
        self.opt_cfg = self.cfg.opt_cfg or Config()
        self.time_s = 0.0
        n = self.cfg.n_cells
        self.T_solid = np.full(n, 120.0, dtype=float)
        self.omega0 = np.full(n, 0.65, dtype=float)
        self.tref = np.full(n, 15.0, dtype=float)
        self.slope = np.full(n, -0.21, dtype=float)
        self.bulk_density = np.full(n, self.opt_cfg.RHO_BULK, dtype=float)
        self.dry_mass = np.zeros(n, dtype=float)
        self.water_mass = np.zeros(n, dtype=float)
        self.omega = np.full(n, 0.3218, dtype=float)
        self._set_all_cell_masses_from_omega(0.3218)
        self._last_Tg_profile = tuple([175.0] * (n + 1))
        self.last_diagnostics = PreheaterDiagnostics()
        self._feed_delay: list[tuple[float, EquivalentProperties]] = []
        self._feed_delay_time_s = 0.0

    def clone(self) -> "PreheaterForwardModel":
        return copy.deepcopy(self)

    @property
    def tau_cell_s(self) -> float:
        return self.cfg.tau_residence_s / max(self.cfg.n_cells, 1)

    @property
    def nominal_cell_mass_kg(self) -> float:
        return self.opt_cfg.M_H / max(self.cfg.n_cells, 1)

    @property
    def cell_mass_kg(self) -> float:
        # Compatibility name for callers that only need the nominal inventory.
        return self.nominal_cell_mass_kg

    @property
    def gas_flow_area_m2(self) -> float:
        return float(self.cfg.gas_flow_area_m2 if self.cfg.gas_flow_area_m2 is not None else self.opt_cfg.A_FLOW_EQ)

    @property
    def gas_heat_area_m2(self) -> float:
        return float(self.cfg.gas_heat_area_m2 if self.cfg.gas_heat_area_m2 is not None else self.opt_cfg.A_HEAT_EQ)

    @property
    def cell_heat_area_m2(self) -> float:
        return self.gas_heat_area_m2 / max(self.cfg.n_cells, 1)

    @staticmethod
    def _water_mass_for_omega(dry_mass_kg: float, omega: float) -> float:
        omega = min(max(float(omega), 0.0), 0.999999)
        dry = max(float(dry_mass_kg), 0.0)
        return omega * dry / max(1.0 - omega, 1e-12)

    def _cell_mass_from_density(self, bulk_density_kg_m3: float | None) -> float:
        density = self.opt_cfg.RHO_BULK if bulk_density_kg_m3 is None else float(bulk_density_kg_m3)
        density = max(density, 1e-9)
        return max(self.nominal_cell_mass_kg * density / max(self.opt_cfg.RHO_BULK, 1e-9), 1e-9)

    def _cell_mass_from_feed_props(self, props: EquivalentProperties) -> float:
        # Prefer explicit engineering mass flow.  With fixed conveyor speed this
        # is the stable interface to the furnace load; density remains a fallback
        # for legacy tests and early feed-characterization experiments.
        if props.wet_mass_flow_kgps is not None:
            return max(float(props.wet_mass_flow_kgps) * self.tau_cell_s, 1e-9)
        return self._cell_mass_from_density(props.bulk_density_kg_m3)

    def _set_all_cell_masses_from_omega(self, omega: float, *, cell_mass_kg: float | None = None) -> None:
        omega = min(max(float(omega), 0.0), self.cfg.omega_max)
        mass = max(self.nominal_cell_mass_kg if cell_mass_kg is None else float(cell_mass_kg), 1e-9)
        self.water_mass[:] = omega * mass
        self.dry_mass[:] = (1.0 - omega) * mass
        self._sync_omega_from_masses()

    def _sync_omega_from_masses(self) -> None:
        self.dry_mass[:] = np.maximum(self.dry_mass, 1e-12)
        self.water_mass[:] = np.maximum(self.water_mass, 0.0)
        # Keep the wet-basis value inside the model's supported upper range by
        # reducing water mass if numerical mixing creates an excessive value.
        max_water = self._water_mass_for_omega(1.0, self.cfg.omega_max) * self.dry_mass
        self.water_mass[:] = np.minimum(self.water_mass, max_water)
        total = np.maximum(self.dry_mass + self.water_mass, 1e-12)
        self.omega[:] = self.water_mass / total

    def _cell_heat_capacity_kJ_per_K(self, i: int) -> float:
        return max(float(self.dry_mass[i]) * self.opt_cfg.CS + float(self.water_mass[i]) * self.opt_cfg.CW, 1e-9)

    def _cell_total_mass(self, i: int) -> float:
        return max(float(self.dry_mass[i] + self.water_mass[i]), 1e-12)

    def initialize(
        self,
        feedstock_or_composition,
        *,
        normalize: bool = False,
        omega_init: float | None = None,
        T_solid_init_C: float = 120.0,
        time_s: float = 0.0,
    ) -> None:
        if isinstance(feedstock_or_composition, FeedstockObservation):
            eq = properties_from_feedstock(feedstock_or_composition)
        else:
            # Legacy/test adapter path. Runtime should characterize composition
            # before it reaches the PlantBackend protocol.
            eq = composition_to_equivalent_properties(feedstock_or_composition, normalize=normalize).equivalent
        self.time_s = float(time_s)
        self.omega0[:] = eq.omega0
        self.tref[:] = eq.tref_min
        self.slope[:] = eq.slope_min_per_c
        self.bulk_density[:] = self.opt_cfg.RHO_BULK if eq.bulk_density_kg_m3 is None else eq.bulk_density_kg_m3
        omega = float(eq.omega0 if omega_init is None else omega_init)
        omega = float(np.clip(omega, self.cfg.omega_min, self.cfg.omega_max))
        self._set_all_cell_masses_from_omega(omega, cell_mass_kg=self._cell_mass_from_feed_props(eq))
        self.T_solid[:] = float(T_solid_init_C)
        self._feed_delay = [
            (self.time_s - max(self.cfg.feed_delay_s, 0.0) - 1e-6, eq),
            (self.time_s, eq),
        ]
        self._feed_delay_time_s = 0.0

    def _equivalent_from_feed(self, feedstock: FeedstockObservation) -> EquivalentProperties:
        return properties_from_feedstock(feedstock)

    @staticmethod
    def _interp_equivalent(a: EquivalentProperties, b: EquivalentProperties, w: float) -> EquivalentProperties:
        w = min(max(float(w), 0.0), 1.0)
        if a.bulk_density_kg_m3 is None and b.bulk_density_kg_m3 is None:
            density = None
        else:
            da = a.bulk_density_kg_m3 if a.bulk_density_kg_m3 is not None else b.bulk_density_kg_m3
            db = b.bulk_density_kg_m3 if b.bulk_density_kg_m3 is not None else da
            density = (1.0 - w) * float(da) + w * float(db)
        if a.wet_mass_flow_kgps is None and b.wet_mass_flow_kgps is None:
            wet_flow = None
        else:
            fa = a.wet_mass_flow_kgps if a.wet_mass_flow_kgps is not None else b.wet_mass_flow_kgps
            fb = b.wet_mass_flow_kgps if b.wet_mass_flow_kgps is not None else fa
            wet_flow = (1.0 - w) * float(fa) + w * float(fb)
        return EquivalentProperties(
            omega0=(1.0 - w) * a.omega0 + w * b.omega0,
            tref_min=(1.0 - w) * a.tref_min + w * b.tref_min,
            slope_min_per_c=(1.0 - w) * a.slope_min_per_c + w * b.slope_min_per_c,
            bulk_density_kg_m3=density,
            wet_mass_flow_kgps=wet_flow,
        )

    def _delayed_feed_props(self, feed: FeedstockObservation, dt_s: float) -> EquivalentProperties:
        eq = self._equivalent_from_feed(feed)
        now = float(feed.time_s)
        if not self._feed_delay:
            self._feed_delay = [(now - max(self.cfg.feed_delay_s, 0.0) - 1e-6, eq), (now, eq)]
        else:
            if now >= self._feed_delay[-1][0] - 1e-12:
                self._feed_delay.append((now, eq))
            else:
                self._feed_delay = [(now - max(self.cfg.feed_delay_s, 0.0) - 1e-6, eq), (now, eq)]

        target_t = now - max(self.cfg.feed_delay_s, 0.0)
        while len(self._feed_delay) > 2 and self._feed_delay[1][0] <= target_t:
            self._feed_delay.pop(0)

        if target_t <= self._feed_delay[0][0]:
            return self._feed_delay[0][1]
        for (t0, e0), (t1, e1) in zip(self._feed_delay[:-1], self._feed_delay[1:]):
            if t0 <= target_t <= t1:
                if abs(t1 - t0) < 1e-12:
                    return e1
                return self._interp_equivalent(e0, e1, (target_t - t0) / (t1 - t0))
        return self._feed_delay[-1][1]

    def _mix_props(self, current: float, upstream: float, m_current: float, m_upstream: float) -> float:
        denom = max(m_current + m_upstream, 1e-12)
        return (m_current * current + m_upstream * upstream) / denom

    def _advect(self, feed_props: EquivalentProperties, dt_s: float) -> None:
        gamma = min(max(dt_s / max(self.tau_cell_s, 1e-9), 0.0), 1.0)
        dry_old = self.dry_mass.copy()
        water_old = self.water_mass.copy()
        T_old = self.T_solid.copy()
        omega0_old = self.omega0.copy()
        tref_old = self.tref.copy()
        slope_old = self.slope.copy()
        density_old = self.bulk_density.copy()
        cap_old = dry_old * self.opt_cfg.CS + water_old * self.opt_cfg.CW
        mass_old = dry_old + water_old

        for i in range(self.cfg.n_cells - 1, 0, -1):
            m_self = (1.0 - gamma) * mass_old[i]
            m_up = gamma * mass_old[i - 1]
            c_self = (1.0 - gamma) * cap_old[i]
            c_up = gamma * cap_old[i - 1]
            self.dry_mass[i] = (1.0 - gamma) * dry_old[i] + gamma * dry_old[i - 1]
            self.water_mass[i] = (1.0 - gamma) * water_old[i] + gamma * water_old[i - 1]
            self.T_solid[i] = (c_self * T_old[i] + c_up * T_old[i - 1]) / max(c_self + c_up, 1e-12)
            self.omega0[i] = self._mix_props(omega0_old[i], omega0_old[i - 1], m_self, m_up)
            self.tref[i] = self._mix_props(tref_old[i], tref_old[i - 1], m_self, m_up)
            self.slope[i] = self._mix_props(slope_old[i], slope_old[i - 1], m_self, m_up)
            self.bulk_density[i] = self._mix_props(density_old[i], density_old[i - 1], m_self, m_up)

        feed_mass = self._cell_mass_from_feed_props(feed_props)
        feed_density = self.opt_cfg.RHO_BULK if feed_props.bulk_density_kg_m3 is None else feed_props.bulk_density_kg_m3
        feed_dry = (1.0 - feed_props.omega0) * feed_mass
        feed_water = feed_props.omega0 * feed_mass
        feed_cap = feed_dry * self.opt_cfg.CS + feed_water * self.opt_cfg.CW
        m_self = (1.0 - gamma) * mass_old[0]
        m_up = gamma * feed_mass
        c_self = (1.0 - gamma) * cap_old[0]
        c_up = gamma * feed_cap
        self.dry_mass[0] = (1.0 - gamma) * dry_old[0] + gamma * feed_dry
        self.water_mass[0] = (1.0 - gamma) * water_old[0] + gamma * feed_water
        self.T_solid[0] = (c_self * T_old[0] + c_up * self.cfg.T_amb_C) / max(c_self + c_up, 1e-12)
        self.omega0[0] = self._mix_props(omega0_old[0], feed_props.omega0, m_self, m_up)
        self.tref[0] = self._mix_props(tref_old[0], feed_props.tref_min, m_self, m_up)
        self.slope[0] = self._mix_props(slope_old[0], feed_props.slope_min_per_c, m_self, m_up)
        self.bulk_density[0] = self._mix_props(density_old[0], feed_density, m_self, m_up)
        self._sync_omega_from_masses()

    def _cell_props(self, i: int) -> EquivalentProperties:
        return EquivalentProperties(float(self.omega0[i]), float(self.tref[i]), float(self.slope[i]), float(self.bulk_density[i]))

    def _drying_kinetic_limit_kg(self, i: int, dt_s: float) -> float:
        omega = float(self.omega[i])
        if omega <= self.cfg.omega_min:
            return 0.0
        props = self._cell_props(i)
        tau20_min = props.tref_min + props.slope_min_per_c * (float(self.T_solid[i]) - self.opt_cfg.T_REF)
        tau20_s = max(self.cfg.min_tau20_min, tau20_min) * 60.0
        d_omega = max(0.0, (omega - self.cfg.omega_min) * dt_s / tau20_s)
        omega_new = max(self.cfg.omega_min, omega - d_omega)
        dry = max(float(self.dry_mass[i]), 1e-12)
        water_new = self._water_mass_for_omega(dry, omega_new)
        return max(0.0, float(self.water_mass[i]) - water_new)

    def _evaporation_room_kg(self, i: int) -> float:
        dry = max(float(self.dry_mass[i]), 1e-12)
        water_min = self._water_mass_for_omega(dry, self.cfg.omega_min)
        return max(0.0, float(self.water_mass[i]) - water_min)

    def _apply_cell_energy(self, i: int, E_kJ: float, dt_s: float) -> tuple[float, float, float]:
        """Apply heat to one cell and return (sensible_kJ, latent_kJ, evaporated_kg)."""
        E = max(float(E_kJ), 0.0)
        if E <= 0.0:
            return 0.0, 0.0, 0.0

        sensible_kJ = 0.0
        latent_kJ = 0.0
        evaporated_kg = 0.0
        cap = self._cell_heat_capacity_kJ_per_K(i)
        if self.T_solid[i] < self.cfg.T_evap_C:
            need = (self.cfg.T_evap_C - float(self.T_solid[i])) * cap
            used = min(E, need)
            self.T_solid[i] += used / cap
            sensible_kJ += used
            E -= used

        if E > 0.0 and self.T_solid[i] >= self.cfg.T_evap_C and self._evaporation_room_kg(i) > 0.0:
            E_evap_budget = max(0.0, min(self.cfg.evaporation_heat_fraction, 1.0)) * E
            dm_energy = E_evap_budget / max(self.opt_cfg.LAMBDA, 1e-12)
            dm_kinetic = self._drying_kinetic_limit_kg(i, dt_s)
            dm_room = self._evaporation_room_kg(i)
            dm_evap = max(0.0, min(dm_energy, dm_kinetic, dm_room))
            if dm_evap > 0.0:
                self.water_mass[i] = max(0.0, self.water_mass[i] - dm_evap)
            E_latent = dm_evap * self.opt_cfg.LAMBDA
            E_remaining = max(E - E_latent, 0.0)
            cap_new = self._cell_heat_capacity_kJ_per_K(i)
            self.T_solid[i] += E_remaining / cap_new
            latent_kJ += E_latent
            sensible_kJ += E_remaining
            evaporated_kg += dm_evap
        elif E > 0.0:
            self.T_solid[i] += E / cap
            sensible_kJ += E

        self.T_solid[i] = float(np.clip(self.T_solid[i], self.cfg.T_amb_C, self.cfg.T_solid_max_C))
        self._sync_omega_from_masses()
        return float(sensible_kJ), float(latent_kJ), float(evaporated_kg)

    def _gas_indices(self) -> list[int]:
        if self.cfg.gas_flow_direction == "counter_current":
            return list(range(self.cfg.n_cells - 1, -1, -1))
        if self.cfg.gas_flow_direction == "co_current":
            return list(range(self.cfg.n_cells))
        raise ValueError("gas_flow_direction must be 'counter_current' or 'co_current'.")

    def _apply_heat_and_drying(self, Tg_in_C: float, vg_mps: float, dt_s: float) -> None:
        n = self.cfg.n_cells
        Tg_by_cell = [float("nan")] * n
        q_kW_by_cell = np.zeros(n, dtype=float)
        Tg_cell = float(Tg_in_C)
        mdot_g = max(rho_g(Tg_in_C, self.opt_cfg) * self.gas_flow_area_m2 * max(vg_mps, 0.0), 1e-9)
        Cg_kW_per_K = max(mdot_g * self.opt_cfg.CPG, 1e-9)
        U = self.opt_cfg.U0 + self.opt_cfg.K_U * (max(vg_mps, 0.0) ** self.opt_cfg.N_U)
        area = self.cell_heat_area_m2
        T_solid_start = self.T_solid.copy()

        for i in self._gas_indices():
            Tg_by_cell[i] = float(Tg_cell)
            deltaT = max(Tg_cell - float(T_solid_start[i]), 0.0)
            if deltaT <= 0.0:
                Q_kW = 0.0
                Tg_next = Tg_cell
            else:
                UA_kW_per_K = max(self.cfg.heat_transfer_factor * U * area / 1000.0, 0.0)
                ntu = UA_kW_per_K / Cg_kW_per_K
                effectiveness = 1.0 - math.exp(-max(ntu, 0.0))
                Q_kW = Cg_kW_per_K * deltaT * effectiveness
                Tg_next = Tg_cell - Q_kW / Cg_kW_per_K
            q_kW_by_cell[i] = max(Q_kW, 0.0)
            Tg_cell = float(Tg_next)

        sensible_kJ = 0.0
        latent_kJ = 0.0
        evaporated_kg = 0.0
        for i in range(n):
            s_kJ, l_kJ, dm_kg = self._apply_cell_energy(i, q_kW_by_cell[i] * dt_s, dt_s)
            sensible_kJ += s_kJ
            latent_kJ += l_kJ
            evaporated_kg += dm_kg
        self._last_Tg_profile = tuple(float(x) for x in Tg_by_cell) + (float(Tg_cell),)
        q_total_kW = float(np.nansum(q_kW_by_cell))
        q_sens_kW = sensible_kJ / max(dt_s, 1e-12)
        q_latent_kW = latent_kJ / max(dt_s, 1e-12)
        self.last_diagnostics = PreheaterDiagnostics(
            time_s=float(self.time_s),
            Tg_in_C=float(Tg_in_C),
            Tg_out_C=float(Tg_cell),
            vg_mps=float(vg_mps),
            mdot_gas_kgps=float(mdot_g),
            U_eff_W_m2K=float(U),
            Q_gas_to_solid_kW=q_total_kW,
            Q_sensible_kW=float(q_sens_kW),
            Q_latent_kW=float(q_latent_kW),
            heat_balance_residual_kW=float(q_total_kW - q_sens_kW - q_latent_kW),
            water_evap_kgps=float(evaporated_kg / max(dt_s, 1e-12)),
            dry_out_kgps=float(self.dry_mass[-1] / max(self.tau_cell_s, 1e-12)),
            water_out_kgps=float(self.water_mass[-1] / max(self.tau_cell_s, 1e-12)),
            wet_out_kgps=float((self.dry_mass[-1] + self.water_mass[-1]) / max(self.tau_cell_s, 1e-12)),
            inventory_dry_kg=float(np.nansum(self.dry_mass)),
            inventory_water_kg=float(np.nansum(self.water_mass)),
            inventory_total_kg=float(np.nansum(self.dry_mass + self.water_mass)),
            Tg_cell_min_C=float(np.nanmin(Tg_by_cell)),
            Tg_cell_max_C=float(np.nanmax(Tg_by_cell)),
        )

    def step(self, feed: FeedstockObservation, Tg_in_C: float, vg_mps: float, dt_s: float) -> PreheaterState:
        feed_props = self._delayed_feed_props(feed, dt_s)
        self._advect(feed_props, dt_s)
        self._apply_heat_and_drying(Tg_in_C, vg_mps, dt_s)
        self.time_s = float(feed.time_s)
        # Store diagnostics on the same public time stamp as the returned state.
        if hasattr(self, 'last_diagnostics'):
            self.last_diagnostics = PreheaterDiagnostics(**{**self.last_diagnostics.__dict__, 'time_s': float(feed.time_s)})
        return self.state(time_s=feed.time_s)

    def state(self, *, time_s: float | None = None) -> PreheaterState:
        self._sync_omega_from_masses()
        n = self.cfg.n_cells
        dz = self.cfg.length_m / max(n, 1)
        cells = []
        for i in range(n):
            remaining = max(0.0, (n - 1 - i + 0.5) * self.tau_cell_s)
            cells.append(PreheaterCellState(
                index=i,
                z_center_m=(i + 0.5) * dz,
                residence_left_s=remaining,
                omega=float(self.omega[i]),
                T_solid_C=float(self.T_solid[i]),
                omega0=float(self.omega0[i]),
                tref_min=float(self.tref[i]),
                slope_min_per_c=float(self.slope[i]),
                dry_mass_kg=float(self.dry_mass[i]),
                water_mass_kg=float(self.water_mass[i]),
                bulk_density_kg_m3=float(self.bulk_density[i]),
            ))
        return PreheaterState(
            time_s=float(self.time_s if time_s is None else time_s),
            cells=tuple(cells),
            omega_out=float(self.omega[-1]),
            T_solid_out_C=float(self.T_solid[-1]),
            Tg_profile_C=tuple(self._last_Tg_profile),
        )

    def representative_cell(self, target_remaining_s: float) -> PreheaterCellState:
        st = self.state()
        return min(st.cells, key=lambda c: abs(c.residence_left_s - target_remaining_s))

    def rollout_constant(self, feed: FeedstockObservation, Tg_C: float, vg_mps: float, horizon_s: float, dt_s: float):
        states = []
        t0 = self.time_s
        steps = max(1, int(math.ceil(horizon_s / max(dt_s, 1e-9))))
        model = self.clone()
        for k in range(steps):
            f = FeedstockObservation(
                time_s=t0 + (k + 1) * dt_s,
                moisture_wb=feed.moisture_wb,
                drying_time_ref_min=feed.drying_time_ref_min,
                drying_sensitivity_min_per_C=feed.drying_sensitivity_min_per_C,
                bulk_density_kg_m3=feed.bulk_density_kg_m3,
                wet_mass_flow_kgps=feed.wet_mass_flow_kgps,
                source=feed.source,
                confidence=feed.confidence,
                raw=feed.raw,
            )
            states.append(model.step(f, Tg_C, vg_mps, dt_s))
        return states
