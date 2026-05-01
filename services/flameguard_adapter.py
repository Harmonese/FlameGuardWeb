from __future__ import annotations

"""Thin adapter exposed to Flask.

Phase 2a uses FlameGuard-main's Python plant plus a fast receding-horizon
rollout NMPC loop.  The full SLSQP NMPC code is vendored for later isolation
in a solver process, but it is not executed inside Flask requests.  The
original phase-1 pseudo generator is kept as a fallback so the web app still
runs on machines without scipy/numpy.
"""

from threading import RLock
from typing import Any

from .composition_adapter import validate_composition
from .telemetry_store import TelemetryStore

try:  # Prefer the real FlameGuard-main path.
    from .realtime_nmpc_generator import RealtimeNMPCGenerator
    _GENERATOR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - fallback for minimal machines
    RealtimeNMPCGenerator = None  # type: ignore
    _GENERATOR_IMPORT_ERROR = exc

from .simulation_generator import Phase1SimulationGenerator


class FlameGuardWebAdapter:
    EXPECTED_GENERATOR_MODE = "realtime_fast_nmpc"

    def __init__(self, history_limit: int = 1200) -> None:
        self._lock = RLock()
        self.store = TelemetryStore(maxlen=history_limit)
        self.generator_mode = "realtime_fast_nmpc"
        if RealtimeNMPCGenerator is not None:
            try:
                self.generator = RealtimeNMPCGenerator()
            except Exception as exc:
                self.generator_mode = "phase1_fallback"
                self.generator = Phase1SimulationGenerator()
                self._startup_error = str(exc)
            else:
                self._startup_error = ""
        else:
            self.generator_mode = "phase1_fallback"
            self.generator = Phase1SimulationGenerator()
            self._startup_error = str(_GENERATOR_IMPORT_ERROR)


    def health(self, *, require_realtime_nmpc: bool = True) -> tuple[dict[str, Any], int]:
        """Return a strict container/readiness health payload.

        The old Flask endpoint only proved that the route was reachable. This
        probe also verifies that the realtime NMPC generator imported,
        initialized, and can produce a minimally valid dashboard snapshot.
        """
        with self._lock:
            snapshot: dict[str, Any] | None = None
            snapshot_error = ""
            try:
                snapshot = self.generator.snapshot()
            except Exception as exc:  # pragma: no cover - defensive runtime probe
                snapshot_error = str(exc)

            required_dashboard_keys = {"success", "time_s", "running", "feedstock", "furnace", "preheater", "control", "health"}
            snapshot_ok = bool(snapshot) and required_dashboard_keys.issubset(snapshot.keys()) and bool(snapshot.get("success"))
            mode_ok = self.generator_mode == self.EXPECTED_GENERATOR_MODE
            startup_ok = self._startup_error == ""
            runtime_import_ok = RealtimeNMPCGenerator is not None

            checks = {
                "runtime_import_ok": runtime_import_ok,
                "runtime_initialized": getattr(self, "generator", None) is not None,
                "startup_ok": startup_ok,
                "mode_ok": mode_ok,
                "dashboard_snapshot_ok": snapshot_ok,
                "realtime_nmpc_required": bool(require_realtime_nmpc),
            }

            ok = bool(
                checks["runtime_initialized"]
                and checks["dashboard_snapshot_ok"]
                and ((checks["startup_ok"] and checks["mode_ok"]) if require_realtime_nmpc else True)
            )

            status: dict[str, Any] = {
                "ok": ok,
                "status": "ready" if ok else "not_ready",
                "mode": self.generator_mode,
                "expected_mode": self.EXPECTED_GENERATOR_MODE,
                "startup_error": self._startup_error,
                "snapshot_error": snapshot_error,
                "checks": checks,
            }
            if snapshot is not None:
                runtime_health = snapshot.get("health") or {}
                status["runtime"] = {
                    "schema": snapshot.get("schema"),
                    "running": snapshot.get("running"),
                    "time_s": snapshot.get("time_s"),
                    "process_health_ok": bool(runtime_health.get("ok", True)),
                    "process_health_message": runtime_health.get("message", ""),
                }

            return status, 200 if ok else 503

    def dashboard(self, *, history_limit: int = 240) -> dict:
        with self._lock:
            payload = self.generator.snapshot()
            payload.setdefault("runtime", {})
            payload["runtime"].update({
                "mode": self.generator_mode,
                "startup_error": self._startup_error,
                "poll_hint_ms": 250,
            })
            self._append_history(payload)
            payload["history"] = self.store.tail(history_limit)
            return payload

    def submit_feedstock(self, data: dict, *, history_limit: int = 240) -> dict:
        composition = validate_composition(data.get("composition", [0.2, 0.2, 0.1, 0.2, 0.2, 0.1]))
        source = data.get("source") or data.get("mode") or "manual"
        confidence = float(data.get("confidence", 1.0))
        wet_mass_flow = data.get("wet_mass_flow_kgps", None)
        if wet_mass_flow in ("", None):
            wet_mass_flow = None
        else:
            wet_mass_flow = float(wet_mass_flow)
        with self._lock:
            payload = self.generator.update_feedstock(
                composition,
                source=source,
                confidence=confidence,
                wet_mass_flow_kgps=wet_mass_flow,
            )
            payload.setdefault("runtime", {})
            payload["runtime"].update({
                "mode": self.generator_mode,
                "startup_error": self._startup_error,
                "poll_hint_ms": 250,
            })
            self._append_history(payload)
            payload["history"] = self.store.tail(history_limit)
            return payload

    def start(self) -> dict:
        with self._lock:
            self.generator.start()
            return self.dashboard(history_limit=240)

    def stop(self) -> dict:
        with self._lock:
            self.generator.stop()
            return self.dashboard(history_limit=240)

    def reset(self) -> dict:
        with self._lock:
            self.generator.reset()
            self.store.clear()
            return self.dashboard(history_limit=240)

    def shutdown(self) -> None:
        gen = getattr(self, "generator", None)
        if gen is not None and hasattr(gen, "shutdown"):
            gen.shutdown()

    def _append_history(self, payload: dict) -> None:
        furnace = payload["furnace"]
        preheater = payload["preheater"]
        control = payload["control"]
        feedstock = payload["feedstock"]
        row = {
            "time_s": payload["time_s"],
            "T_avg_C": furnace["T_avg_C"],
            "T_stack_C": furnace["T_stack_C"],
            "v_stack_mps": furnace["v_stack_mps"],
            "T_set_C": furnace["T_set_C"],
            "T_compliance_min_C": furnace["T_compliance_min_C"],
            "Tg_ref_C": control["Tg_ref_C"],
            "Tg_cmd_C": control["Tg_cmd_C"],
            "vg_ref_mps": control["vg_ref_mps"],
            "vg_cmd_mps": control["vg_cmd_mps"],
            "omega_out": preheater["omega_out"],
            "T_solid_out_C": preheater["T_solid_out_C"],
            "safety_margin_C": control["safety_margin_C"],
            "Q_aux_heat_kW": control["Q_aux_heat_kW"],
            "nmpc_last_solve_ms": control.get("nmpc_last_solve_ms"),
            "nmpc_plan_age_s": control.get("nmpc_plan_age_s"),
            "nmpc_async_job_running": 1.0 if control.get("nmpc_async_job_running") else 0.0,
            "disturbance_Tavg_C": furnace.get("disturbance_Tavg_C", 0.0),
            "feed_moisture_wb": feedstock.get("moisture_wb"),
        }
        self.store.append(row)
