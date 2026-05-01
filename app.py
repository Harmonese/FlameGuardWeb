from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from services.flameguard_adapter import FlameGuardWebAdapter

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

def _int_env(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


HISTORY_LIMIT = _int_env("FLAMEGUARD_HISTORY_LIMIT", 1800, min_value=120, max_value=20000)
DASHBOARD_DEFAULT_LIMIT = _int_env("FLAMEGUARD_DASHBOARD_DEFAULT_LIMIT", 360, min_value=10, max_value=5000)
DASHBOARD_MAX_LIMIT = _int_env("FLAMEGUARD_DASHBOARD_MAX_LIMIT", 1200, min_value=60, max_value=20000)
REFRESH_MS = _int_env("FLAMEGUARD_REFRESH_MS", 250, min_value=100, max_value=5000)
HEALTH_REQUIRE_NMPC = _bool_env("FLAMEGUARD_HEALTH_REQUIRE_NMPC", True)

adapter = FlameGuardWebAdapter(history_limit=HISTORY_LIMIT)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    try:
        limit = int(request.args.get("limit", DASHBOARD_DEFAULT_LIMIT))
        limit = max(10, min(DASHBOARD_MAX_LIMIT, limit))
        return jsonify(adapter.dashboard(history_limit=limit))
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500




@app.route("/healthz", methods=["GET"])
def healthz():
    payload, status_code = adapter.health(require_realtime_nmpc=HEALTH_REQUIRE_NMPC)
    payload.update({
        "service": "FlameGuardWeb",
        "environment": os.environ.get("FLAMEGUARD_ENV", "development"),
    })
    return jsonify(payload), status_code


@app.route("/api/config", methods=["GET"])
def runtime_config():
    return jsonify({
        "success": True,
        "refresh_ms": REFRESH_MS,
        "dashboard_default_limit": DASHBOARD_DEFAULT_LIMIT,
        "dashboard_max_limit": DASHBOARD_MAX_LIMIT,
        "history_limit": HISTORY_LIMIT,
        "environment": os.environ.get("FLAMEGUARD_ENV", "development"),
    })


@app.route("/api/feedstock", methods=["POST"])
def feedstock():
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify(adapter.submit_feedstock(payload, history_limit=120))
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/start", methods=["POST"])
def start_monitoring():
    try:
        return jsonify(adapter.start())
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/stop", methods=["POST"])
def stop_monitoring():
    try:
        return jsonify(adapter.stop())
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_monitoring():
    try:
        return jsonify(adapter.reset())
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/solve", methods=["POST"])
def solve_compatibility():
    """Compatibility endpoint for the old front-end button.

    The static optimizer has been retired. This endpoint now means:
    accept composition/manual properties, update the realtime NMPC feedstock
    stream, and return a dashboard payload plus a legacy-shaped mapping so older
    UI code will not break during incremental migration.
    """
    try:
        payload = request.get_json(silent=True) or {}
        dashboard_payload = adapter.submit_feedstock(payload, history_limit=120)
        return jsonify({
            "success": True,
            "dashboard": dashboard_payload,
            "result": legacy_result_from_dashboard(dashboard_payload),
            "message": "实时监控兼容模式：/api/solve 已映射为 /api/feedstock + /api/dashboard。",
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


def legacy_result_from_dashboard(payload: dict) -> dict:
    furnace = payload["furnace"]
    preheater = payload["preheater"]
    control = payload["control"]
    feedstock = payload["feedstock"]
    return {
        "x": feedstock["composition"],
        "outer": {
            "T_tar_k": furnace["T_set_C"],
            "w_tar_k": control["omega_target"] * 100.0,
        },
        "optimal": {
            "Tg": control["Tg_cmd_C"],
            "vg": control["vg_cmd_mps"],
            "Tm": preheater["T_solid_out_C"],
            "w_opt": preheater["omega_out"] * 100.0,
            "Qreq_kW": max(0.0, control["Q_aux_heat_kW"]),
            "Qsup_kW": max(0.0, control["Q_aux_heat_kW"]),
            "mdot_preheater": control["mdot_preheater_kgps"],
            "T_stack_cap": furnace["T_stack_C"],
            "Tmin_burn": furnace["T_avg_C"],
            "d_w_minus": 0.0,
            "d_w_plus": abs(preheater["omega_out"] - control["omega_target"]) * 100.0,
        },
        "overall": {
            "numerically_feasible": bool(control["operator_feasible"]),
        },
        "config": {
            "T_SET": furnace["T_set_C"],
        },
    }


if __name__ == "__main__":
    host = os.environ.get("FLAMEGUARD_HOST", "127.0.0.1")
    port = _int_env("FLAMEGUARD_PORT", 5000, min_value=1, max_value=65535)
    debug = os.environ.get("FLAMEGUARD_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
