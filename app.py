from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from services.flameguard_adapter import FlameGuardWebAdapter

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

adapter = FlameGuardWebAdapter(history_limit=1800)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    try:
        limit = int(request.args.get("limit", 360))
        limit = max(10, min(1200, limit))
        return jsonify(adapter.dashboard(history_limit=limit))
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


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
    app.run(host="127.0.0.1", port=5000, debug=False)
