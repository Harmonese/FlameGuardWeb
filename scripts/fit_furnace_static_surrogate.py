#!/usr/bin/env python3
"""Fit the COMSOL furnace steady-state surrogate.

Input CSVs are COMSOL table exports with columns:
    rd,w_b,value
where w_b is moisture in wet-basis percent and temperature values are in K.

The fitted model uses a total-degree polynomial in centered variables:
    x = rd - 3.0
    z = omega_b - 0.3218, where omega_b = w_b / 100

Temperature outputs are converted from K to degC before fitting.  The emitted
coefficients can be copied into plant/python_model/furnace.py and
controller/predictor/furnace.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

RD_CENTER = 3.0
OMEGA_B_CENTER = 0.3218
DEFAULT_DEGREE = 3

CSV_MAP = {
    "T_avg_C": ("燃烧区域平均温度-含水率、干基比例.csv", True),
    "T_stack_C": ("烟囱排气温度平均值-含水率、干基比例.csv", True),
    "v_stack_mps": ("烟囱排气速度平均值-含水率、干基比例.csv", False),
    "T_surface_min_C": ("燃烧面最低温度-含水率、干基比例.csv", True),
    "T_surface_max_C": ("燃烧面最高温度-含水率、干基比例.csv", True),
    "T_surface_std_C": ("燃烧面温度标准差-含水率、干基比例.csv", False),
}


def polynomial_powers(degree: int) -> list[tuple[int, int]]:
    """Return total-degree 2D polynomial powers in deterministic order."""
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]


def read_comsol_csv(path: Path, *, temperature_k: bool) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for line in fh:
            if not line.strip() or line.startswith("%"):
                continue
            rd_s, wb_s, value_s = next(csv.reader([line]))[:3]
            rd = float(rd_s)
            omega_b = float(wb_s) / 100.0
            value = float(value_s)
            if temperature_k:
                value -= 273.15
            rows.append((rd, omega_b, value))
    if not rows:
        raise ValueError(f"no data rows parsed from {path}")
    return rows


def design_matrix(rows: Iterable[tuple[float, float, float]], powers: list[tuple[int, int]]) -> np.ndarray:
    X = []
    for rd, omega_b, _ in rows:
        x = float(rd) - RD_CENTER
        z = float(omega_b) - OMEGA_B_CENTER
        X.append([(x ** i) * (z ** j) for i, j in powers])
    return np.asarray(X, dtype=float)


def fit_one(rows: list[tuple[float, float, float]], powers: list[tuple[int, int]]) -> dict[str, object]:
    X = design_matrix(rows, powers)
    y = np.asarray([r[2] for r in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    residual = pred - y

    loo_residuals = []
    for k in range(len(rows)):
        keep = np.ones(len(rows), dtype=bool)
        keep[k] = False
        coef_k, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        loo_residuals.append(float((X[~keep] @ coef_k)[0] - y[~keep][0]))
    loo = np.asarray(loo_residuals, dtype=float)

    return {
        "coefficients": [float(v) for v in coef],
        "train_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "train_max_abs": float(np.max(np.abs(residual))),
        "loo_rmse": float(np.sqrt(np.mean(loo ** 2))),
        "loo_max_abs": float(np.max(np.abs(loo))),
        "n_samples": int(len(rows)),
    }


def fit_all(data_dir: Path, *, degree: int) -> dict[str, object]:
    powers = polynomial_powers(degree)
    result: dict[str, object] = {
        "degree": int(degree),
        "rd_center": RD_CENTER,
        "omega_b_center": OMEGA_B_CENTER,
        "powers": powers,
        "outputs": {},
    }
    outputs: dict[str, object] = {}
    for output_name, (filename, temperature_k) in CSV_MAP.items():
        rows = read_comsol_csv(data_dir / filename, temperature_k=temperature_k)
        outputs[output_name] = fit_one(rows, powers)
    result["outputs"] = outputs
    return result


def python_literal(result: dict[str, object]) -> str:
    powers = [tuple(p) for p in result["powers"]]  # type: ignore[index]
    lines = [
        f"_SURROGATE_RD_CENTER = {result['rd_center']!r}",
        f"_SURROGATE_OMEGA_B_CENTER = {result['omega_b_center']!r}",
        f"_SURROGATE_POWERS = {powers!r}",
        "_SURROGATE_COEFFS = {",
    ]
    outputs = result["outputs"]  # type: ignore[index]
    assert isinstance(outputs, dict)
    for key in ["T_avg_C", "T_stack_C", "v_stack_mps", "T_surface_min_C", "T_surface_max_C", "T_surface_std_C"]:
        coeffs = outputs[key]["coefficients"]  # type: ignore[index]
        lines.append(f"    {key!r}: {coeffs!r},")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=here / "data" / "furnace_static_comsol")
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--out", type=Path, default=here / "furnace_static_surrogate_fit.json")
    parser.add_argument("--emit-python", action="store_true", help="also print a Python coefficient literal")
    args = parser.parse_args()

    result = fit_all(args.data_dir, degree=args.degree)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"degree={result['degree']} terms={len(result['powers'])}")
    for name, metrics in result["outputs"].items():  # type: ignore[index]
        print(
            f"{name:18s} train_rmse={metrics['train_rmse']:.4g} "
            f"train_max={metrics['train_max_abs']:.4g} "
            f"loo_rmse={metrics['loo_rmse']:.4g} "
            f"loo_max={metrics['loo_max_abs']:.4g}"
        )
    if args.emit_python:
        print("\n" + python_literal(result))


if __name__ == "__main__":
    main()
