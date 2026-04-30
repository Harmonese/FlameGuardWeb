"""Runtime layer: simulation orchestration, telemetry, and plotting."""

from .simulator import (
    EventWindow,
    History,
    SimConfig,
    history_to_csv_rows,
    plot_history,
    print_metrics_table,
    run_case,
    save_case_artifacts,
)

__all__ = [
    "EventWindow",
    "History",
    "SimConfig",
    "run_case",
    "save_case_artifacts",
    "history_to_csv_rows",
    "plot_history",
    "print_metrics_table",
]
