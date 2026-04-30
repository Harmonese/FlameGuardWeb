from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runtime.tests.scenario_suite import run_all_cases


if __name__ == '__main__':
    run_all_cases()
