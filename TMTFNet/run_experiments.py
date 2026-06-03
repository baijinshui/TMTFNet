#!/usr/bin/env python3
import os
import runpy
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "TMTFNet")
ENTRYPOINT = os.path.join(PROJECT_DIR, "run_experiments.py")

sys.path.insert(0, PROJECT_DIR)
runpy.run_path(ENTRYPOINT, run_name="__main__")
