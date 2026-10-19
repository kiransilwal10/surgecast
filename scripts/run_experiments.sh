#!/usr/bin/env bash
# End to end: download -> build -> 2x2 ablation (graph x storm node) -> evaluate -> figures.
set -euo pipefail

python scripts/download.py --start 2008 --end 2025
python scripts/build_dataset.py

CFG=configs/surgecast.yaml
python -m surgecast.train --config $CFG --name gru        --set model.use_graph=false model.use_storms=false
python -m surgecast.train --config $CFG --name gru_storms --set model.use_graph=false
python -m surgecast.train --config $CFG --name gnn        --set model.use_storms=false
python -m surgecast.train --config $CFG --name surgecast

python -m surgecast.evaluate --runs gru gru_storms gnn surgecast
python scripts/make_figures.py
