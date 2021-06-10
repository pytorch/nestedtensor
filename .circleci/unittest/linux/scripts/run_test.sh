#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
find test -name test\*.py | xargs -I {} -n 1 bash -c "python {} --verbose -f || exit 255"
