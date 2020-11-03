#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
find test -name test\*.py | xargs -I {} -n 1 bash -c "python {} || exit 255"

pushd third_party/pytorch/test
for name in test_nn.py test_torch.py; do
    python $name -v;
done
popd

