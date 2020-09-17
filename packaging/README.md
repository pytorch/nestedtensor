# Building nestedtensor packages for release

## Anaconda packages

### Linux

```bash
nvidia-docker run -it --ipc=host --rm -v $(pwd):/remote soumith/conda-cuda bash
pushd remote/conda

./build_nestedtensor.sh 9.0
./build_nestedtensor.sh 10.0
./build_nestedtensor.sh cpu

# copy packages over to /remote
# exit docker
# anaconda upload -u pytorch nestedtensor*.bz2
```

## Wheels

### Linux

pushd wheel

```bash
nvidia-docker run -it --ipc=host --rm -v $(pwd):/remote soumith/manylinux-cuda90:latest bash
cd remote
./linux_manywheel.sh cu90

rm -rf /usr/local/cuda*
./linux_manywheel.sh cpu
```

```bash
nvidia-docker run -it --ipc=host --rm -v $(pwd):/remote soumith/manylinux-cuda100:latest bash
cd remote
./linux_manywheel.sh cu100
```

wheels are in the folders `cpu`, `cu90`, `cu100`.

You can upload the `cu90` wheels to twine with `twine upload *.whl`.
Which wheels we upload depends on which wheels PyTorch uploads as default, and right now, it's `cu90`.
