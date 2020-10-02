# The nestedtensor package

The nestedtensor package is now on road to prototype release (end of October 2020).

It is developed [against a fork](https://github.com/cpuhrsch/pytorchnestedtensor) of PyTorch to enable cutting-edge features such as improved performance or better torch.vmap integration.

Developers wills thus need to build from source, but users can use the binary we will ship on a nightly basis once the prototype is released.


## Who can use this?

If you want to use the binaries you need to run on Linux, use Python 3.8+ and have a CUDA GPU with CUDA11.

If you want to build from source you can probably get it to work on many platforms, but supporting this won't take priority over development on the main platform. We're happy to review community contributions that achieve this however.

## Why use this?

In general we batch data for efficiency, but one batched kernels need regular, statically-shaped data.

One way of dealing with dynamic shapes then, is via padding and masking.
[Various](https://github.com/pytorch/fairseq/blob/54b934417d95baa1b0076089c61bde32728e34cf/fairseq/data/audio/raw_audio_dataset.py#L92)
[projects](https://github.com/facebookresearch/ParlAI/blob/8200396cdd08cfd26b01fe52b4a3bd0654081182/parlai/agents/drqa/utils.py#L143)
[construct](https://github.com/facebookresearch/detr/blob/4e1a9281bc5621dcd65f3438631de25e255c4269/util/misc.py#L306)
[masks](https://github.com/pytorch/vision/blob/24f16a338391d6f45aa6291c48eb6d5513771631/references/detection/utils.py#L102)
[that](https://github.com/pytorch/audio/blob/3250d3df168c956389bd16956aa458ce111570d0/examples/pipeline_wav2letter/datasets.py#L90), together with a data Tensor, are used as a representation for lists of dynamically shaped Tensors.

Obviously this is inefficient from a memory and compute perspective if the Tensors within this list are sufficient diverse.

You can also trace through the codebase where these masks are used and what kind of code that might cause (for example [universal_sentence_embedding](https://github.com/facebookresearch/ParlAI/blob/8200396cdd08cfd26b01fe52b4a3bd0654081182/parlai/agents/drqa/utils.py#L143)).

Otherwise we also have 
[one-off](https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pack_padded_sequence.html?highlight=pack_padded_sequence)
[operator](https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
[support](https://pytorch.org/docs/master/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention)
[in](https://pytorch.org/docs/master/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag) 
PyTorch that aim to support dynamic shapes via extra arguments such as a
[padding index](https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).
Of course the upside here is that these are fast and sometimes memory efficient, but don't provide a consistent interface.

Other users simply gave up and started writing [for-loops](https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/detection/transform.py#L97), or discovered that batching didn't help.

We want to have a single abstraction that is consistent, fast, memory efficient and readable and the nestedtensor project aims to provide that.

## Description

NestedTensors are a generalization of torch Tensors which eases working with data of different sizes and length. 
In a nutshell, Tensors have scalar entries (e.g. floats) and NestedTensors have Tensor entries. However, note that
a NestedTensor still is a Tensor. That means it needs to have a single dimension, single dtype, single device and single layout.

## Tensor entry constraints
 - Each Tensor constituent is of the dtype, layout and device of the containing NestedTensor.
 - The dimension of a constituent Tensor must be less than the dimension of the NestedTensor. 
 - An empty NestedTensor is of dimension zero.

## Prerequisites

- pytorch (installed from nestedtensor/third_party/pytorch submodule)
- torchvision (needed for examples and tests)
- ipython (needed for examples)
- notebook (needed for examples)

## Build for development

Get the source

```
git clone --recursive https://github.com/pytorch/nestedtensor
cd nestedtensor
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

Install the build tools

```
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests
conda install -c pytorch magma-cuda110
```

Build from scratch
```
./clean_build_with_submodule.sh
```

Incremental builds
```
./build_with_submodule.sh
```

## Tutorials

Please see the notebooks under [examples](https://github.com/pytorch/nestedtensor/tree/master/examples).

## Contribution
The project is under active development. If you have a suggestions or found an bug, please file an issue!
