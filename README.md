# BIG UPDATE: NestedTensor [in core](https://pytorch.org/docs/master/nested.html)!

## March 15 2022
As of recently we landed a minimal version of NestedTensor [in core PyTorch](https://pytorch.org/docs/master/nested.html)!
Operator coverage and migration of features is possible, but must be backed by issues (feature requests). If you have demand for specific NestedTensor operators, please open a feature request on [pytorch/pytorch](https://github.com/pytorch/pytorch/issues/new?assignees=&labels=&template=feature-request.yml). For a more impactful submission please include your motivation, use case and list of operators.
<br />
<br />
<br />
<br />
<br />
<br />

# The nestedtensor package [prototype](https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype)

If you are here because you ran into a runtime error due to a missing feature or some kind of bug, please [open an issue and fill in the appropiate template](https://github.com/pytorch/nestedtensor/issues/new/choose). If you have general feedback about this prototype [you can use our suggested template](https://github.com/pytorch/nestedtensor/issues/new?assignees=&labels=&template=prototype-feedback.md&title=) or just open a free-form issue if you like. Thank you for contributing to this project!

## Tutorials

If you are new to this project, we recommend you take a look at our [whirlwind introduction](https://colab.research.google.com/github/pytorch/nestedtensor/blob/master/tutorials/notebooks/basic.ipynb) to get started.

## Autograd support

Due to missing extensibility features of PyTorch nestedtensor currently lacks autograd support. We're actively working on this and recognize that it severely limits the applicability of the project. Please run nestedtensor operations within the [inference mode](https://github.com/ailzhang/rfcs/blob/rfc0011/RFC-0011-InferenceMode.md) context to prevent any adverse interactions with the autograd system.

For example
```
sentences = [torch.randn(10, 5), torch.randn(5, 5), torch.randn(9, 5)]
with torch.inference_mode():    
    nt = nestedtensor.nested_tensor(sentences)
    nt.sum(1)
```

## Binaries

Due to the development velocity of PyTorch the nestedtensor project is built on top of and dependent on a fixed, recent PyTorch nightly.

| Version | Python | CUDA | Wheels |
| --- | ---- | ------ | ---- |
| 0.1.1 | 3.6 | CPU-only | [nestedtensor](https://download.pytorch.org/nestedtensor/whl/nightly/cpu/py3.6/nestedtensor-0.1.1_cpu-cp36-cp36m-linux_x86_64.whl) |
| 0.1.1 | 3.7 | CPU-only | [nestedtensor](https://download.pytorch.org/nestedtensor/whl/nightly/cpu/py3.7/nestedtensor-0.1.1_cpu-cp37-cp37m-linux_x86_64.whl) |
| 0.1.1 | 3.8 | CPU-only | [nestedtensor](https://download.pytorch.org/nestedtensor/whl/nightly/cpu/py3.8/nestedtensor-0.1.1_cpu-cp38-cp38m-linux_x86_64.whl) |
| 0.1.1 | 3.6 | CUDA 10.2 | [nestedtensor](https://download.pytorch.org/nestedtensor/whl/nightly/cpu/py3.6/nestedtensor-0.1.1_cu102-cp36-cp36m-linux_x86_64.whl) |
| 0.1.1 | 3.7 | CUDA 10.2 | [nestedtensor](https://download.pytorch.org/nestedtensor/whl/nightly/cpu/py3.7/nestedtensor-0.1.1_cu102-cp37-cp37m-linux_x86_64.whl) |
| 0.1.1 | 3.8 | CUDA 10.2 | [nestedtensor](https://download.pytorch.org/nestedtensor/whl/nightly/cpu/py3.8/nestedtensor-0.1.1_cu102-cp38-cp38m-linux_x86_64.whl) |

When installing a binary please specify the corresponding torch nightly link archive to automatically pull in the correct PyTorch nightly.

CPU
```
pip install https://download.pytorch.org/nestedtensor/whl/nightly/cpu/py3.7/nestedtensor-0.1.1_cpu-cp37-cp37m-linux_x86_64.whl -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

CUDA 10.2
```
pip install https://download.pytorch.org/nestedtensor/whl/nightly/cu102/py3.7/nestedtensor-0.1.1_cu102-cp37-cp37m-linux_x86_64.whl -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
```

## Why consider using this? / Dealing with dynamic shapes

In general we batch data for efficiency, but usually batched kernels need, or greatly benefit from, regular, statically-shaped data.

One way of dealing with dynamic shapes then, is via padding and masking.
[Various](https://github.com/pytorch/fairseq/blob/54b934417d95baa1b0076089c61bde32728e34cf/fairseq/data/audio/raw_audio_dataset.py#L92)
[projects](https://github.com/facebookresearch/ParlAI/blob/8200396cdd08cfd26b01fe52b4a3bd0654081182/parlai/agents/drqa/utils.py#L143)
[construct](https://github.com/facebookresearch/detr/blob/4e1a9281bc5621dcd65f3438631de25e255c4269/util/misc.py#L306)
[masks](https://github.com/pytorch/vision/blob/24f16a338391d6f45aa6291c48eb6d5513771631/references/detection/utils.py#L102)
[that](https://github.com/pytorch/audio/blob/3250d3df168c956389bd16956aa458ce111570d0/examples/pipeline_wav2letter/datasets.py#L90), together with a data Tensor, are used as a representation for lists of dynamically shaped Tensors.

Obviously this is inefficient from a memory and compute perspective if the Tensors within this list are sufficiently diverse.

You can also trace through the codebase where these masks are used and observe the kind of code this approach often leads to. See for example [universal_sentence_embedding](https://github.com/facebookresearch/ParlAI/blob/8200396cdd08cfd26b01fe52b4a3bd0654081182/parlai/agents/drqa/utils.py#L143).

Otherwise we also have 
[one-off](https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pack_padded_sequence.html?highlight=pack_padded_sequence)
[operator](https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
[support](https://pytorch.org/docs/master/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention)
[in](https://pytorch.org/docs/master/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag) 
PyTorch that aims to support dynamic shapes via extra arguments such as a
[padding index](https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).
Of course, while these functions are fast and sometimes memory efficient, they don't provide a consistent interface.

Other users simply gave up and started writing [for-loops](https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/detection/transform.py#L97), or discovered that batching didn't help.

We want to have a single abstraction that is consistent, fast, memory efficient and readable and the nestedtensor project aims to provide that.

## How does nestedtensor help here?

NestedTensors are a generalization of torch Tensors which eases working with data of different shapes and lengths. 
In a nutshell, Tensors have scalar entries (e.g. floats) and NestedTensors have Tensor entries. However, note that
a NestedTensor is still a Tensor. That means it needs to have a single dimension, single dtype, single device and single layout.

 Tensor entry constraints:
 - Each Tensor constituent is of the dtype, layout and device of the containing NestedTensor.
 - The dimension of a constituent Tensor must be less than the dimension of the NestedTensor. 
 - An empty NestedTensor is of dimension zero.

## Prototype classification

The nestedtensor package is a prototype intended for early stage feedback and testing. It is on the road to a beta classification, but there is no definitive timeline yet. See [PyTorch feature classification](https://pytorch.org/docs/stable/index.html) for what prototype, beta and stale means.

## Dependencies

- pytorch (installed from nestedtensor/third_party/pytorch submodule)
- torchvision (needed for examples and tests)
- ipython (needed for examples)
- notebook (needed for examples)

## Contribution
The project is under active development. If you have a suggestions or found a bug, please file an issue!
