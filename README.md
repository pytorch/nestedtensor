# The nestedtensor package

NOTE: nestedtensor is under active development and various aspects may change.

NOTE: We test and develop against nightlies! Please use the most recent version of PyTorch if you plan to use this code.

## Motivation

We often want to manipulate collections of Tensors of different shapes. For example, paragraphs of text, images of different sizes or audio files of different lengths. We don't have a first class generalization that eases the concurrent manipulation of collections of this type of data. We further often want to batch arbitrary data and operations for efficiency, which then leads us to write awkward workarounds such as padding.

## Description

NestedTensors are a generalization of torch Tensor which eases working with data of different sizes and length. 
In general, there are two cases for which NestedTensors provide computational representations: list of tensors and lists of NestedTensors.

## Constraints
 - Each Tensor constituent of the list it represents, if any, must be of its dtype, layout and device. 
 - The dimension of a constituent Tensor must be one less than the dimension of the NestedTensor. 
 - An empty list of Tensors yields a NestedTensor of dimension zero.
 - Each constituent NestedTensor must be of its dtype, layout and device. 
 - The dimension of a constituent NestedTensor must be one less than the dimension of the NestedTensor.

## Prerequisites

- pytorch
- torchvision (needed for examples)
- ipython (needed for examples)
- notebook (needed for examples)

If you have conda installed on your machine, you can install these via
```
conda install ipython pytorch notebook torchvision -c pytorch-nightly
```

## Build 
Run 
```
python setup.py develop
```

NOTE: This repository uses a C++ extension. Please file an issue if you want into compilation errors.

## Usage
Import nested tensors and torch via ```from nestedtensor import torch```

### Creation

```
nt = nestedtensor.nested_tensor(
    [
        [
            torch.rand(2, 3),
            torch.rand(4, 5)
        ],
        [
            torch.rand(1, 2)
        ]
    ])
```

```
a = torch.tensor([1])
b = torch.tensor([[2, 2],
                  [3, 3],
                  [4, 4],
                  [5, 5]])
nt2 = nestedtensor.nested_tensor([[a],[b]])
```

The level of nesting is inferred from the input. The constructor always copies. Whatever you pass into the constructor will share no data with what the constructor returns. This matches torch.tensor's behavior.

If given a NestedTensor or Tensor it will return a detached copy, which is consistent with the behavior of torch.tensor. Remember that you cannot mix Tensors and NestedTensors within a given list.

A side-note on naming: nestedtensor is a python packed and as such [shouldn't have underscores and is lower case](https://www.python.org/dev/peps/pep-0008/#package-and-module-names), but nested_tensor is a python function and as [such should use underscores](https://www.python.org/dev/peps/pep-0008/#function-and-variable-names) in contrast to the [CapWorded NestedTensor class](https://www.python.org/dev/peps/pep-0008/#class-names).

### Conversion/unbind()
A user can retrieve the constituent Tensors via unbind. Unbind is currently used by torch to turn Tensors into tuples of Tensors. Unbind always returns a tuple of views.

```
>>> from nestedtensor import torch
>>>
>>> a = [
...        [torch.rand(1, 2), torch.rand(2, 1)],
...        [torch.rand(3, 2)]
...     ]
>>>
>>> b = nestedtensor.nested_tensor(a)
>>> print(b)
nested_tensor([
    [
        tensor([[0.5356, 0.5609]]),
        tensor([[0.1567],
                [0.8880]])
    ],
    [
        tensor([[0.4060, 0.4359],
                [0.4069, 0.3802],
                [0.0040, 0.3759]])
    ]
])
>>> b1 = b.unbind() # Tuple of 2 NestedTensors
>>> print(b1)
(nested_tensor([
    tensor([[0.5356, 0.5609]]),
    tensor([[0.1567],
            [0.8880]])
]), nested_tensor([
    tensor([[0.4060, 0.4359],
            [0.4069, 0.3802],
            [0.0040, 0.3759]])
]))
>>> b2 = b1[0].unbind() # Tuple of 2 Tensors
>>> print(b2)
(tensor([[0.5356, 0.5609]]),
 tensor([[0.1567],
		 [0.8880]]))
```

### Other Ops
We currently lack detailed documentation for all supported ops. Please see the examples and stay tuned for updates on this front.


## The tensorwise decorator
The nestedtensor package allows the user to decorate existing functions with a tensorwise decorator. This decorator lifts the given function to check for NestedTensor arguments and recursively apply it to their constituents.

```
>>> from nestedtensor import torch
>>>
>>> @torch.tensorwise()
... def simple_fn(t1, t2):
...     return t1 + 1 + t2
...
>>>
>>> a = torch.tensor([1, 2])
>>> b = torch.tensor([7, 8])
>>> print(simple_fn(a, b))
tensor([ 9, 11])
>>> c = torch.tensor([4, 3])
>>> d = torch.tensor([5, 6])
>>> print(simple_fn(c, d))
tensor([10, 10])
>>>
>>> n = nestedtensor.nested_tensor([a, c])
>>> m = nestedtensor.nested_tensor([b, d])
>>> print(simple_fn(n, m))
nested_tensor([
    tensor([ 9, 11]),
    tensor([10, 10])
])
>>> print(simple_fn(a, m)) # Broadcasting
nested_tensor([
    tensor([ 9, 11]),
    tensor([7, 9])
])
>>> print(a)
tensor([1, 2])
>>> print(m)
nested_tensor([
    tensor([7, 8]),
    tensor([5, 6])
])
>>> print(simple_fn(a, m)) # Broadcasting
nested_tensor([
    tensor([ 9, 11]),
    tensor([7, 9])
])
```

## Contribution
The project is under active development. If you have a suggestions or found an bug, please file an issue!
