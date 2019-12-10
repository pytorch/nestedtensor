PLEASE NOTE, NESTEDTENSORS ARE IN AN ACTIVE DEVELOPMENT AND EVERYTHING HERE IS A SUBJECT TO CHANGE.

## Motivation

Many fields manipulate collections of Tensors of different shapes. For example, paragraphs of text, images of different sizes or audio files of different lengths. We don't have a first class abstrageneralizationction that enables the concurrent manipulation of collections of this type of data. We further often need to batch arbitrary data and operations for efficiency.

## Decription

NestedTensor is an generalization of torch tensor which allows working with data of different sizes and length. 
In general, there are two cases for which NestedTensors provide computational representations: list of tensors and lists of nestedtensors.



**Lists of NestedTensors**


## Constraints
 - Each Tensor constituent of the list it represents, if any, must be of its dtype, layout and device. 
 - The dimension of a constituent Tensor must be one less than the dimension of the NestedTensor. 
 - An empty list of Tensors yields a NestedTensor of dimension one. 
 - Each constituent NestedTensor must be of its dtype, layout and device. 
 - The dimension of a constituent NestedTensor must be one less than the dimension of the NestedTensor.

## Prerequisites

You will need this packages to be installed:
- pytorch
- torchvision
- ipython (needed for examples)
- notebook (needed for examples)

If you have conda installed on your machine, you can install these via
```
conda install ipython pytorch notebook torchvision cpuonly -c pytorch-nightly
```

## Build 
Run 
```
python setup.py develop
```

## Usage
Import nested tensors and torch via ```from nestedtensor import torch```
### Creation
You can create a nester tensor by initializing its values direcrtly or by creating tensors first and then add them to nested tensor.

```
nt = torch.nested_tensor(
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
nt2 = torch.nested_tensor([[a],[b]])
```

The level of nesting is inferred from the input. The constructor always copies. Whatever you pass into the constructor will share no data with what the constructor returns. This matches torch.tensor's behavior.

If given a NestedTensor or Tensor it will return a detached copy, which is consistent with the behavior of torch.tensor. Remember that you cannot mix Tensors and NestedTensors within a given list.

### Conversion/unbind()
A user can retrieve the constituent Tensors via unbind. Unbind is currently used by torch to turn Tensors into tuples of Tensors. Unbind always returns a tuple of views. We do not (yet support a dimension argument to unbind for NestedTensors, because it forces us to argue about shape.

```
a = [ \
       [torch.rand(2, 3), torch.rand(4, 5)], \
       [torch.rand(6, 7)] \
      ]

b = torch.nested_tensor(a)
b1 = b.unbind() # Tuple of 2 NestedTensors
b2 = b1[0].unbind() # Tuple of 2 Tensors
```

### Other Ops
For all other ops, please, see examples.


## The tensorwise decorator
The nestedtensor package allows the user to decorate existing functions with a tensorwise decorator. This decorator lifts the given function to check for NestedTensor arguments and apply recursively apply it to their constiuents with all other arguments untouched.

Decorating the function as tensorwise does not affect its behavior with respect to non-NestedTensor arguments. In particular, the tensorwise decorator will search all arguments for a NestedTensor and if none is found dispatch to exactly the given function.

```
@tensorwise()
@tensorwise()
def simple_fn(t1, t2):
    return t1 + 1 + t2


a = torch.tensor([1, 2])
b = torch.tensor([7, 8])
print(simple_fn(a, b))
```

## Contribution
The project is in an active development. If you have a suggestions or have found an issue, please file an issue!
