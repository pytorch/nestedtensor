Below are tables built on top of the stable 1.7 ops [documention](https://pytorch.org/docs/1.7.0/torch.html), which aim to track the implementation of various operations.

### Pointwise Ops
<details>
  
| Name | Native | Derivative |
| ---- | ------ | ---------- |
|abs|&#9745;||
|absolute|||
|acos|&#9745;||
|arccos|||
|acosh|||
|arccosh|||
|add|||
|addcdiv|||
|addcmul|||
|angle|||
|asin|&#9745;||
|arcsin|||
|asinh|||
|arcsinh|||
|atan|&#9745;||
|arctan|||
|atanh|||
|arctanh|||
|atan2|||
|bitwise_not|||
|bitwise_and|||
|bitwise_or|||
|bitwise_xor|||
|ceil|&#9745;||
|clamp|&#9745;||
|clip|||
|conj|||
|cos|&#9745;||
|cosh|&#9745;||
|deg2rad|||
|div|||
|divide|||
|digamma|&#9745;||
|erf|&#9745;||
|erfc|&#9745;||
|erfinv|&#9745;||
|exp|&#9745;||
|exp2|||
|expm1|&#9745;||
|fix|||
|floor|&#9745;||
|floor_divide|||
|fmod|||
|frac|&#9745;||
|imag|||
|lerp|||
|lgamma|&#9745;||
|log|&#9745;||
|log10|&#9745;||
|log1p|&#9745;||
|log2|&#9745;||
|logaddexp|||
|logaddexp2|||
|logical_and|||
|logical_not|||
|logical_or|||
|logical_xor|||
|logit|||
|hypot|||
|i0|||
|mul|||
|multiply|||
|mvlgamma|&#9745;||
|neg|&#9745;||
|negative|||
|nextafter|||
|polygamma|||
|pow|||
|rad2deg|||
|real|||
|reciprocal|&#9745;||
|remainder|||
|round|&#9745;||
|rsqrt|&#9745;||
|sigmoid|&#9745;||
|sign|&#9745;||
|signbit|||
|sin|&#9745;||
|sinh|&#9745;||
|sqrt|&#9745;||
|square|||
|sub|||
|subtract|||
|tan|&#9745;||
|tanh|&#9745;||
|true_divide|||
|trunc|&#9745;||

</details>

### Reduction Ops

<details>

| Name | Native | Derivative |
| ---- | ------ | ---------- |
| argmax |||
| argmin |||
| amax |||
| amin |||
| max |||
| min |||
| dist |||
| logsumexp |||
| mean |&#9745;||
| median |||
| nanmedian |||
| mode |||
| norm |||
| nansum |||
| prod |&#9745;||
| quantile |||
| nanquantile |||
| std |||
| std_mean |||
| sum |&#9745;||
| unique |||
| unique_consecutive |||
| var |||
| var_mean |||
| count_nonzero |||

</details>

### Non-linear Activations

<details>
  
| Name | Native | Derivative |
| ---- | ------ | ---------- |
| nn.Softmin |||
| nn.Softmax |&#9745;||
| nn.Softmax2d |||
| nn.LogSoftmax |||
| nn.AdaptiveLogSoftmaxWithLoss |||
  
</details>
