#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace at {

Tensor transpose_buffer(
        Tensor nt_sizes_,
        Tensor input_buffer,
        Tensor output_buffer);

Tensor transpose_nhwc_nchw(Tensor input);

Tensor transpose_nchw_nhwc(Tensor input);

}
