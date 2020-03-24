#pragma once
#include <nested_tensor.h>

namespace torch {
namespace nested_tensor {

NestedTensor copy_(NestedTensor self, NestedTensor source, 
                bool non_blocking=false);
                

NestedTensor squeeze(NestedTensor input, c10::optional<int64_t> dim,
        c10::optional<NestedTensor> out);

}
}
