#pragma once
#include <nested_tensor.h>

namespace torch {
namespace nested_tensor {

NestedTensor squeeze(NestedTensor input, c10::optional<int64_t> dim,
        c10::optional<NestedTensor> out);

}
}
