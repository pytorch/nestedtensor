#include <nested_node.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

using namespace torch::jit;
using namespace torch::autograd::utils;

void pretty_print_tensor(std::stringstream& result, const std::string& python_tensor_string, const std::string& tabs) {
  auto tensor = strdup(python_tensor_string.c_str());
  auto tokens = strtok(tensor, "\n");
  while (tokens != NULL) {
    result << "\n" << tabs << tokens;
    tokens = strtok(NULL, "\n");
  }
}

std::string _NestedNode___str__(const TensorNode& nested_node, const std::string& tabs) {
  std::stringstream result;
  auto tabs_ = tabs + "\t";
  result << "nested_tensor([";
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      PyObject* objectsRepresentation =
          PyObject_Str(THPVariable_Wrap(nested_node.payload(i)));
      pretty_print_tensor(result, THPUtils_unpackString(objectsRepresentation), tabs_);
      result << ",";
    }
    // to remove the excess `,`
    result.seekp(-1, result.cur);
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result << "\n" << tabs_;
      result << _NestedNode___str__(nested_node.children(i), tabs_);
      result << ",";
    }
    result.seekp(-1, result.cur);
  }
  result << std::endl;
  result << tabs << "])";
  return result.str();
}

c10::optional<IValue> py_obj_to_ivalue(py::object py_obj) {
  auto inferred_type = tryToInferType(py_obj);
  if (!inferred_type.success()) {
    return c10::nullopt;
  }
  auto payload = toIValue(py_obj, inferred_type.type());
  return payload;
}

int64_t nested_node_numel(const TensorNode& meta_node) {
  int64_t result = 0;
  if (meta_node.is_leaf()) {
    for (size_t i = 0; i < meta_node.size(); i++) {
      result += meta_node.payload(i).numel();
    }
  } else {
    for (size_t i = 0; i < meta_node.degree(); i++) {
      result += nested_node_numel(meta_node.children(i));
    }
  }
  return result;
}

bool all_contiguous(const TensorNode& meta_node) {
  bool ac = true;
  if (meta_node.is_leaf()) {
    for (size_t i = 0; i < meta_node.size(); i++) {
      if (ac) {
        ac = ac && meta_node.payload(i).is_contiguous();
      }
    }
  } else {
    for (size_t i = 0; i < meta_node.degree(); i++) {
      if (ac) {
        ac = ac && all_contiguous(meta_node.children(i));
      }
    }
  }
  return ac;
}

bool all_size_equal(const SizeNode& nested_size) {
  if (nested_size.is_leaf()) {
    if (nested_size.size() > 0) {
      auto size0 = nested_size.payload(0);
      for (size_t i = 1; i < nested_size.size(); i++) {
        for (size_t j = 0; j < nested_size.payload(i).size(); j++) {
          if (size0[j] != nested_size.payload(i)[j]) {
            return false;
          }
        }
      }
    }
  } else {
    if (nested_size.degree() > 0) {
      // A child might be a leaf and degree will encode that.
      size_t nested_size0 = nested_size.children(0).degree();
      for (size_t i = 1; i < nested_size.degree(); i++) {
        if (nested_size0 != nested_size.children(i).degree() ||
            !all_size_equal(nested_size.children(i)))
          return false;
      }
    }
  }
  return true;
}

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride) {
  if (size.size() == 0) {
    return 0;
  }
  return size[0] * stride[0];
}

int64_t size_node_memory(SizeNode nested_size, SizeNode nested_stride) {
  int64_t result = 0;
  if (nested_size.is_leaf()) {
    for (size_t i = 0; i < nested_size.size(); i++) {
      result += num_memory(nested_size.payload(i), nested_stride.payload(i));
    }
  } else {
    for (size_t i = 0; i < nested_size.degree(); i++) {
      result +=
          size_node_memory(nested_size.children(i), nested_stride.children(i));
    }
  }
  return result;
}

at::Tensor _get_first_variable(TensorNode nested_node) {
  TensorNode leaf = get_first_leaf(nested_node);
  if (leaf.size()) {
    return leaf.payload(0);
  } else {
    return torch::ones({});
  }
}

at::Tensor NestedNode_to_tensor(const NestedNode<at::Tensor>& nested_node) {
  std::vector<at::Tensor> variables;
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      variables.emplace_back(nested_node.payload(i));
    }
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      variables.emplace_back(NestedNode_to_tensor(nested_node.children(i)));
    }
  }
  return stack(variables);
}

bool _verify_variables(
    const torch::autograd::Variable& first_variable,
    const TensorNode nested_node) {
  // The attributes must match across all constiuents
  //
  // The NestedTensor's attributes then become that of its
  // constiuents.
  //
  // data must be a list of Tensors or NestedTensors
  //
  // Attributes:
  //     dim()
  //     layout
  //     device
  //     dtype
  //     requires_grad
  //     is_pinned()
  bool valid = true;
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      at::Tensor variable = nested_node.payload(i);
      // TODO: Add more checks?
      valid = valid && (variable.dim() == first_variable.dim());
      valid = valid && (variable.layout() == first_variable.layout());
      valid = valid && (variable.device() == first_variable.device());
      valid = valid && (variable.dtype() == first_variable.dtype());
      valid =
          valid && (variable.requires_grad() == first_variable.requires_grad());
      // NOTE: This is a very costly check! For now we'll let this to be enabled
      // manually. valid = valid && (variable_.is_pinned() ==
      // first_variable.is_pinned());
    }
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      valid =
          valid && _verify_variables(first_variable, nested_node.children(i));
    }
  }
  return valid;
}

} // namespace nested_tensor
} // namespace torch
