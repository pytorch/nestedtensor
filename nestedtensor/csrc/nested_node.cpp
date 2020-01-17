#include <nested_node.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

#include <cstring>

namespace torch {
namespace nested_tensor {

using namespace torch::jit;
using namespace torch::autograd::utils;

std::vector<std::string> split_str(std::string s, std::string delimiter) {
  std::vector<std::string> result;
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    result.push_back(token);
    s.erase(0, pos + delimiter.length());
  }
  result.push_back(s);
  return result;
}

std::string TensorNode___str__(
    const TensorNode& nested_node,
    const std::string& tabs) {
  std::stringstream result;
  auto tabs_ = tabs + "\t";
  result << "nested_tensor([";
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      if (i > 0) {
        result << ",";
      }
      auto tokens = split_str(
          THPUtils_unpackString(
              PyObject_Str(THPVariable_Wrap(nested_node.payload(i)))),
          "\n");
      for (const auto& token : tokens) {
        result << "\n" << tabs_ << token;
      }
    }
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      if (i > 0) {
        result << ",";
      }
      result << "\n" << tabs_;
      result << TensorNode___str__(nested_node.children(i), tabs_);
    }
  }
  result << std::endl;
  result << tabs << "])";
  return result.str();
}

std::string SizeNode___str__(
    const SizeNode& nested_node,
    const std::string name,
    const std::string& tabs) {
  std::stringstream result;
  auto tabs_ = tabs + "\t";
  result << name << "([";
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      if (i > 0) {
        result << ",";
      }
      c10::List<int64_t> size_node_payload = nested_node.payload(i);
      result << "\n" << tabs_ << "(";
      for (size_t j = 0; j < size_node_payload.size(); j++) {
        if (j > 0) {
          result << ", ";
        }
        result << size_node_payload[j];
      }
      result << ")";
    }
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      if (i > 0) {
        result << ",";
      }
      result << "\n" << tabs_;
      result << SizeNode___str__(nested_node.children(i), name, tabs_);
    }
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
      // NOTE: This is a very costly check! For now we'll let this to be
      // enabled manually. valid = valid && (variable_.is_pinned() ==
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
