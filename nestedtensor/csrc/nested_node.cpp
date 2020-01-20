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
      // TODO: Parameterize this to allow printing torch.Size etc.
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

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride) {
  if (size.size() == 0) {
    return 0;
  }
  return size[0] * stride[0];
}

int64_t size_node_memory(SizeNode nested_size, SizeNode nested_stride) {
  auto fn = [](c10::List<int64_t> size,
               c10::List<int64_t> stride,
               int64_t input) { return num_memory(size, stride) + input; };
  return reduce<decltype(fn), int64_t, c10::List<int64_t>, c10::List<int64_t>>(
      nested_size, nested_stride, fn, 0);
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

std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node) {
  if (size_node.is_leaf()) {
    std::vector<c10::optional<int64_t>> result;
    result.push_back(size_node.size());
    if (size_node.size() == 0) {
      return result;
    }

    for (const auto& size : size_node.payload(0)) {
      result.push_back(size);
    }

    for (size_t j = 1; j < result.size(); j++) {
      for (size_t i = 1; i < size_node.size(); i++) {
        if (!result[j]) {
          break;
        }
        if ((*(result[j])) != size_node.payload(i)[j - 1]) {
          result[j] = c10::nullopt;
        }
      }
    }
    return result;
  }
  std::vector<c10::optional<int64_t>> result;
  result.push_back(size_node.degree());

  if (size_node.degree() > 0) {
    for (const auto& size : construct_size(size_node.children(0))) {
      result.push_back(size);
    }
    for (size_t i = 1; i < size_node.degree(); i++) {
      auto size_node_i = construct_size(size_node.children(i));
      for (size_t j = 1; j < result.size(); j++) {
        if (result[j] && ((*result[j]) != size_node_i[j - 1])) {
          result[j] = c10::nullopt;
        }
      }
    }
  }

  return result;
}

} // namespace nested_tensor
} // namespace torch
