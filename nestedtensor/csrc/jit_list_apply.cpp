#include <jit_list_apply.h>
#include <python_list_nested_tensor.h>

namespace torch {
namespace nested_tensor {

struct ArgWrapper {
  ArgWrapper(TensorNode nested_tensor)
      : _is_nested_tensor(true), _nested_tensor(nested_tensor) {}
  ArgWrapper(c10::IValue ivalue) : _is_nested_tensor(false), _ivalue(ivalue) {}

  bool is_nested_tensor() {
    return _is_nested_tensor;
  }

  c10::IValue ivalue() {
    return _ivalue;
  }

  TensorNode nested_tensor() {
    return _nested_tensor;
  }

 private:
  bool _is_nested_tensor;
  c10::IValue _ivalue;
  TensorNode _nested_tensor;
};

// TODO: Assert that one arg must be a nestedtensor?
static TensorNode apply_jit_function(
    const std::vector<ArgWrapper>& args,
    Function& fn) {
  bool all_leaf = true;
  for (size_t i = 0; i < args.size(); i++) {
    if (args[i].is_nested_tensor()) {
      all_leaf = all_leaf && args[i].nested_tensor().is_leaf();
    }
  }
  if (all_leaf) {
    // NOTE: We assume no named tensors and no sparse variables as
    // appropriate for TorchScript.
    // TODO: Assert leaf sizes match and are non-zero, otherwise this isn't
    // a NestedTensor function.
    size_t leaf_size = 0;
    for (size_t i = 0; i < args.size(); i++) {
      if (args[i].is_nested_tensor()) {
        leaf_size = args[i].size();
        break;
      }
    }
    std::vector<std::vector<IValue>> stacks(leaf_size);
    for (size_t j = 0; j < leaf_size; j++) {
      for (size_t i = 0; i < args.size(); i++) {
        if (args[i].is_nested_tensor() {
          stacks[j].push_back(args[i].nested_tensor().payload(j));
        } else {
          stacks[j].push_back(args[i].ivalue());
        }
      }
    }
    c10::List<at::Tensor> results;
    for (size_t i = 0; i < stacks.size(); i++) {
      result.push_back(fn(stacks[i]));
    }
    return TensorNode(result);
  } else {
    bool broadcastable = true;
    size_t num_children = 0;
    for (size_t i = 0; i < args.size(); i++) {
      if (args[i].is_nested_tensor() && !args[i].is_leaf()) {
        if (num_children > 0) {
          broadcastable = broadcastable && (num_children == args[i].degree());
        } else {
          num_children = args[i].degree();
        }
      }
    }
    TORCH_CHECK(broadcastable, "Can't broadcast given nested tensors");
    std::vector<TensorNode> result;
    for (size_t i = 0; i < num_children; i++) {
      std::vector<ArgWrapper> local_args;
      for (size_t j = 0; j < args.size(); j++) {
        if (args[j].is_nested_tensor()) {
          if (args[j].nested_tensor().is_leaf()) {
            local_args.push_back(args[j]);
          } else {
            local_args.push_back(
                ArgWrapper(args[j].nested_tensor().children(i)));
          }
        } else {
          local_args.push_back(ArgWrapper(args[j].ivalue()));
        }
      }
      result.push_back(apply_jit_function(local_args, fn));
    }
    return TensorNode(result);
  }
}
THP_ListNestedTensor jit_apply_function(
    std::vector<THP_ListNestedTensor> nts_,
    py::object fn) {
  std::vector<_ListNestedTensor> nts;
  for (size_t i = 0; i < nts_.size(); i++) {
    nts.push_back(nts_[i].data());
  }
  auto sfn = py::cast<StrongFunctionPtr>(fn);
  auto tracing_state = tracer::getTracingState();
  TORCH_CHECK(!tracing_state, "doesnt support tracing");
  Function& callee = *sfn.function_;
  auto schema = callee.getSchema();
  TORCH_CHECK(
      schema.arguments().size() == nts.size(),
      "Give NestedTensors don't match function args.");
  std::vector<ArgWrapper> nested_nodes;
  for (size_t i = 0; i < nts.size(); i++) {
    nested_nodes.push_back(ArgWrapper(nts[i].get_structure()));
  }
  py::gil_scoped_release release;
  TensorNode nested_node = apply_jit_function(nested_nodes, callee);
  py::gil_scoped_acquire acquire;
  return THP_ListNestedTensor(_ListNestedTensor(nested_node));
}

py::cpp_function jit_tensorwise() {
  return py::cpp_function([](py::object fn) {
    return py::cpp_function([fn](py::args args, py::kwargs kwargs) {
      auto sfn = py::cast<StrongFunctionPtr>(fn);
      Function& f = *sfn.function_;
      std::vector<ArgWrapper> nested_nodes;
      for (size_t i = 0; i < args.size(); i++) {
        nested_nodes.push_back(ArgWrapper(
            py::cast<THP_ListNestedTensor>(args[i]).data().get_structure()));
      }
      py::gil_scoped_release release;
      TensorNode result = apply_jit_function(nested_nodes, f);
      py::gil_scoped_acquire acquire;
      return THP_ListNestedTensor(_ListNestedTensor(result));
    });
  });
}

} // namespace nested_tensor
} // namespace torch
