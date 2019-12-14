#include <jit_list_apply.h>
#include <python_list_nested_tensor.h>

namespace torch {
namespace nested_tensor {
static TensorNode apply_jit_function(
    const std::vector<TensorNode>& nested_nodes,
    Function& fn) {
  bool all_leaf = true;
  for (size_t i = 0; i < nested_nodes.size(); i++) {
    all_leaf = all_leaf && nested_nodes[i].is_leaf();
  }
  if (all_leaf) {
    // NOTE: Assuming this is a pure function not a method (no self!)
    // NOTE: We assume there is only one Tensor inputs.
    // NOTE: We assume no named tensors and no sparse variables as
    // appropriate
    // for TorchScript. NOTE: We know the IValues of the argument, there is
    // no
    // need to cast around.
    c10::List<at::Tensor> result;
    for (size_t j = 0; j < nested_nodes[0].size(); j++) {
      Stack stack;
      for (size_t i = 0; i < nested_nodes.size(); i++) {
        push(stack, nested_nodes[i].payload(j));
      }
      fn.run(stack);
      result.push_back(stack.back().toTensor());
    }
    return TensorNode(result);
  } else {
    bool broadcastable = true;
    size_t num_children = 0;
    for (size_t i = 0; i < nested_nodes.size(); i++) {
      if (!nested_nodes[i].is_leaf()) {
        if (num_children > 0) {
          broadcastable =
              broadcastable && (num_children == nested_nodes[i].degree());
        } else {
          num_children = nested_nodes[i].degree();
        }
      }
    }
    TORCH_CHECK(broadcastable, "Can't broadcast given nested tensors");
    std::vector<TensorNode> result;
    for (size_t i = 0; i < num_children; i++) {
      std::vector<TensorNode> local_args;
      for (size_t j = 0; j < nested_nodes.size(); j++) {
        if (nested_nodes[j].is_leaf()) {
          local_args.push_back(nested_nodes[j]);
        } else {
          local_args.push_back(nested_nodes[j].children(i));
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
  std::vector<TensorNode> nested_nodes;
  for (size_t i = 0; i < nts.size(); i++) {
    nested_nodes.push_back(nts[i].get_structure());
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
      std::vector<TensorNode> nested_nodes;
      for (size_t i = 0; i < args.size(); i++) {
        nested_nodes.push_back(
            py::cast<THP_ListNestedTensor>(args[i]).data().get_structure());
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
