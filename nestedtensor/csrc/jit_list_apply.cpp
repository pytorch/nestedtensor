#include <ATen/core/interned_strings.h>
#include <jit_list_apply.h>
#include <python_buffer_nested_tensor.h>
#include <python_list_nested_tensor.h>
#include <torch/csrc/jit/script/builtin_functions.h>
#include <torch/csrc/jit/script/sugared_value.h>

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
    std::vector<ArgWrapper>& args,
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
        leaf_size = args[i].nested_tensor().size();
        break;
      }
    }
    std::vector<std::vector<IValue>> stacks(leaf_size);
    for (size_t j = 0; j < leaf_size; j++) {
      for (size_t i = 0; i < args.size(); i++) {
        if (args[i].is_nested_tensor()) {
          stacks[j].push_back(args[i].nested_tensor().payload(j));
        } else {
          stacks[j].push_back(args[i].ivalue());
        }
      }
    }
    c10::List<at::Tensor> results;
    for (size_t i = 0; i < stacks.size(); i++) {
      results.push_back(fn(stacks[i]).toTensor());
    }
    return TensorNode(results);
  } else {
    bool broadcastable = true;
    size_t num_children = 0;
    for (size_t i = 0; i < args.size(); i++) {
      if (args[i].is_nested_tensor() && !args[i].nested_tensor().is_leaf()) {
        if (num_children > 0) {
          broadcastable = broadcastable &&
              (num_children == args[i].nested_tensor().degree());
        } else {
          num_children = args[i].nested_tensor().degree();
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

// TODO: This should support 3 types of functions
// fn might be scripted (i.e. StrongFunctionPtr)
// fn might be a builtin (need to resolve!)
// fn might be neither, so we just dispatch to some regular python for-loops
// (not fast!)
py::cpp_function jit_tensorwise() {
  return py::cpp_function([](py::object fn) {
    return py::cpp_function([fn](py::args args, py::kwargs kwargs) {
      auto sfn = py::cast<StrongFunctionPtr>(fn);
      Function& f = *sfn.function_;
      std::vector<ArgWrapper> nested_nodes;
      for (size_t i = 0; i < args.size(); i++) {
        if (py::isinstance<THP_ListNestedTensor>(args[i])) {
          nested_nodes.push_back(ArgWrapper(
              py::cast<THP_ListNestedTensor>(args[i]).data().get_structure()));
        } else if (py::isinstance<THP_BufferNestedTensor>(args[i])) {
          nested_nodes.push_back(
              ArgWrapper(py::cast<THP_BufferNestedTensor>(args[i])
                             .data()
                             .get_structure()));
        } else {
          nested_nodes.push_back(ArgWrapper(toTypeInferredIValue(args[i])));
        }
      }
      py::gil_scoped_release release;
      TensorNode result = apply_jit_function(nested_nodes, f);
      py::gil_scoped_acquire acquire;
      return THP_ListNestedTensor(_ListNestedTensor(result));
    });
  });
}

// const std::vector<Function*>& w =
//     torch::jit::script::getAllBuiltinFunctionsFor(asdf->symbol);
// for (size_t i = 0; i < w.size(); i++) {
//   std::cout << w[i]->getSchema() << std::endl;
// }

void resolve_builtin(py::object obj, py::args args) {
  std::vector<TypePtr> arg_types;
  for (size_t i = 0; i < args.size(); i++) {
    arg_types.push_back(toTypeInferredIValue(args[i]).type());
  }
  for (size_t i = 0; i < arg_types.size(); i++) {
    std::cout << "\targ_types[" << i << "]: " << arg_types[i]->str();
  }
  std::cout << std::endl;
  py::object builtin_name =
      py::module::import("torch.jit").attr("_find_builtin")(obj);
  auto builtin = std::make_shared<torch::jit::script::BuiltinFunction>(
      c10::Symbol::fromQualString(py::str(builtin_name)), c10::nullopt);
  const std::vector<std::shared_ptr<Operator>>& ops =
      torch::jit::getAllOperatorsFor(builtin->symbol);
  std::vector<std::vector<TypePtr>> candidate_arg_types;
  for (size_t i = 0; i < ops.size(); i++) {
    const std::vector<Argument>& op_args = ops[i]->schema().arguments();
    for (size_t j = 0; j < op_args.size(); j++) {
      std::cout << "args[" << j << "]: " << op_args[j].type()->str();
    }
    std::cout << std::endl;

    if (op_args.size() != arg_types.size()) {
      continue;
    }
    bool match = true;
    for (size_t j = 0; j < op_args.size(); j++) {
      match = match && (op_args[j].type()->kind() == arg_types[j]->kind());
    }
    if (match) {
      for (size_t j = 0; j < op_args.size(); j++) {
        std::cout << "\targs[" << j << "]: " << op_args[j].type()->str();
      }
      std::cout << std::endl;
    }
  }
}

} // namespace nested_tensor
} // namespace torch
