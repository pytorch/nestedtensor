#include <ATen/core/interned_strings.h>
#include <jit_list_apply.h>
#include <python_buffer_nested_tensor.h>
#include <python_list_nested_tensor.h>
#include <torch/csrc/jit/script/builtin_functions.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/script/sugared_value.h>

namespace torch {
namespace nested_tensor {

using namespace torch::jit;
using namespace torch::jit::script;

struct ArgWrapper {
  ArgWrapper(TensorNode nested_tensor)
      : _is_nested_tensor(true), _nested_tensor(nested_tensor) {}
  ArgWrapper(c10::IValue ivalue) : _is_nested_tensor(false), _ivalue(ivalue) {}
  ArgWrapper(std::string name, c10::IValue ivalue)
      : _name(name), _is_nested_tensor(false), _ivalue(ivalue) {}

  bool is_nested_tensor() {
    return _is_nested_tensor;
  }

  c10::IValue ivalue() {
    return _ivalue;
  }

  TensorNode nested_tensor() {
    return _nested_tensor;
  }

  std::string name() {
    return _name;
  }

 private:
  std::string _name;
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

// TODO: Write comparison operation based on a subset of Argument comparison
at::Tensor resolve_builtin(
    py::object obj,
    py::args py_args,
    py::kwargs py_kwargs) {
  py::object builtin_name =
      py::module::import("torch.jit").attr("_find_builtin")(obj);
  auto name = c10::Symbol::fromQualString(py::str(builtin_name));

  std::cout << "builtin_name: " << builtin_name << std::endl;
  std::cout << "name: " << name << std::endl;

  const auto& variants = getAllOperatorsFor(name);
  const auto& builtin_functions = getAllBuiltinFunctionsFor(name);

  std::stringstream failure_messages;
  std::vector<const FunctionSchema*> schemas;
  for (const std::shared_ptr<Operator>& op : variants) {
    schemas.push_back(&op->schema());
  }
  for (const auto method : builtin_functions) {
    method->ensure_defined();
    schemas.push_back(&method->getSchema());
  }

  // Go through each Schema candidate based on the overloads
  // The order here matters and is given by the way we construct schemas.
  // This is a subset of matchSchemas within jit/script/schema_matching.cpp
  // and only implements the argument matching based on features such as types.
  // It could eventually live in the JIT as a subcomponent that can implement
  // overload resolution generically and outside a graph context.
  //
  // In essence we spend most of our time resolving types (e.g. turn
  // single floats into lists of floats, resolving concrete types) or dealing
  // with the unordered nature of kwargs.
  for (size_t i = 0; i < schemas.size(); i++) {
    const FunctionSchema* schema = schemas[i];
    std::cout << "schema[" << i << "]:\t" << *schemas[i];
    std::cout << " - overload_name: " << schemas[i]->overload_name()
              << std::endl;
    // In the end it's only a match when this counter fully depleted the args.
    size_t py_args_i = 0;
    size_t used_kwargs = 0;
    std::vector<bool> used_kwarg(py_kwargs.size(), false);
    const std::vector<Argument>& schema_args = schema->arguments();
    std::vector<ArgWrapper> parse_py_args;
    // For each argument in the Schema, see if it can be matched up with the
    // given python arguments to determine whether it's the right overload.
    //
    // First we resolve the python arguments to build list of candidate
    // wrapped arguments. It's not enough to parse these arguments
    // outside of a given Schema because of the type environment
    // and conversions. It's possible to match a Python call
    // signature to an overload with different types such as
    // Scalar and Tensor etc. simply by requiring conversion.
    for (size_t j = 0; j < schema_args.size(); j++) {
      // TODO: Support for self as in tryMatchArgument?
      Argument schema_arg = schema_args[i];
      if (!schema_arg.kwarg_only() && py_args_i < py_args.size()) {
        // TODO: Add support to allow conversions.
        IValue type_ptr = toTypeInferredIValue(py_args[py_args_i]);
        parse_py_args.emplace_back(ArgWrapper(type_ptr));
        py_args_i++;
      } else if (py_kwargs.contains(schema_arg.name().c_str())) {
        // TODO: Check for no presence of duplicates in given schemas[i]
        py::handle py_kwarg_object = py_kwargs[schema_arg.name().c_str()];
        parse_py_args.emplace_back(ArgWrapper(
            schema_arg.name(), toTypeInferredIValue(py_kwarg_object)));
        used_kwargs++;
      } else if (schema_arg.default_value()) {
        parse_py_args.emplace_back(ArgWrapper(*schema_arg.default_value()));
      } else {
        std::cout << "FAIL" << std::endl;
      }
    }
    if (py_args_i == py_args.size() - 1 && used_kwargs == py_kwargs.size()) {
      std::cout << "WIN - ";
      std::cout << "schema: " << schema;
    }
  }
  return torch::ones({});
}

} // namespace nested_tensor
} // namespace torch
