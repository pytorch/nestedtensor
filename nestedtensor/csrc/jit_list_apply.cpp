#include <ATen/core/interned_strings.h>
#include <jit_list_apply.h>
#include <torch/csrc/jit/script/builtin_functions.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/script/sugared_value.h>

namespace torch {
namespace nested_tensor {

namespace py = pybind11;

using namespace torch::jit;
using namespace torch::jit::script;

// TODO Expand to IValues to support generic lists?
template <class F>
at::Tensor run_function(std::vector<c10::IValue> stack, F& fn);

template <>
at::Tensor run_function(std::vector<c10::IValue> stack, Function& fn) {
  if (DEBUG) {
    std::cout << "run_function_Function" << std::endl;
  }
  c10::IValue result = fn(stack);
  if (DEBUG) {
    std::cout << "finished result_Function" << std::endl;
  }
  return result.toTensor();
}

template <>
at::Tensor run_function(std::vector<c10::IValue> stack, Operation& fn) {
  if (DEBUG) {
    size_t i = 0;
    for (c10::IValue& ival : stack) {
      std::cout << "ival " << i << " : " << ival.tagKind() << std::endl;
      i++;
    }
    std::cout << "run_function_Operation" << std::endl;
  }
  fn(stack);
  if (DEBUG) {
    std::cout << "run_function_Operation stack finished" << std::endl;
  }
  c10::IValue result = stack.front();
  if (DEBUG) {
    std::cout << "finished result_Operation" << std::endl;
  }
  return result.toTensor();
}

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
    if (_is_nested_tensor) {
      TensorNode first_tensor_node = get_first_leaf(_nested_tensor);
      // TODO: What if this is empty?
      return c10::IValue(first_tensor_node.payload(0));
    }
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
template <class F>
static TensorNode apply_jit_function(std::vector<ArgWrapper>& args, F& fn) {
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
    // TODO: getSchema().checkAndNormalizeInputs(stack, kwargs);?
    c10::List<at::Tensor> results;
    for (size_t i = 0; i < stacks.size(); i++) {
      results.push_back(run_function<F>(stacks[i], fn));
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
      result.push_back(apply_jit_function<F>(local_args, fn));
    }
    return TensorNode(result);
  }
}

// NestedTensor taken as Tensor type
static ArgWrapper wrap_arg(
    py::object arg,
    c10::optional<c10::TypePtr> type_ptr = c10::nullopt) {
  if (py::isinstance<THPNestedTensor>(arg)) {
    TORCH_CHECK((*type_ptr)->kind() == TensorType::Kind);
    return ArgWrapper(
        py::cast<THPNestedTensor>(arg).get_structure());
  }
  if (type_ptr) {
    return ArgWrapper(toIValue(arg, *type_ptr));
  } else {
    return ArgWrapper(toTypeInferredIValue(arg));
  }
}

static std::vector<ArgWrapper> flatten_args(
    py::args args_,
    py::kwargs kwargs_) {
  std::vector<ArgWrapper> flat_args;
  for (size_t i = 0; i < args_.size(); i++) {
    flat_args.push_back(wrap_arg(args_[i]));
  }
  std::unordered_map<std::string, ArgWrapper> kwargs;
  for (const std::pair<py::handle, py::handle>& pair : kwargs_) {
    flat_args.push_back(
        wrap_arg(py::reinterpret_borrow<py::object>(pair.second)));
  }
  return flat_args;
}

template <class F>
static THPNestedTensor apply_jit_function_helper(
    std::vector<ArgWrapper>& flat_args,
    F& op) {
  py::gil_scoped_release release;
  TensorNode result = apply_jit_function(flat_args, op);
  py::gil_scoped_acquire acquire;
  return THPNestedTensor(_ListNestedTensor(result));
}

THPNestedTensor jit_apply_function(
    std::vector<THPNestedTensor> nts,
    py::object fn) {
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
  TensorNode nested_node = apply_jit_function<Function>(nested_nodes, callee);
  py::gil_scoped_acquire acquire;
  return THPNestedTensor(_ListNestedTensor(nested_node));
}

// TODO: Write separate C++ test for overloads as test cases
// TODO: Match return values!
static c10::optional<std::vector<ArgWrapper>> try_match_schema(
    const FunctionSchema* schema,
    py::args py_args,
    py::kwargs py_kwargs) {
  // const std::vector<ArgWrapper>& py_args,
  // const std::unordered_map<std::string, ArgWrapper>& py_kwargs) {
  if (DEBUG) {
    std::cout << "Checking match for schema: " << *schema << std::endl;
  }
  // In the end it's only a match when this counter fully depleted the args.
  size_t py_args_i = 0;
  size_t used_kwargs = 0;
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
    Argument schema_arg = schema_args[j];
    py::object py_arg;
    if (!schema_arg.kwarg_only() && py_args_i < py_args.size()) {
      // TODO: Add support to allow conversions.
      py_arg = py_args[py_args_i];
      py_args_i++;
    } else if (py_kwargs.contains(schema_arg.name().c_str())) {
      // TODO: Check for no presence of duplicates in given schema
      py_arg = py_kwargs[schema_arg.name().c_str()];
      used_kwargs++;
    } else if (schema_arg.default_value()) {
      // TODO: How is this converted to ScalarType if it's a int (usually)?
      // What mechanism currently does this kind of conversion.
      py_arg = toPyObject(*schema_arg.default_value());
    } else {
      // The given schema cannot find either a positional or keyword argument to
      // match against for this given schema argument. There also is no default
      // value specified for this schema argument. Therefore this schema cannot
      // be the correct overload.
      if (DEBUG) {
        std::cout << "ARGS COUNT OFF!" << std::endl;
      }
      return c10::nullopt;
    }
    // TODO: NestedTensor support
    try {
      ArgWrapper arg = wrap_arg(py_arg, schema_arg.type());
      parse_py_args.push_back(arg);
    } catch (std::exception& e) {
      if (DEBUG) {
        std::cout << "Wrap arg exception: " << e.what() << std::endl;
      }
      return c10::nullopt;
    }
  }
  if (
      // Check whether all positional arguments were matched by given Schema
      (py_args.size() == py_args_i) &&
      // Check if all kwargs were matched by given Schema
      (used_kwargs == py_kwargs.size())) {
    //    bool types_match = true;
    //    TypeEnv type_env;
    //    for (size_t j = 0; j < parse_py_args.size(); j++) {
    //      // std::cout << " ; parse_py_args[" << j
    //      //           << "]: " << type_j->str();
    //      // Now that we found that the overall schema matches, we need to
    //      check
    //      // whether the types match.
    //      // TODO: Need Subtypes and argument type conversions (e.g. convert
    //      one
    //      // float to list of floats with right number of elements).
    //      // MatchTypeReturn match =
    //      //     matchTypeVariables(schema_args[j].type(), type_j, type_env);
    //      TypePtr type_j = parse_py_args[j].ivalue().type();
    //      std::cout << " x parse_py_args[" << j << "]: " << type_j->str();
    //      std::cout << "\t=\t"
    //                << "schema_args[" << j << "]: " <<
    //                schema_args[j].type()->str();
    //      // TODO: We want to know whether the actual argument is a
    //      convertible
    //      // subtype to the one used in the schema.
    //      // TODO: Need type env?
    //      // types_match = types_match &&
    //      matchTypeVariables(schema_args[j].type(),
    //      // type_j, type_env).success();
    //      types_match =
    //          types_match && (schema_args[j].type()->kind() ==
    //          type_j->kind());
    //      std::cout << "\t types_match: " << types_match;
    //      std::cout << std::endl;
    //    }
    //    std::cout << std::endl;
    //    if (types_match) {
    if (DEBUG) {
      std::cout << "FOUND IT!" << std::endl;
    }
    return parse_py_args;
    //    }
  }
  if (DEBUG) {
    std::cout << "ARGS SIZES MISMATCHED" << std::endl;
  }
  return c10::nullopt;
}

// TODO: Write comparison operation based on a subset of Argument comparison
// TODO: Move this into jit_tensorwise and add support for all 3 cases.
// TODO: Template apply_jit_function to work with Operation and Function.
c10::optional<Symbol> is_builtin(py::object fn) {
  py::object builtin_name =
      py::module::import("torch.jit").attr("_find_builtin")(fn);
  Symbol name = c10::Symbol::fromQualString(py::str(builtin_name));

  // TODO: Is there a cheaper way to do this?
  const auto& variants = getAllOperatorsFor(name);
  const auto& builtin_functions = getAllBuiltinFunctionsFor(name);
  if (variants.size() == 0 && builtin_functions.size() == 0) {
    return c10::nullopt;
  }
  return name;
}

//  // Go through each Schema candidate based on the overloads
//  // The order here matters and is given by the way we construct schemas.
//  // This is a subset of matchSchemas within jit/script/schema_matching.cpp
//  // and only implements the argument matching based on features such as
//  types.
//  // It could eventually live in the JIT as a subcomponent that can implement
//  // overload resolution generically and outside a graph context.
//  //
//  // In essence we spend most of our time resolving types (e.g. turn
//  // single floats into lists of floats, resolving concrete types) or dealing
//  // with the unordered nature of kwargs.
//  for (size_t i = 0; i < schemas.size(); i++) {
//    if (try_match_schema(schemas[i], py_args, py_kwargs)) {
//      std::cout << "schema[" << i << "]:\t" << *schemas[i];
//      std::cout << " - overload_name: " << schemas[i]->overload_name();
//      std::cout << "WIN" << std::endl;
//    }
//  }
//  return torch::ones({});

// TODO: This should support 3 types of functions
// fn might be scripted (i.e. StrongFunctionPtr)
// fn might be a builtin (need to resolve!)
// fn might be neither, so we just dispatch to some regular python for-loops
// (not fast!)
py::cpp_function jit_tensorwise() {
  return py::cpp_function([](py::object fn) {
    return py::cpp_function([fn](py::args args, py::kwargs kwargs) {
      if (py::isinstance<StrongFunctionPtr>(fn)) {
        std::cout << "is StrongFunctionPtr" << std::endl;
        auto sfn = py::cast<StrongFunctionPtr>(fn);
        Function& op = *sfn.function_;
        std::vector<ArgWrapper> flat_args = flatten_args(args, kwargs);
        return apply_jit_function_helper<Function>(flat_args, op);
      }
      // TODO: Support for no NestedTensor arguments
      if (auto name = is_builtin(fn)) {
        for (const auto& op : getAllOperatorsFor(*name)) {
          if (auto flat_args = try_match_schema(&op->schema(), args, kwargs)) {
            if (DEBUG) {
              std::cout << "is builtin Operation with schema: " << op->schema()
                        << std::endl;
            }
            Operation actual = op->getOperation();
            return apply_jit_function_helper<Operation>(*flat_args, actual);
          }
        }
        for (const auto& op : getAllBuiltinFunctionsFor(*name)) {
          if (auto flat_args =
                  try_match_schema(&op->getSchema(), args, kwargs)) {
            if (DEBUG) {
              std::cout << "is builtin Function with schema: "
                        << op->getSchema() << std::endl;
            }
            return apply_jit_function_helper<Function>(*flat_args, *op);
          }
        }
      }
      // TODO: Need implementation of generic python version.
      std::stringstream ss;
      ss << "FAIL! Can't find something for " << fn;
      TORCH_CHECK(false, ss.str());
      TensorNode result;
      return THPNestedTensor(_ListNestedTensor(result));
    });
  });
}
} // namespace nested_tensor
} // namespace torch
