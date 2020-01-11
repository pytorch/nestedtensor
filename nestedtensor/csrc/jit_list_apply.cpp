#include <jit_list_apply.h>
#include <torch/csrc/jit/script/builtin_functions.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

namespace py = pybind11;

using namespace torch::jit;
using namespace torch::jit::script;

// TODO Expand to IValues to support generic lists?
at::Tensor run_function(Stack&& stack, Function& fn) {
  return std::move(fn(stack).toTensor());
}

at::Tensor run_function(Stack&& stack, Operation& fn) {
  fn(stack);
  return std::move(stack.front().toTensor());
}

// TODO: Assert that one arg must be a nestedtensor?
template <class F>
static TensorNode apply_jit_function(
    Stack& stack_template,
    const std::set<size_t>& tensor_node_i,
    const std::vector<TensorNode>& tensor_nodes,
    F& fn) {
  // std::cout << "t0" << std::endl;
  // for (size_t i = 0; i < tensor_nodes.size(); i++) {
  //   std::cout << "tensor_nodes[" << i << "]" << std::endl;
  // }
  // std::cout << "t1" << std::endl;
  bool all_leaf = true;
  for (const auto& node : tensor_nodes) {
    all_leaf = all_leaf && node.is_leaf();
  }
  if (all_leaf) {
    // NOTE: We assume no named tensors and no sparse variables as
    // appropriate for TorchScript.
    // TODO: Assert leaf sizes match and are non-zero, otherwise this isn't
    // a NestedTensor function.
    size_t leaf_size = tensor_nodes[0].size();
    c10::List<at::Tensor> results;
    results.reserve(leaf_size);
    for (size_t j = 0; j < leaf_size; j++) {
      Stack stack(stack_template);
      size_t ni = 0;
      for (size_t i = 0; i < stack.size(); i++) {
        if (tensor_node_i.count(i)) {
          stack[i] = tensor_nodes[ni].payload(j);
          ni++;
        }
      }
      results.push_back(run_function(std::move(stack), fn));
    }
    return TensorNode(std::move(results));
  } else {
    bool broadcastable = true;
    size_t num_children = 0;
    for (const auto& node : tensor_nodes) {
      if (!node.is_leaf()) {
        if (num_children > 0) {
          broadcastable = broadcastable && (num_children == node.degree());
        } else {
          num_children = node.degree();
        }
      }
    }
    TORCH_CHECK(broadcastable, "Can't broadcast given nested tensors");
    std::vector<TensorNode> result;
    for (size_t i = 0; i < num_children; i++) {
      std::vector<TensorNode> local_args;
      for (const auto& node : tensor_nodes) {
        if (node.is_leaf()) {
          local_args.push_back(node);
        } else {
          local_args.push_back(node.children(i));
        }
      }
      result.push_back(
          apply_jit_function<F>(stack_template, tensor_node_i, local_args, fn));
    }
    return TensorNode(result);
  }
}

c10::optional<Symbol> is_builtin(py::object fn) {
  py::object builtin_name =
      py::module::import("torch.jit").attr("_find_builtin")(fn);
  Symbol name = c10::Symbol::fromQualString(py::str(builtin_name));

  // TODO: Is there a cheaper way to do this?
  const auto& variants = getAllOperatorsFor(name);
  if (variants.size() == 0) {
    return c10::nullopt;
  }
  return name;
}

c10::optional<TensorNode> try_nested_node(
    Argument argument,
    py::object py_arg) {
  // std::cout << "1" << std::endl;
  InferredType inferred_type = tryToInferType(py_arg);
  // std::cout << "2" << std::endl;
  // Nestedtensor must not be a valid IValue
  if (inferred_type.success()) {
    // std::cout << "3" << std::endl;
    return c10::nullopt;
  }
  // std::cout << "4" << std::endl;
  if (argument.type()->kind() == TypeKind::TensorType &&
      py::isinstance<THPNestedTensor>(py_arg)) {
    // std::cout << "5" << std::endl;
    TensorNode node = py::cast<THPNestedTensor>(py_arg).get_structure();
    // std::cout << "51" << std::endl;
    return node;
  }
  // std::cout << "6" << std::endl;
  return c10::nullopt;
}

// TODO: Add support for NestedTEnsor
// TODO: Replace throw with nullopt returns
inline c10::optional<
    std::tuple<Stack, std::set<size_t>, std::vector<TensorNode>>>
my_createStackForSchema(
    const FunctionSchema& schema,
    const tuple_slice& args,
    const py::kwargs& kwargs,
    c10::optional<IValue> self) {
  size_t all_arguments = (self ? 1 : 0) + args.size() + kwargs.size();
  if (all_arguments > schema.arguments().size()) {
    // throw std::runtime_error(c10::str(
    //     schema.name(),
    //     "() expected at most ",
    //     schema.arguments().size(),
    //     " argument(s) but received ",
    //     all_arguments,
    //     " argument(s). Declaration: ",
    //     schema));
    return c10::nullopt;
  }
  Stack stack;
  stack.reserve(schema.arguments().size());

  std::set<size_t> tensor_node_i;
  std::vector<TensorNode> tensor_nodes;

  if (self) {
    // NOTE: self cannot be a NestedTensor because it cannot be an ivalue.
    push(stack, std::move(*self));
  }
  // First push all positional args.
  for (size_t i = 0; i < args.size(); i++) {
    // Use the type information from the schema to convert the PyObject.
    const auto& schema_arg = schema.arguments()[i];
    // std::cout << "schema_arg 0: " << schema_arg << std::endl;
    // std::cout << "schema_arg.type(): " << schema_arg.type() << std::endl;
    // std::cout << "schema_arg.type()->kind(): "
    //           << typeKindToString(schema_arg.type()->kind()) << std::endl;
    if (auto tensor_node = try_nested_node(schema_arg, args[i])) {
      // std::cout << "found nested tensor" << std::endl;
      tensor_nodes.push_back(*tensor_node);
      tensor_node_i.insert(i);
      push(stack, torch::jit::IValue(torch::zeros({})));
    } else {
      // TODO: Should this throw an error?
      if (schema_arg.type()->kind() != tryToInferType(args[i]).type()->kind()) {
        return c10::nullopt;
      }
      push(stack, argumentToIValue(schema, stack.size(), args[i]));
    }
  }

  // Now for every remaining non-positional argument in the schema, look for it
  // in the kwargs dict and push it if found, or use its default value if it
  // has one.
  size_t consumed_kwargs = 0;
  for (size_t i = stack.size(); i < schema.arguments().size(); ++i) {
    const auto& schema_arg = schema.arguments()[i];
    // std::cout << "schema_arg 1: " << schema_arg << std::endl;
    if (kwargs.contains(schema_arg.name().c_str())) {
      auto kwarg = kwargs[schema_arg.name().c_str()];
      if (auto tensor_node = try_nested_node(schema_arg, kwarg)) {
        tensor_nodes.push_back(*tensor_node);
        tensor_node_i.insert(i);
        push(stack, torch::jit::IValue(torch::zeros({})));
      } else {
        // TODO: Should this throw an error?
        if (schema_arg.type()->kind() != tryToInferType(kwarg).type()->kind()) {
          return c10::nullopt;
        }
        push(stack, argumentToIValue(schema, stack.size(), args[i]));
      }
      consumed_kwargs += 1;
    } else if (schema_arg.default_value()) {
      push(stack, *schema_arg.default_value());
    } else {
      // throw std::runtime_error(c10::str(
      //     schema.name(),
      //     "() is missing value for argument '",
      //     arg.name(),
      //     "'. Declaration: ",
      //     schema));
      return c10::nullopt;
    }
  }

  if (consumed_kwargs != kwargs.size()) {
    std::vector<std::string> names;
    for (const auto& kwarg : kwargs) {
      names.emplace_back(py::cast<std::string>(kwarg.first));
    }
    schema.findErrorInKwargs(names);
  }

  return std::make_tuple(stack, tensor_node_i, tensor_nodes);
}

// TODO: This should support 3 types of functions
// fn might be scripted (i.e. StrongFunctionPtr)
// fn might be a builtin (need to resolve!)
// fn might be neither, so we just dispatch to some regular python for-loops
// (not fast!)
py::cpp_function jit_tensorwise() {
  return py::cpp_function([](py::object fn) {
    return py::cpp_function([fn](py::args args, py::kwargs kwargs) {
      // // TODO: Support for no NestedTensor arguments

      if (py::isinstance<StrongFunctionPtr>(fn)) {
        auto sfn = py::cast<StrongFunctionPtr>(fn);
        Function& operation = *sfn.function_;
        if (auto pack = my_createStackForSchema(
                operation.getSchema(), args, kwargs, c10::nullopt)) {
        // std::cout << "GOT ONE 0" << std::endl;
          // py::gil_scoped_release release;
          THPNestedTensor result =
              THPNestedTensor(_ListNestedTensor(apply_jit_function(
                  std::get<0>(*pack),
                  std::get<1>(*pack),
                  std::get<2>(*pack),
                  operation)));
            // std::cout << "done 0" << std::endl;
          return result;
        }
      }
      if (auto name = is_builtin(fn)) {
        for (std::shared_ptr<Operator> op : getAllOperatorsFor(*name)) {
          if (auto pack = my_createStackForSchema(
                  op->schema(), args, kwargs, c10::nullopt)) {
        // std::cout << "GOT ONE 1" << std::endl;
            auto operation = op->getOperation();
            // py::gil_scoped_release release;
            THPNestedTensor result =
                THPNestedTensor(_ListNestedTensor(apply_jit_function(
                    std::get<0>(*pack),
                    std::get<1>(*pack),
                    std::get<2>(*pack),
                    operation)));
            // std::cout << "done 1" << std::endl;
            return result;
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
