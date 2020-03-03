#include <jit_list_apply.h>
#include <torch/csrc/jit/frontend/builtin_functions.h>
#include <torch/extension.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace nested_tensor {

namespace py = pybind11;

using namespace torch::jit;
using namespace torch::jit::script;

// TODO Expand to IValues to support generic lists?
// TODO: Assert that one arg must be a nestedtensor?
template <class F>
static TensorNode apply_jit_function(
    const std::vector<IValueNode>& tensor_nodes,
    F& fn) {
  NestedNode<std::vector<c10::IValue>> zipped = zip(tensor_nodes);
  return map(
      [&fn](std::vector<c10::IValue> stack) {
        fn(stack);
        return std::move(stack.front().toTensor());
      },
      zipped);
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

c10::optional<IValueNode> try_nested_node(
    Argument argument,
    py::object py_arg) {
  InferredType inferred_type = tryToInferType(py_arg);
  // Nestedtensor must not be a valid IValue
  if (inferred_type.success()) {
    return c10::nullopt;
  }
  if (argument.type()->kind() == TypeKind::TensorType &&
      py::isinstance<THPNestedTensor>(py_arg)) {
    auto thp = py::cast<THPNestedTensor>(py_arg);
    auto thp_node = thp.data().get_structure();
    IValueNode node = map(
        [](at::Tensor tensor) { return torch::jit::IValue(tensor); }, thp_node);
    return node;
  }
  return c10::nullopt;
}

inline c10::optional<std::vector<IValueNode>> my_createStackForSchema(
    const FunctionSchema& schema,
    const tuple_slice& args,
    const py::kwargs& kwargs,
    c10::optional<IValue> self) {
  size_t all_arguments = (self ? 1 : 0) + args.size() + kwargs.size();
  if (all_arguments > schema.arguments().size()) {
    return c10::nullopt;
  }
  std::vector<IValueNode> tensor_nodes;
  tensor_nodes.reserve(schema.arguments().size());

  if (self) {
    // NOTE: self cannot be a NestedTensor because it cannot be an ivalue.
    tensor_nodes.push_back(IValueNode(std::move(*self)));
  }
  // First push all positional args.
  for (size_t i = 0; i < args.size(); i++) {
    // Use the type information from the schema to convert the PyObject.
    const auto& schema_arg = schema.arguments()[i];
    if (auto tensor_node = try_nested_node(schema_arg, args[i])) {
      tensor_nodes.push_back(*tensor_node);
    } else {
      // TODO: This is expensive because argumentToIValue constructs an error
      // message.
      try {
        tensor_nodes.push_back(
            IValueNode(argumentToIValue(schema, i, args[i])));
      } catch (const std::runtime_error& e) {
        return c10::nullopt;
      }
    }
  }

  // Now for every remaining non-positional argument in the schema, look for it
  // in the kwargs dict and push it if found, or use its default value if it
  // has one.
  size_t consumed_kwargs = 0;
  for (size_t i = tensor_nodes.size(); i < schema.arguments().size(); ++i) {
    const auto& schema_arg = schema.arguments()[i];
    if (kwargs.contains(schema_arg.name().c_str())) {
      auto kwarg = kwargs[schema_arg.name().c_str()];
      if (auto tensor_node = try_nested_node(schema_arg, kwarg)) {
        tensor_nodes.push_back(*tensor_node);
      } else {
        // TODO: This is expensive because argumentToIValue constructs an error
        // message.
        try {
          tensor_nodes.push_back(
              IValueNode(argumentToIValue(schema, i, kwarg)));
        } catch (const std::runtime_error& e) {
          return c10::nullopt;
        }
      }
      consumed_kwargs += 1;
    } else if (schema_arg.default_value()) {
      c10::IValue def = *schema_arg.default_value();
      tensor_nodes.push_back(IValueNode(std::move(def)));
    } else {
      return c10::nullopt;
    }
  }

  if (consumed_kwargs != kwargs.size()) {
    std::vector<std::string> names;
    for (const auto& kwarg : kwargs) {
      names.emplace_back(py::cast<std::string>(kwarg.first));
    }
    try {
      schema.findErrorInKwargs(names);
    } catch (const std::runtime_error& e) {
      return c10::nullopt;
    }
  }

  return tensor_nodes;
}

// TODO: This should support 3 types of functions
// fn might be scripted (i.e. StrongFunctionPtr)
// fn might be a builtin (need to resolve!)
// fn might be neither, so we just dispatch to some regular python for-loops
// (not fast!)
// TODO: Support for no NestedTensor arguments
// NOTE: For now this is a private function
py::cpp_function jit_tensorwise() {
  return py::cpp_function([](py::object fn) {
    return py::cpp_function([fn](py::args args, py::kwargs kwargs) {
      if (py::isinstance<StrongFunctionPtr>(fn)) {
        auto sfn = py::cast<StrongFunctionPtr>(fn);
        Function& operation = *sfn.function_;
        if (auto pack = my_createStackForSchema(
                operation.getSchema(), args, kwargs, c10::nullopt)) {
          py::gil_scoped_release release;
          THPNestedTensor result = THPNestedTensor(
              NestedTensor(apply_jit_function(*pack, operation)));
          return result;
        }
      }
      if (auto name = is_builtin(fn)) {
        // TODO: Why doesn't argumentToIValue deal with NoneType for a kwarg?
        // See also
        // https://github.com/pytorch/pytorch/blob/7d630278daee00ea2db6bc01e8a2a5f160bd8e81/torch/csrc/jit/pybind_utils.h#L778
        // If out is NoneType for a builtin we'll simply remove it.
        bool out_is_none = false;
        for (const auto& kwarg : kwargs) {
          if (py::cast<std::string>(kwarg.first) == "out") {
            auto inferred_type = tryToInferType(kwarg.second);
            if (inferred_type.success() &&
                inferred_type.type()->kind() == TypeKind::NoneType) {
              out_is_none = true;
            }
          }
        }
        if (out_is_none) {
          py::dict new_kwargs;
          for (const auto& kwarg : kwargs) {
            if (py::cast<std::string>(kwarg.first) == "out") {
              continue;
            }
            new_kwargs[kwarg.first] = kwarg.second;
          }
          kwargs = py::kwargs(new_kwargs);
        }
        for (std::shared_ptr<Operator> op : getAllOperatorsFor(*name)) {
          if (auto pack = my_createStackForSchema(
                  op->schema(), args, kwargs, c10::nullopt)) {
            auto operation = op->getOperation();
            py::gil_scoped_release release;
            THPNestedTensor result = THPNestedTensor(
                NestedTensor(apply_jit_function(*pack, operation)));
            return result;
          }
        }
      }
      // TODO: Need implementation of generic python version.
      std::stringstream ss;
      ss << "FAIL! Can't find something for " << fn;
      TORCH_CHECK(false, ss.str());
    });
  });
}
} // namespace nested_tensor
} // namespace torch
