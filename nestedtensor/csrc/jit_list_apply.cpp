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
    const std::map<void*, TensorNode>& nested_nodes_map,
    Stack& stack_template,
    F& fn) {
  bool all_leaf = true;
  for (const auto& entry : nested_nodes_map) {
    all_leaf = all_leaf && entry.second.is_leaf();
  }
  if (all_leaf) {
    // NOTE: We assume no named tensors and no sparse variables as
    // appropriate for TorchScript.
    // TODO: Assert leaf sizes match and are non-zero, otherwise this isn't
    // a NestedTensor function.
    size_t leaf_size = nested_nodes_map.begin()->second.size();
    c10::List<at::Tensor> results;
    results.reserve(leaf_size);
    for (size_t j = 0; j < leaf_size; j++) {
      Stack stack(stack_template);
      // size_t ni = 0;
      for (size_t i = 0; i < stack.size(); i++) {
        if (stack[i].isTensor()) {
          void* candidate_key = stack[i].toTensor().data_ptr();
          if (nested_nodes_map.count(candidate_key)) {
            stack[i] = nested_nodes_map.find(candidate_key)->second.payload(j);
          }
        }
        // if (nested_arg_i.count(i)) {
        //   stack[i] = nested_nodes[ni].payload(j);
        //   ni++;
        // }
      }
      results.push_back(run_function(std::move(stack), fn));
    }
    return TensorNode(std::move(results));
  } else {
    bool broadcastable = true;
    size_t num_children = 0;
    for (const auto& entry : nested_nodes_map) {
      if (!entry.second.is_leaf()) {
        if (num_children > 0) {
          broadcastable =
              broadcastable && (num_children == entry.second.degree());
        } else {
          num_children = entry.second.degree();
        }
      }
    }
    TORCH_CHECK(broadcastable, "Can't broadcast given nested tensors");
    std::vector<TensorNode> result;
    for (size_t i = 0; i < num_children; i++) {
      std::map<void*, TensorNode> local_args;
      for (const auto& entry : nested_nodes_map) {
        if (entry.second.is_leaf()) {
          local_args.insert(entry);
          // local_args[entry.first] = entry.second;
        } else {
          local_args.insert({entry.first, entry.second.children(i)});
        }
      }
      result.push_back(
          apply_jit_function<F>(local_args, stack_template, fn));
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

// TODO: This should support 3 types of functions
// fn might be scripted (i.e. StrongFunctionPtr)
// fn might be a builtin (need to resolve!)
// fn might be neither, so we just dispatch to some regular python for-loops
// (not fast!)
py::cpp_function jit_tensorwise() {
  return py::cpp_function([](py::object fn) {
    return py::cpp_function([fn](py::args args_, py::kwargs kwargs_) {
      // TODO: Support for no NestedTensor arguments
      std::map<void*, TensorNode> nested_nodes_map;

      // std::cout << "processing args" << std::endl;
      py::list args_vector;
      // std::set<size_t> nested_arg_i;
      // std::vector<TensorNode> nested_nodes;
      for (size_t i = 0; i < args_.size(); i++) {
        py::object arg = args_[i];
        if (py::isinstance<THPNestedTensor>(arg)) {
          TensorNode nested_node =
              py::cast<THPNestedTensor>(arg).get_structure();
          at::Tensor first_tensor = _get_first_variable(nested_node);
          args_vector.append(first_tensor);
          // nested_nodes.emplace_back(std::move(nested_node));
          // nested_arg_i.insert(i);
          nested_nodes_map.insert({first_tensor.data_ptr(), nested_node});
        } else {
          args_vector.append(arg);
        }
      }
      py::args args = py::args(args_vector);

      // std::cout << "processing kwargs" << std::endl;
      py::dict kwargs_dict;
      for (const auto& kwarg : kwargs_) {
        // std::cout << "kwarg.first: " << kwarg.first << std::endl;
        py::handle arg = kwarg.second;
        if (py::isinstance<THPNestedTensor>(arg)) {
          TensorNode nested_node =
              py::cast<THPNestedTensor>(arg).get_structure();
          at::Tensor first_tensor = _get_first_variable(nested_node);
          kwargs_dict[kwarg.first] = first_tensor;
          nested_nodes_map.insert({first_tensor.data_ptr(), nested_node});
        } else {
          kwargs_dict[kwarg.first] = kwarg.second;
        }
      }
      py::kwargs kwargs = py::kwargs(kwargs_dict);

      if (py::isinstance<StrongFunctionPtr>(fn)) {
        auto sfn = py::cast<StrongFunctionPtr>(fn);
        Function& operation = *sfn.function_;
        Stack stack = createStackForSchema(
            operation.getSchema(), args, kwargs, c10::nullopt);
        py::gil_scoped_release release;
        THPNestedTensor result = THPNestedTensor(_ListNestedTensor(
            apply_jit_function(nested_nodes_map, stack, operation)));
        return result;
      }
      if (auto name = is_builtin(fn)) {
        Stack stack;
        for (std::shared_ptr<Operator> op : getAllOperatorsFor(*name)) {
          try {
            // std::cout << "op->schema(): " << op->schema() << std::endl;
            stack =
                createStackForSchema(op->schema(), args, kwargs, c10::nullopt);
          } catch (std::exception& e) {
            continue;
          }
          auto operation = op->getOperation();
          py::gil_scoped_release release;
          THPNestedTensor result =
              THPNestedTensor(_ListNestedTensor(apply_jit_function(
                  nested_nodes_map, stack, operation)));
          return result;
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
