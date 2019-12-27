#include <jit_list_apply.h>
#include <torch/csrc/jit/script/builtin_functions.h>

namespace torch {
namespace nested_tensor {

namespace py = pybind11;

using namespace torch::jit;
using namespace torch::jit::script;

// TODO Expand to IValues to support generic lists?
at::Tensor run_function(Stack& stack, Function& fn) {
  if (DEBUG) {
    std::cout << "run_function_Function" << std::endl;
  }
  c10::IValue result = fn(stack);
  if (DEBUG) {
    std::cout << "finished result_Function" << std::endl;
  }
  return result.toTensor();
}

at::Tensor run_function(Stack& stack, Operation& fn) {
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

// TODO: Assert that one arg must be a nestedtensor?
template <class F>
static TensorNode apply_jit_function(
    const std::vector<TensorNode>& nested_nodes,
    const std::set<size_t>& nested_arg_i,
    Stack& stack_template,
    F& fn) {
  bool all_leaf = true;
  for (const auto& nested_node : nested_nodes) {
    all_leaf = all_leaf && nested_node.is_leaf();
  }
  if (all_leaf) {
    // NOTE: We assume no named tensors and no sparse variables as
    // appropriate for TorchScript.
    // TODO: Assert leaf sizes match and are non-zero, otherwise this isn't
    // a NestedTensor function.
    size_t leaf_size = nested_nodes[0].size();
    c10::List<at::Tensor> results;
    for (size_t j = 0; j < leaf_size; j++) {
      Stack stack(stack_template);
      size_t ni = 0;
      for (size_t i = 0; i < stack.size(); i++) {
        if (nested_arg_i.count(i)) {
          stack[i] = nested_nodes[ni].payload(j);
        }
      }
      results.push_back(run_function(stack, fn));
    }
    return TensorNode(results);
  } else {
    bool broadcastable = true;
    size_t num_children = 0;
    for (const auto& nested_node : nested_nodes) {
      if (!nested_node.is_leaf()) {
        if (num_children > 0) {
          broadcastable =
              broadcastable && (num_children == nested_node.degree());
        } else {
          num_children = nested_node.degree();
        }
      }
    }
    TORCH_CHECK(broadcastable, "Can't broadcast given nested tensors");
    std::vector<TensorNode> result;
    for (size_t i = 0; i < num_children; i++) {
      std::vector<TensorNode> local_args;
      for (const auto& nested_node : nested_nodes) {
        if (nested_node.is_leaf()) {
          local_args.push_back(nested_node);
        } else {
          local_args.push_back(nested_node.children(i));
        }
      }
      result.push_back(
          apply_jit_function<F>(local_args, nested_arg_i, stack_template, fn));
    }
    return TensorNode(result);
  }
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
      // std::cout << "given args_: " << args_ << std::endl;
      // if (py::isinstance<StrongFunctionPtr>(fn)) {
      //   std::cout << "is StrongFunctionPtr" << std::endl;
      //   auto sfn = py::cast<StrongFunctionPtr>(fn);
      //   Function& op = *sfn.function_;
      //   std::vector<ArgWrapper> flat_args = flatten_args(args_, kwargs_);
      //   return apply_jit_function_helper<Function>(flat_args, op);
      // }
      // TODO: Support for no NestedTensor arguments
      if (auto name = is_builtin(fn)) {
        py::list args_vector;
        std::set<size_t> nested_arg_i;
        // std::cout << "args.size(): " << args_.size() << std::endl;
        for (size_t i = 0; i < args_.size(); i++) {
          py::object arg = args_[i];
          if (py::isinstance<THPNestedTensor>(arg)) {
            // std::cout << "assigning first tensor" << std::endl;
            args_vector.append(_get_first_variable(
                py::cast<THPNestedTensor>(arg).get_structure()));
            nested_arg_i.insert(i);
          } else {
            args_vector.append(arg);
          }
        }
        py::args args = py::args(args_vector);
        // std::cout << "new_args: " << args << std::endl;
        Stack stack;
        for (std::shared_ptr<Operator> op : getAllOperatorsFor(*name)) {
          try {
            // std::cout << "trying op->schema(): " << op->schema() << std::endl;
            stack =
                createStackForSchema(op->schema(), args, kwargs_, c10::nullopt);
          } catch (std::exception& e) {
            // std::cout << "e.what(): " << e.what() << std::endl;
            continue;
          }
          std::vector<TensorNode> nested_nodes;
          for (const auto& arg : args_) {
            if (py::isinstance<THPNestedTensor>(arg)) {
              nested_nodes.push_back(
                  py::cast<THPNestedTensor>(arg).get_structure());
            }
          }
          auto operation = op->getOperation();
          return THPNestedTensor(_ListNestedTensor(apply_jit_function(nested_nodes, nested_arg_i, stack, operation)));
          // Stack stack2(stack);
          // op->getOperation()(stack2);
          // std::cout << "return value1: "
          //           << torch::jit::createPyObjectForStack(std::move(stack2))
          //           << std::endl;
          // Stack stack3(stack);
          // op->getOperation()(stack3);
          // std::cout << "return value2: "
          //           << torch::jit::createPyObjectForStack(std::move(stack3))
          //           << std::endl;
        }
        // exit(1);
        // std::cout << "DONE createStackForSchema" << std::endl;
        // for (const auto& op : getAllOperatorsFor(*name)) {
        //   if (auto flat_args = try_match_schema(&op->schema(), args,
        //   kwargs_)) {
        //     if (DEBUG) {
        //       std::cout << "is builtin Operation with schema: " <<
        //       op->schema()
        //                 << std::endl;
        //     }
        //     Operation actual = op->getOperation();
        //     return apply_jit_function_helper<Operation>(*flat_args, actual);
        //   }
        // }
        // for (const auto& op : getAllBuiltinFunctionsFor(*name)) {
        //   if (auto flat_args =
        //           try_match_schema(&op->getSchema(), args, kwargs_)) {
        //     if (DEBUG) {
        //       std::cout << "is builtin Function with schema: "
        //                 << op->getSchema() << std::endl;
        //     }
        //     return apply_jit_function_helper<Function>(*flat_args, *op);
        //   }
        // }
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
