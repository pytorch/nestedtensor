#include <Python.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>
// NOTE: Causes linktime error for requested symbol as_function
// #include <torch/csrc/jit/script/python_sugared_value.h>
// NOTE: torch/csrc/tensor/python_tensor.h can't be found and will raise compile
// error
// TODO: enable "to" by fixing this.
// #include <torch/csrc/autograd/utils/python_arg_parsing.h>

namespace torch {
namespace nested_tensor {

// using namespace at;
// using namespace torch::autograd;
using namespace torch::autograd::utils;
using namespace torch::jit;
// using namespace torch::jit::script;
namespace py = pybind11;

// The implicit contract is that, if there are no children, variable_node is
// defined.
struct _NestedNode {
  _NestedNode() : _payload() {}
  _NestedNode(const std::vector<_NestedNode> children)
      : _children(children), _payload() {}
  _NestedNode(c10::IValue payload) : _payload(payload) {}
  inline bool is_leaf() const {
    return _children.size() == 0;
  }
  inline c10::IValue payload() const {
    return _payload;
  }
  inline const std::vector<_NestedNode> children() const {
    return _children;
  }
  inline _NestedNode children(size_t i) const {
    return _children[i];
  }
  inline const _NestedNode* children_data(size_t i) const {
    return _children.data() + i;
  }
  inline size_t degree() const {
    return _children.size();
  }

 private:
  const std::vector<_NestedNode> _children;
  // TODO: Make this const?
  // _VariableNode _variable_node;
  c10::IValue _payload;
};

static inline int64_t _numel(const _NestedNode& meta_node) {
  if (meta_node.is_leaf()) {
    return meta_node.payload().toTensor().numel();
  } else {
    int64_t result = 0;
    for (size_t i = 0; i < meta_node.degree(); i++) {
      result += _numel(meta_node.children(i));
    }
    return result;
  }
}

static inline at::Tensor _get_first_variable(_NestedNode nested_node) {
  const _NestedNode* start = &nested_node;
  while (!start->is_leaf()) {
    start = start->children_data(0);
  }
  if (!start->payload().isNone()) {
    return start->payload().toTensor();
  } else {
    // PyObject* fake_args = PyTuple_New(0);
    // PyObject* fake_kwargs = PyDict_New();
    // TODO: Update if python_variable updates it too
    // torch::tensor
    return torch::ones({2, 2});
    // return torch::utils::legacy_tensor_ctor(
    //     torch::tensors::get_default_tensor_type_id(),
    //     torch::tensors::get_default_scalar_type(),
    //     fake_args,
    //     fake_kwargs);
  }
}

template <typename T, class F>
static inline T map(_NestedNode nested_node, F fn) {
  if (nested_node.is_leaf()) {
    // TODO: For now we assume the user doesn't want to apply her function if
    // the payload is None.
    T new_nested_node(fn(nested_node.payload()));
    return new_nested_node;
  } else {
    std::vector<T> new_children;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      new_children.push_back(T(map<T>(nested_node.children(i), fn)));
    }
    return T(new_children);
  }
}

template <class F>
static inline void apply2(
    _NestedNode nested_node1,
    _NestedNode nested_node2,
    F fn) {
  if (nested_node1.is_leaf()) {
    fn(nested_node1.payload().toTensor(), nested_node2.payload().toTensor());
  } else {
    for (size_t i = 0; i < nested_node1.degree(); i++) {
      apply2(nested_node1.children(i), nested_node2.children(i), fn);
    }
  }
}

static inline torch::autograd::Variable _NestedNode_to_tensor(
    const _NestedNode& nested_node) {
  if (nested_node.is_leaf()) {
    return nested_node.payload().toTensor();
  } else {
    std::vector<at::Tensor> variables;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      variables.push_back(_NestedNode_to_tensor(nested_node.children(i)));
    }
    return stack(variables);
  }
}

static inline bool _verify_variables(
    const torch::autograd::Variable& first_variable,
    const _NestedNode nested_node) {
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
    at::Tensor variable = nested_node.payload().toTensor();
    // TODO: Add more checks?
    valid = valid && (variable.dim() == first_variable.dim());
    valid = valid && (variable.layout() == first_variable.layout());
    valid = valid && (variable.device() == first_variable.device());
    valid = valid && (variable.dtype() == first_variable.dtype());
    valid =
        valid && (variable.requires_grad() == first_variable.requires_grad());
    // NOTE: This is a very costly check! For now we'll let this to be enabled
    // manually. valid = valid && (variable_.is_pinned() ==
    // first_variable.is_pinned());
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      valid =
          valid && _verify_variables(first_variable, nested_node.children(i));
    }
  }
  return valid;
}

// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct TORCH_API _ListNestedTensor {
  _ListNestedTensor() = delete;
  _ListNestedTensor(_NestedNode structure)
      : _structure(structure),
        _first_variable(_get_first_variable(_structure)) {
    if (__len__() > 0) {
      TORCH_CHECK(
          _verify_variables(_first_variable, _structure),
          "Tensors don't line up.");
    }
  }
  int64_t element_size() {
    return _first_variable.element_size();
  }
  _NestedNode nested_size() {
    if (nested_dim() == 0) {
      return _NestedNode(at::IntArrayRef());
    }
    return map<_NestedNode>(
        _structure, [&](c10::IValue tensor) -> at::IntArrayRef {
          return tensor.toTensor().sizes();
        });
  }
  _NestedNode nested_stride() {
    if (nested_dim() == 0) {
      return _NestedNode(at::IntArrayRef());
    }
    return map<_NestedNode>(
        _structure, [&](c10::IValue tensor) -> at::IntArrayRef {
          return tensor.toTensor().strides();
        });
  }
  _ListNestedTensor to(
      at::TensorOptions options,
      bool non_blocking,
      bool copy,
      c10::optional<c10::MemoryFormat> memory_format) {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [&](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().to(
              options, non_blocking, copy, memory_format);
        }));
  }
  _ListNestedTensor to(
      at::ScalarType dtype,
      bool non_blocking,
      bool copy,
      c10::optional<c10::MemoryFormat> memory_format) {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [&](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().to(dtype, non_blocking, copy, memory_format);
        }));
  }
  _ListNestedTensor to(
      at::Device device,
      at::ScalarType dtype,
      bool non_blocking,
      bool copy,
      c10::optional<c10::MemoryFormat> memory_format) {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [&](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().to(
              device, dtype, non_blocking, copy, memory_format);
        }));
  }
  _ListNestedTensor pin_memory() {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().pin_memory();
        }));
  }
  _ListNestedTensor grad() {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().grad();
        }));
  }
  _ListNestedTensor detach() {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().detach();
        }));
  }
  _ListNestedTensor requires_grad_(bool requires_grad) {
    return _ListNestedTensor(map<_NestedNode>(
        _structure, [requires_grad](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().requires_grad_(requires_grad);
        }));
  }
  void backward(
      _ListNestedTensor gradient,
      bool retain_graph,
      bool create_graph) {
    apply2(
        _structure,
        gradient.get_structure(),
        [retain_graph, create_graph](at::Tensor tensor1, at::Tensor tensor2) {
          tensor1.backward(tensor2, retain_graph, create_graph);
        });
  }
  int64_t __len__() {
    return _structure.degree();
  }
  std::string __str__();
  std::string __repr__();
  torch::autograd::Variable to_tensor() {
    return _NestedNode_to_tensor(_structure);
  }
  int64_t nested_dim() {
    const _NestedNode* start_structure = &_structure;
    int64_t depth = 0;
    while (!start_structure->is_leaf()) {
      depth++;
      start_structure = start_structure->children_data(0);
    }
    return depth;
  }
  at::ScalarType scalar_type() {
    return _first_variable.scalar_type();
  }
  at::Backend backend() {
    return _first_variable.type().backend();
  }
  at::Device device() {
    return _first_variable.device();
  }
  at::TensorOptions options() {
    return _first_variable.options();
  }
  bool requires_grad() {
    return _first_variable.requires_grad();
  }
  int64_t dim() {
    return _first_variable.dim() + nested_dim();
  }
  int64_t numel() {
    return _numel(_structure);
  }
  bool is_pinned() {
    return _first_variable.is_pinned();
  }
  bool is_contiguous() {
    return false;
  }
  _NestedNode get_structure() {
    return _structure;
  }
  // TODO: Implement these and call into them isntead of implementing them
  // separately in Variable dispatch functions.
  // _ListNestedTensor to - it's a pain due to the 100s of to overloads
  // py::tuple size(int64_t dim);
  // separately in Variable dispatch functions.
  // std::vector<py::object> unbind();
  // std::string __str__();
  // std::string __repr__();
  // py::tuple size(int64_t dim);

 private:
  const _NestedNode _structure;
  at::Tensor _first_variable;
};

inline PyObject* wrap_list(std::vector<PyObject*> list) {
  auto r = THPObjectPtr{PyTuple_New(list.size())};
  if (!r)
    throw python_error();
  for (size_t i = 0; i < list.size(); ++i) {
    PyTuple_SET_ITEM(r.get(), i, list[i]);
  }
  return r.release();
}

inline PyObject* wrap_nested_node(_NestedNode nested_node) {
  if (nested_node.is_leaf()) {
    return torch::jit::toPyObject(nested_node.payload()).release().ptr();
  } else {
    std::vector<PyObject*> result;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result.push_back(wrap_nested_node(nested_node.children(i)));
    }
    return wrap_list(result);
  }
}

static std::string _NestedNode___str__(const _NestedNode& nested_node) {
  std::stringstream result;
  if (nested_node.is_leaf()) {
    PyObject* objectsRepresentation =
        PyObject_Str(THPVariable_Wrap(nested_node.payload().toTensor()));
    result << THPUtils_unpackString(objectsRepresentation);
    return result.str();
  } else {
    result << "nested_tensor([";
    result << std::endl;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result << "  ";
      result << _NestedNode___str__(nested_node.children(i));
      result << ",";
      result << std::endl;
    }
    result << "])";
    return result.str();
  }
}

static inline _NestedNode _get_structure(PyObject* tensors) {
  if (THPVariable_Check(tensors)) {
    auto variable = THPVariable_Unpack(tensors);
    return _NestedNode(variable);
  } else {
    std::vector<_NestedNode> meta_nodes;
    Py_ssize_t i, n;
    n = PyObject_Length(tensors);
    PyObject* item;
    if (n < 0) {
      throw python_error();
    }
    for (i = 0; i < n; i++) {
      item = PyList_GetItem(tensors, i);
      _NestedNode node = _get_structure(item);
      meta_nodes.push_back(node);
    }
    return _NestedNode(meta_nodes);
  }
}

static inline torch::autograd::Variable _get_first_tensor(PyObject* tensors) {
  if (THPVariable_Check(tensors)) {
    return THPVariable_Unpack(tensors);
  } else {
    return _get_first_tensor(PyList_GetItem(tensors, 0));
  }
}

struct THP_ListNestedTensor {
  THP_ListNestedTensor() = delete;
  THP_ListNestedTensor(py::list list)
      : _data(_ListNestedTensor(_get_structure(list.ptr()))) {}
  THP_ListNestedTensor(_ListNestedTensor data) : _data(data) {}
  int64_t element_size() {
    return _data.element_size();
  }
  py::object nested_size() {
    return py::reinterpret_steal<py::object>(
        wrap_nested_node(_data.nested_size()));
  }
  py::object nested_stride() {
    return py::reinterpret_steal<py::object>(
        wrap_nested_node(_data.nested_stride()));
  }
  //  THP_ListNestedTensor to(py::args args, py::kwargs kwargs) {
  //    auto parsed =
  //        parse_to_conversion(args.ptr(), kwargs.ptr(), /*allow_copy*/ true);
  //    auto &device = std::get<0>(parsed);
  //    auto &scalarType = std::get<1>(parsed);
  //    auto non_blocking = std::get<2>(parsed);
  //    auto copy = std::get<3>(parsed);
  //    auto opt_memory_format = std::get<4>(parsed);
  //    if (device && device->is_cuda()) {
  //      torch::utils::cuda_lazy_init();
  //    }
  //    if (!device && !scalarType && !copy) {
  //      return *this;
  //    } else if (!device) {
  //      return THP_ListNestedTensor(
  //          _data.to(scalarType.value(), non_blocking, copy,
  //          opt_memory_format));
  //    } else if (!scalarType) {
  //      return THP_ListNestedTensor(_data.to(_data.options().device(device),
  //                                           non_blocking, copy,
  //                                           opt_memory_format));
  //    } else {
  //      return THP_ListNestedTensor(_data.to(device.value(),
  //      scalarType.value(),
  //                                           non_blocking, copy,
  //                                           opt_memory_format));
  //    }
  //  }
  THP_ListNestedTensor pin_memory() {
    return THP_ListNestedTensor(_data.pin_memory());
  }
  THP_ListNestedTensor grad() {
    return THP_ListNestedTensor(_data.grad());
  }
  THP_ListNestedTensor detach() {
    return THP_ListNestedTensor(_data.detach());
  }
  THP_ListNestedTensor requires_grad_(py::bool_ requires_grad) {
    return THP_ListNestedTensor(_data.requires_grad_(requires_grad));
  }
  // ADD
  int64_t nested_dim() {
    return _data.nested_dim();
  }
  int64_t dim() {
    return _data.dim();
  }
  bool is_contiguous() {
    return _data.is_contiguous();
  }
  bool is_pinned() {
    return _data.is_pinned();
  }
  bool requires_grad() {
    return _data.requires_grad();
  }
  int64_t numel() {
    return _data.numel();
  }
  int64_t len() {
    return _data.__len__();
  }
  at::Tensor to_tensor() {
    return _data.to_tensor();
  }
  // NOTE: Don't delete this. repr is an important concept, this
  // implementation is just faulty due to torch.Tensor.__repr__
  // TODO: Assuming that there is no difference in __str__ and __repr__ for
  // torch.Tensor.
  std::string str() {
    return _NestedNode___str__(_data.get_structure());
  }
  py::object getDtype() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getDtype(_data.scalar_type())));
  }
  py::object getLayout() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getLayout(_data.backend())));
  }
  py::object getDevice() {
    return toPyObject(_data.device());
  }
  _ListNestedTensor data() {
    return _data;
  }
  void backward(
      THP_ListNestedTensor gradient,
      bool retain_graph,
      bool create_graph) {
    _data.backward(gradient.data(), retain_graph, create_graph);
  }

 private:
  _ListNestedTensor _data;
};

static _NestedNode apply_jit_function(
    const std::vector<_NestedNode>& nested_nodes,
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
    Stack stack;
    stack.reserve(nested_nodes.size());
    for (size_t i = 0; i < nested_nodes.size(); i++) {
      push(stack, nested_nodes[i].payload().toTensor());
    }
    fn.run(stack);
    Variable result = stack.back().toTensor();
    auto result_node = _NestedNode(result);
    return result_node;
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
    std::vector<_NestedNode> result;
    for (size_t i = 0; i < num_children; i++) {
      std::vector<_NestedNode> local_args;
      for (size_t j = 0; j < nested_nodes.size(); j++) {
        if (nested_nodes[j].is_leaf()) {
          local_args.push_back(nested_nodes[j]);
        } else {
          local_args.push_back(nested_nodes[j].children(i));
        }
      }
      result.push_back(apply_jit_function(local_args, fn));
    }
    return _NestedNode(result);
  }
}
static THP_ListNestedTensor jit_apply_function(
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
  std::vector<_NestedNode> nested_nodes;
  for (size_t i = 0; i < nts.size(); i++) {
    nested_nodes.push_back(nts[i].get_structure());
  }
  py::gil_scoped_release release;
  _NestedNode nested_node = apply_jit_function(nested_nodes, callee);
  py::gil_scoped_acquire acquire;
  return THP_ListNestedTensor(_ListNestedTensor(nested_node));
}

} // namespace nested_tensor
} // namespace torch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // NOTE: Never forget about pybind return value policies
  // since you can expect transparent changes to the constiuents
  // via unbind.
  py::class_<torch::nested_tensor::THP_ListNestedTensor>(m, "_ListNestedTensor")
      .def(py::init<py::object>(), py::keep_alive<1, 2>())
      .def_property_readonly(
          "dtype", &torch::nested_tensor::THP_ListNestedTensor::getDtype)
      .def_property_readonly(
          "layout", &torch::nested_tensor::THP_ListNestedTensor::getLayout)
      .def_property_readonly(
          "device", &torch::nested_tensor::THP_ListNestedTensor::getDevice)
      .def_property_readonly(
          "requires_grad",
          &torch::nested_tensor::THP_ListNestedTensor::requires_grad)
      .def(
          "unbind",
          [](torch::nested_tensor::THP_ListNestedTensor self) {
            std::vector<py::object> result;
            if (self.nested_dim() == 1) {
              for (int64_t i = 0; i < self.len(); i++) {
                result.push_back(torch::jit::toPyObject(
                    self.data().get_structure().children(i).payload()));
              }
            } else {
              for (int64_t i = 0; i < self.len(); i++) {
                result.push_back(
                    py::cast(torch::nested_tensor::THP_ListNestedTensor(
                        torch::nested_tensor::_ListNestedTensor(
                            self.data().get_structure().children(i)))));
              }
            }
            return result;
          })
      .def("backward", &torch::nested_tensor::THP_ListNestedTensor::backward)
      .def(
          "element_size",
          &torch::nested_tensor::THP_ListNestedTensor::element_size)
      .def("numel", &torch::nested_tensor::THP_ListNestedTensor::numel)
      .def("__len__", &torch::nested_tensor::THP_ListNestedTensor::len)
      .def(
          "nested_dim", &torch::nested_tensor::THP_ListNestedTensor::nested_dim)
      .def(
          "is_contiguous",
          &torch::nested_tensor::THP_ListNestedTensor::is_contiguous)
      .def("is_pinned", &torch::nested_tensor::THP_ListNestedTensor::is_pinned)
      .def("dim", &torch::nested_tensor::THP_ListNestedTensor::dim)
      .def(
          "nested_size",
          &torch::nested_tensor::THP_ListNestedTensor::nested_size)
      .def(
          "nested_stride",
          &torch::nested_tensor::THP_ListNestedTensor::nested_stride)
      .def(
          "pin_memory", &torch::nested_tensor::THP_ListNestedTensor::pin_memory)
      .def("grad", &torch::nested_tensor::THP_ListNestedTensor::grad)
      .def("detach", &torch::nested_tensor::THP_ListNestedTensor::detach)
      .def(
          "requires_grad_",
          &torch::nested_tensor::THP_ListNestedTensor::requires_grad_)
      .def("to_tensor", &torch::nested_tensor::THP_ListNestedTensor::to_tensor)
      .def("__str__", &torch::nested_tensor::THP_ListNestedTensor::str)
      .def("__repr__", &torch::nested_tensor::THP_ListNestedTensor::str);
  // .def("to", &THP_ListNestedTensor::to);
  m.def("jit_apply_function", &torch::nested_tensor::jit_apply_function);
}
