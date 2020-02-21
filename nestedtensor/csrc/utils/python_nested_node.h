namespace torch {
namespace nested_tensor {

template <typename T>
struct THPNestedNode {
  THPNestedNode(NestedNode<T> size_node, std::string name)
      : _size_node(size_node), _name(name) {}
  int64_t len() {
    return _size_node.degree();
  }
  std::string str() {
    return NestedNode___str__(
        _size_node, _name, [](c10::IValue payload, const std::string& tabs) {
          std::stringstream ss;
          ss << tabs << payload;
          return ss.str();
        });
  }
  const NestedNode<T>& get_node() const {
    return _size_node;
  }
  std::string get_name() {
    return _name;
  }

  py::object unbind() {
    std::vector<py::object> result;
    for (const auto& child : _size_node.unbind()) {
      if (child.height() == 0) {
        result.push_back(wrap_nested_node(child));
      } else {
        result.push_back(py::cast(THPNestedNode<T>(child, _name)));
      }
    }
    return py::cast(result);
  }

 private:
  NestedNode<T> _size_node;
  std::string _name;
};
}
}
