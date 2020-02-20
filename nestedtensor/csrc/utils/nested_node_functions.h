#pragma once
#include <nested_node.h>

namespace torch {
namespace nested_tensor {

namespace template_utils {

template <class A>
bool equal(A first) {
  return first;
}

template <class A, class B>
bool equal(A first, B other) {
  return first == other;
}

template <class A, class B, class... C>
bool equal(A first, B second, C... other) {
  return (first == second) && equal(other...);
}

template <class A>
bool all(A first) {
  return first;
}

template <class A, class B>
bool all(A first, B other) {
  return first && other;
}

template <class A, class B, class... C>
bool all(A first, B second, C... other) {
  return (first && second) && all(other...);
}

template <class A>
bool any(A first) {
  return first;
}

template <class A, class B>
bool any(A first, B other) {
  return first || other;
}

template <class A, class B, class... C>
bool any(A first, B second, C... other) {
  return (first || second) || any(other...);
}

} // namespace template_utils

template <class... A>
inline bool shape_matches(const NestedNode<A>&... a) {
  if (!template_utils::equal(a.height()...)) {
    return false;
  }
  if (!template_utils::equal(a.degree()...)) {
    return false;
  }
  auto first_node = std::get<0>(std::forward_as_tuple(a...));
  if (first_node.is_leaf() && !template_utils::all(a.is_leaf()...)) {
    return false;
  }
  for (size_t i = 0; i < first_node.degree(); i++) {
    if (!shape_matches(a.children(i)...)) {
      return false;
    }
  }
  return true;
}

// TODO: Assuming all NestedNodes have same shape.
template <typename F, typename... B>
inline bool all(F&& fn, const NestedNode<B>&... nested_node) {
  if (template_utils::all(nested_node.is_leaf()...)) {
    return template_utils::all(std::forward<F>(fn)(nested_node.payload()...));
  }
  auto first_node = std::get<0>(std::forward_as_tuple(nested_node...));
  for (size_t i = 0; i < first_node.degree(); i++) {
    if (!all<F, B...>(std::forward<F>(fn), nested_node.children(i)...)) {
      return false;
    }
  }
  return true;
}

// TODO: Assuming all NestedNodes have same shape.
template <typename F, typename... B>
inline bool any(F&& fn, const NestedNode<B>&... nested_node) {
  if (template_utils::all(nested_node.is_leaf()...)) {
    return template_utils::any(std::forward<F>(fn)(nested_node.payload()...));
  }
  auto first_node = std::get<0>(std::forward_as_tuple(nested_node...));
  for (size_t i = 0; i < first_node.degree(); i++) {
    if (any<F, B...>(std::forward<F>(fn), nested_node.children(i)...)) {
      return true;
    }
  }
  return false;
}

} // namespace nested_tensor
} // namespace torch
