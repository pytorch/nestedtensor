# NOTES:
# Look at torch/include/ATen/Functions.h for confusing cases (i.e. unexpected argument order)
# TODO: Add pow and scalar other variants. Write templates more compactly.

HEADER = """# include <nestedtensor/csrc/BinaryOps.h>

namespace at {

using namespace torch::nested_tensor;
"""
BINARY_OP_DEFAULT = """
Tensor NestedTensor_{op}(const Tensor & self_, const Tensor & other_) {{
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) {{ return at::{op}(s, o); }}, self, other);
}}
"""

BINARY_OP = """
Tensor NestedTensor_{op}_Tensor(const Tensor & self_, const Tensor & other_) {{
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) {{ return at::{op}(s, o); }}, self, other);
}}
"""
BINARY_OP_SCALAR = """
Tensor NestedTensor_{op}_Tensor(const Tensor & self_, const Tensor & other_, const Scalar& alpha) {{
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [&alpha](Tensor s, Tensor o) {{ return at::{op}(s, o, alpha); }}, self, other);
}}
"""
BINARY_INPLACE_OP = """
Tensor & NestedTensor_{op}__Tensor(Tensor & self_, const Tensor & other_) {{
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) {{ tensor.{op}_(other); return tensor;}},
      self,
      other);
  return self_;
}}
"""
BINARY_INPLACE_OP_DEFAULT = """
Tensor & NestedTensor_{op}_(Tensor & self_, const Tensor & other_) {{
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) {{ tensor.{op}_(other); return tensor;}},
      self,
      other);
  return self_;
}}
"""
BINARY_INPLACE_OP_SCALAR = """
Tensor & NestedTensor_{op}__Tensor(Tensor & self_, const Tensor & other_, const Scalar& alpha) {{
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [&alpha](Tensor& tensor, const Tensor other) {{ tensor.{op}_(other, alpha); return tensor;}},
      self,
      other);
  return self_;
}}
"""
BINARY_OUT_OP = """
Tensor & NestedTensor_{op}_out(
const Tensor & self, 
const Tensor & other,
Tensor & out) {{
  TORCH_CHECK(
      is_nested_tensor_impl(out),
      "NT binary out variant requires NT as out argument.");
  TORCH_CHECK(
      is_nested_tensor_impl(out, self, other),
      "binary_out doesn't support non-NT arguments.")
  apply_nested_tensor(
      [](Tensor& self, Tensor& other, Tensor& out) {{
        return at::{op}_out(self, other, out);
      }},
      self, other, out);
  return out;
}}
"""
BINARY_OUT_OP_SCALAR = """
Tensor & NestedTensor_{op}_out(
const Tensor & self, 
const Tensor & other,
const Scalar& alpha,
Tensor & out) {{
  TORCH_CHECK(
      is_nested_tensor_impl(out),
      "NT binary out variant requires NT as out argument.");
  TORCH_CHECK(
      is_nested_tensor_impl(out, self, other),
      "binary_out doesn't support non-NT arguments.")
  apply_nested_tensor(
      [&alpha](Tensor& self, Tensor& other, Tensor& out) {{
        return at::{op}_out(out, self, other, alpha);
      }},
      self, other, out);
  return out;
}}
"""
BINARY_SCALAR_OP = """
Tensor NestedTensor_{op}_Scalar(const Tensor & self, const Scalar & other) {{
return self;
}}
"""
BINARY_INPLACE_SCALAR_OP = """
Tensor & NestedTensor_{op}__Scalar(Tensor & self, const Scalar & other) {{
return self;
}}
"""
BINARY_TEMPLATES = [
    BINARY_OP,
    BINARY_INPLACE_OP,
    BINARY_OUT_OP,
    BINARY_SCALAR_OP,
    BINARY_INPLACE_SCALAR_OP
]

REGISTRATION_HEADER = """
TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
"""
REGISTRATION_FOOTER = """
}
"""

FOOTER = """
} // namespace at
"""


def print_file(template_map):
    print(HEADER, end='')
    for k, v in template_map.items():
        print(v)
    print(REGISTRATION_HEADER, end='')
    for k, v in template_map.items():
        reg = "nt_impl(m, \""
        reg += k
        reg += "\", NestedTensor_"
        reg += k.replace('.', '_')
        reg += ");"
        print(reg)
    print(REGISTRATION_FOOTER, end='')
    print(FOOTER, end='')


def parse_registration_declarations(path):
    with open(path) as f:
        import hashlib
        path_hash = hashlib.md5(f.read().encode('utf-8')).hexdigest()
        # Based on PT GH master commit hash bd3c63aeeb
        if path_hash != "b1200869a8c0b75d7fdb91d6c0f5569b":
            raise RuntimeError("RegistrationDeclarations file changed again.")
    with open(path) as f:
        lines = f.read().split("\n")
    ops = []
    for line in lines:
        if "//" in line:
            declaration, schema_dict = line.split("//")
            if declaration.strip() != '':
                schema_dict = eval(schema_dict)
                schema = schema_dict['schema']
                assert schema[:6] == "aten::"
                ops.append((declaration, schema[6:]))
    return ops


def get_binary_functions():
    return [
        'add',
        'mul',
        'sub',
        'div',
        'pow',
        'atan2',
        'remainder',
    ]


TEMPLATE_MAP = {
    "mul.Tensor": BINARY_OP,
    "mul.Tensor": BINARY_OP,
    "mul_.Tensor": BINARY_INPLACE_OP,
    "mul.out": BINARY_OUT_OP,
    "mul.Scalar": BINARY_SCALAR_OP,
    "mul_.Scalar": BINARY_INPLACE_SCALAR_OP
}


def create_template_map(ops):
    template_map = {}
    for op in ops:
        op_reg, op_args = op[1].split("(", 1)
        op_args = "(" + op_args
        variant = None
        if "." in op_reg:
            op_name, variant = op_reg.split(".", 1)
        else:
            op_name = op_reg
        for b in get_binary_functions():
            if op_name == b:
                if variant is None:
                    template_map[op_reg] = BINARY_OP_DEFAULT.format(op=b)
                if variant == "Tensor":
                    if "Scalar & alpha" in op[0]:
                        template_map[op_reg] = BINARY_OP_SCALAR.format(op=b)
                    else:
                        template_map[op_reg] = BINARY_OP.format(op=b)
                if variant == "out":
                    if "Scalar & alpha" in op[0]:
                        template_map[op_reg] = BINARY_OUT_OP_SCALAR.format(op=b)
                    else:
                        template_map[op_reg] = BINARY_OUT_OP.format(op=b)
            if op_name == b + "_":
                if variant is None:
                    template_map[op_reg] = BINARY_INPLACE_OP_DEFAULT.format(op=b)
                if variant == "Tensor":
                    if "Scalar & alpha" in op[0]:
                        template_map[op_reg] = BINARY_INPLACE_OP_SCALAR.format(op=b)
                    else:
                        template_map[op_reg] = BINARY_INPLACE_OP.format(op=b)
    return template_map


if __name__ == "__main__":
    import sys
    import os
    if not os.path.exists(sys.argv[1]):
        raise RuntimeError("Must provide path as argument")
    path = os.path.abspath(sys.argv[1])
    ops = parse_registration_declarations(path)
    template_map = create_template_map(ops)
    print_file(template_map)
