from torch.nn.init import constant_
from torch.nn.init import xavier_uniform_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch import nn, Tensor
from torch.nn.modules.module import Module
import torch
import torch.nn.functional as F
import nestedtensor

# NT case query, key, value have nested_dim 1 and are of shape (bsz, tgt_len, embed_dim)


def multi_head_attention_forward(query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 # type: Optional[Tensor]
                                 bias_k,
                                 # type: Optional[Tensor]
                                 bias_v,
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 # type: Optional[Tensor]
                                 key_padding_mask=None,
                                 need_weights=True,               # type: bool
                                 # type: Optional[Tensor]
                                 attn_mask=None,
                                 use_separate_proj_weight=False,  # type: bool
                                 # type: Optional[Tensor]
                                 q_proj_weight=None,
                                 # type: Optional[Tensor]
                                 k_proj_weight=None,
                                 # type: Optional[Tensor]
                                 v_proj_weight=None,
                                 # type: Optional[Tensor]
                                 static_k=None,
                                 # type: Optional[Tensor]
                                 static_v=None
                                 ):
    assert isinstance(query, nestedtensor.NestedTensor)
    assert isinstance(key, nestedtensor.NestedTensor)
    assert isinstance(value, nestedtensor.NestedTensor)
    bsz, tgt_len, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    assert not use_separate_proj_weight
    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = 0
    _end = embed_dim
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q = F.linear(query, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim
    _end = embed_dim * 2
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k = F.linear(key, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 2
    _end = None
    _w = in_proj_weight[_start:, :]
    if _b is not None:
        _b = _b[_start:]
    v = F.linear(value, _w, _b)
    q = q * scaling

    assert attn_mask is None
    assert key_padding_mask is None
    assert bias_k is None
    assert bias_v is None
    assert static_k is None
    assert static_v is None
    assert not add_zero_attn

    print("02q:\n", q)
    print("02k:\n", k)
    print("02v:\n", v)

    # NOTE: This is usually contiguous plus a view
    q = q.reshape(-1, -1, num_heads, head_dim).transpose(1, 2)
    if k is not None:
        k = k.reshape(-1, -1, num_heads, head_dim).transpose(1, 2)
    if v is not None:
        v = v.reshape(-1, -1, num_heads, head_dim).transpose(1, 2)

    print("12q:\n", q)
    print("12k:\n", k)
    print("12v:\n", v)

    # src_len = k.size(1)

    attn_output_weights = torch.matmul(q, k.transpose(2, 3))
    # attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    # NEXT: Softmax
    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(
        attn_output_weights, p=dropout_p, training=training)
    print('2attn_output_weights')
    print(attn_output_weights)

    attn_output = torch.matmul(attn_output_weights, v)
    # assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]

    # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # import pdb; pdb.set_trace()
    attn_output = attn_output.transpose(1, 2).reshape(-1, -1, embed_dim)
    print('2attn_output')
    print(attn_output)

    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    # if need_weights:
    #     # average attention weights over heads
    #     attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    #     return attn_output, attn_output_weights.sum(dim=1) / num_heads
    # else:
    # TODO: All relevant callsites in detr ignore the second value
    return attn_output, None


class MultiheadAttention(Module):
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ['q_proj_weight', 'k_proj_weight',
                     'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
            num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
