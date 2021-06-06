import torch
import nestedtensor

# NT case query, key, value have nested_dim 1 and are of shape (bsz, tgt_len, embed_dim)


def multi_head_attention_forward(query,
                                 key,
                                 value,
                                 embed_dim_to_check,
                                 num_heads,
                                 in_proj_weight,
                                 in_proj_bias,
                                 bias_k,
                                 bias_v,
                                 add_zero_attn,
                                 dropout_p,
                                 out_proj_weight,
                                 out_proj_bias,
                                 training=True,
                                 key_padding_mask=None,
                                 need_weights=True,
                                 attn_mask=None,
                                 use_separate_proj_weight=False,
                                 q_proj_weight=None,
                                 k_proj_weight=None,
                                 v_proj_weight=None,
                                 static_k=None,
                                 static_v=None
                                 ):
    assert isinstance(query, nestedtensor.NestedTensor)
    assert isinstance(key, nestedtensor.NestedTensor)
    assert isinstance(value, nestedtensor.NestedTensor)
    assert torch.is_tensor(out_proj_weight)
    assert torch.is_tensor(out_proj_bias)

    # TODO: Explicitly unsupported flags
    assert not use_separate_proj_weight
    assert attn_mask is None
    assert key_padding_mask is None
    assert bias_k is None
    assert bias_v is None
    assert static_k is None
    assert static_v is None
    assert not add_zero_attn
    # assert not need_weights

    bsz, tgt_len, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if query is key and key is value and in_proj_weight.is_cuda:
        return torch.ops.nestedtensor.bt_min_mha(num_heads,
                                                 head_dim,
                                                 0.5,
                                                 False,
                                                 query,
                                                 query,
                                                 query,
                                                 in_proj_weight,
                                                 in_proj_bias,
                                                 scaling,
                                                 out_proj_weight,
                                                 in_proj_bias), None

    return nestedtensor.nested.nested._wrap_result(
        torch.ops.nestedtensor.min_mha(num_heads,
                                       head_dim,
                                       dropout_p,
                                       training,
                                       query._impl,
                                       key._impl,
                                       value._impl,
                                       in_proj_weight,
                                       in_proj_bias,
                                       scaling,
                                       out_proj_weight,
                                       out_proj_bias)), None
