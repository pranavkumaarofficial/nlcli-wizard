"""
2025.11.2
2025.11.1
4.57.1
0.24.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import torch
import importlib.util
import math
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)

import logging
logger_compiler = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger_compiler.setLevel(logging.DEBUG)

global INFERENCE_RUNS
INFERENCE_RUNS = 0

try:
    import torch._dynamo.eval_frame as torch_dynamo_eval_frame
    torch_dynamo_eval_frame._stance.stance
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_dynamo_eval_frame = None
    torch_compiler_set_stance = None
pass


from unsloth_zoo.loss_utils import (
    fused_linear_cross_entropy,
    unsloth_fused_ce_loss,
)

if UNSLOTH_STUDIO_ENABLED:
    from unsloth_zoo.loss_utils import fast_linear_cross_entropy

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


from transformers.modeling_flash_attention_utils import is_flash_attn_available

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    except:
        flash_attn_supports_top_left_mask = None
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except:
        _flash_attention_forward = None
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except:
        FlashAttentionKwargs = None
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
    except:
        flash_attn_varlen_func = None
else:
    flash_attn_supports_top_left_mask = None
    _flash_attention_forward = None
    FlashAttentionKwargs = None
    flash_attn_varlen_func = None
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 4, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': 0, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': False, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True}

from torch.nn import CrossEntropyLoss

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass


def mask_attention_mask_out(labels = None, attention_mask = None):
    if labels is not None and attention_mask is not None:
        attention_mask = attention_mask.to(device = labels.device)
        labels[attention_mask == 0] = -100
    return labels
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from transformers.models.gemma3.modeling_gemma3 import (Callable, Optional, Union, torch, nn, ACT2FN, Cache, PretrainedConfig, GenerationMixin, BaseModelOutputWithPast, ModelOutput, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, dynamic_rope_update, PreTrainedModel, can_return_tuple, Gemma3Config, Gemma3TextConfig, logger, __name__, Gemma3Model, Gemma3CausalLMOutputWithPast, Gemma3PreTrainedModel, Gemma3TextModel, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, create_masks_for_generate)

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def Gemma3MLP_forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return Gemma3MLP_forward(self, x)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def Gemma3RMSNorm_forward(self, x):
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    hidden_states_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
    output_fp32 = hidden_states_fp32 * (1.0 + self.weight.to(torch.float32))
    return output_fp32.to(x.dtype)

class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def Gemma3RotaryEmbedding_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Gemma3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


    def forward(self, x, position_ids):
        return Gemma3RotaryEmbedding_forward(self, x, position_ids)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def apply_rotary_pos_emb(q, k, cos, sin,  unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(attn_weights.dtype).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def Gemma3MultiModalProjector_forward(self, vision_outputs: torch.Tensor):
    batch_size, _, seq_length = vision_outputs.shape

    reshaped_vision_outputs = vision_outputs.transpose(1, 2)
    reshaped_vision_outputs = reshaped_vision_outputs.reshape(
        batch_size, seq_length, self.patches_per_image, self.patches_per_image
    )
    reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

    pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
    pooled_vision_outputs = pooled_vision_outputs.flatten(2)
    pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

    normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

    projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
    return projected_vision_outputs.type_as(vision_outputs)

class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        return Gemma3MultiModalProjector_forward(self, vision_outputs)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def _bidirectional_window_overlay(sliding_window: int) -> Callable[[int, int, int, int], bool]:
    """
    Enables a bidirectional mask within the sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """A token can attend to any other token if their absolute distance is within
        the (exclusive) sliding window size (distance < sliding_window)."""
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


@torch.compiler.disable(recursive = False)
@can_return_tuple
def Gemma3ForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> CausalLMOutputWithPast:
    r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

    >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (kwargs) != () and type(kwargs) is dict:
        n_items = (kwargs).get("num_items_in_batch", None) or (kwargs).get("n_items", None)
    if n_items is None:
        all_locals = locals()
        if 'loss_kwargs' in all_locals:
            __kwargs = all_locals['loss_kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None and 'kwargs' in all_locals:
            __kwargs = all_locals['kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None:
            all_locals = all_locals.values()
            for __kwargs in all_locals:
                if type(__kwargs) is dict:
                    n_items = __kwargs.get("num_items_in_batch", None)
                    if n_items is None: n_items = __kwargs.get("n_items", None)
                    break
    pass
    
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        if torch_dynamo_eval_frame is not None:
            old_stance = torch_dynamo_eval_frame._stance.stance
        else:
            old_stance = None
        if old_stance is not None and INFERENCE_RUNS == 1:
            # Skip guards and return to eager -> we still need guards!
            torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Removing compiler guards after 1 inference run. "\
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
        elif old_stance == "eager_on_recompile":
            pass
        elif old_stance == "default" and INFERENCE_RUNS > 1:
            # Reset compiler stance
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Reseting guards. "\
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
            INFERENCE_RUNS = 0
        INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    elif (() == () and () == ()) and (UNSLOTH_ENABLE_CCE) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if (self.config.final_logit_softcapping) == () else (self.config.final_logit_softcapping),
        )
    elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = labels,
            mask                 = None,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = (self.config.final_logit_softcapping) if (self.config.final_logit_softcapping) != () else 0,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.vocab_size, **kwargs)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3TextConfig
    base_model_prefix = "language_model"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return Gemma3ForCausalLM_forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, cache_position, logits_to_keep, **kwargs)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def token_type_ids_mask_function(
    token_type_ids: Optional[torch.Tensor],
    image_group_ids: Optional[torch.Tensor],
    tokens_per_image: int,
) -> Optional[Callable]:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """
    # Do not return an additional mask in this case
    if token_type_ids is None:
        return None

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # If it's 1 for both query and key/value, we are in an image block
        # NOTE: static cache shape goes beyond input seq length, while token_type_ids.shape[1] == input seq length
        # Since vmap doesn't support `if statement` we workaround it with `torch.where`
        safe_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
        token_type_ids_at_kv_idx = token_type_ids[batch_idx, safe_idx]
        token_type_ids_at_kv_idx = torch.where(kv_idx < token_type_ids.shape[1], token_type_ids_at_kv_idx, 0)

        image_group_ids_at_kv_idx = image_group_ids[batch_idx, safe_idx]
        image_group_ids_at_kv_idx = torch.where(kv_idx < image_group_ids.shape[1], image_group_ids_at_kv_idx, -1)

        is_image_block = (token_type_ids[batch_idx, q_idx] == 1) & (token_type_ids_at_kv_idx == 1)
        same_image_block = image_group_ids[batch_idx, q_idx] == image_group_ids_at_kv_idx

        # This is bidirectional attention whenever we are dealing with image tokens
        return is_image_block & same_image_block

    return inner_mask


@torch.compiler.disable(recursive = False)
def Gemma3ForConditionalGeneration_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
    >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    >>> messages = [
    ...     {
    ...         "role": "system",
    ...         "content": [
    ...             {"type": "text", "text": "You are a helpful assistant."}
    ...         ]
    ...     },
    ...     {
    ...         "role": "user", "content": [
    ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
    ...             {"type": "text", "text": "Where is the cat standing?"},
    ...         ]
    ...     },
    ... ]

    >>> inputs = processor.apply_chat_template(
    ...     messages,
    ...     tokenize=True,
    ...     return_dict=True,
    ...     return_tensors="pt",
    ...     add_generation_prompt=True
    ... )
    >>> # Generate
    >>> generate_ids = model.generate(**inputs)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
    ```
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        labels=mask_attention_mask_out(labels = labels, attention_mask = attention_mask),
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **lm_kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    all_locals = locals()
    n_items = None
    if 'loss_kwargs' in all_locals:
        __kwargs = all_locals['loss_kwargs']
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None)
            if n_items is None: n_items = __kwargs.get("n_items", None)
    if n_items is None and 'kwargs' in all_locals:
        __kwargs = all_locals['kwargs']
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None)
            if n_items is None: n_items = __kwargs.get("n_items", None)
    if n_items is None:
        all_locals = all_locals.values()
        for __kwargs in all_locals:
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
                break
    pass
    
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        if torch_dynamo_eval_frame is not None:
            old_stance = torch_dynamo_eval_frame._stance.stance
        else:
            old_stance = None
        if old_stance is not None and INFERENCE_RUNS == 1:
            # Skip guards and return to eager -> we still need guards!
            torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Removing compiler guards after 1 inference run. "\
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
        elif old_stance == "eager_on_recompile":
            pass
        elif old_stance == "default" and INFERENCE_RUNS > 1:
            # Reset compiler stance
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Reseting guards. "\
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
            INFERENCE_RUNS = 0
        INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    else:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        if attention_mask is not None:
            torch._dynamo.mark_dynamic(attention_mask, 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = labels,
            mask                 = attention_mask,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = () if () != () else 0,
        )


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Gemma3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
    )


class Gemma3ForConditionalGeneration(Gemma3PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    # Fix: https://github.com/huggingface/transformers/issues/40564
    accepts_loss_kwargs = False

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
        return Gemma3ForConditionalGeneration_forward(self, input_ids, pixel_values, attention_mask, position_ids, past_key_values, token_type_ids, cache_position, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, logits_to_keep, **lm_kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        position_ids: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # Prepare mask arguments
        mask_kwargs = {
            "config": config.get_text_config(),
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Add the token type ids mask for generate as well
        if token_type_ids is not None and input_embeds.shape[1] != 1:
            # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`

            # First find where a new image block starts: 1 if image and previous not image
            # The images cannot attend to future images, but can attend to all prev images and to itself bidirectionally
            is_image = (token_type_ids == 1).to(cache_position.device)
            new_image_start = is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
            image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
            image_group_ids = torch.where(is_image, image_group_ids, torch.full_like(token_type_ids, -1))
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device), image_group_ids, config.mm_tokens_per_image
            )

        return create_masks_for_generate(**mask_kwargs)


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

