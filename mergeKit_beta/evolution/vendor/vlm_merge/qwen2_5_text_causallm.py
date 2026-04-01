# qwen2_5_text_causallm.py

import copy
import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLTextModel,
)


class Qwen2_5_VLTextForCausalLM(PreTrainedModel):
    """
    Qwen2.5-VL 文本塔的 CausalLM Wrapper：
    - 内部 self.model: Qwen2_5_VLTextModel
    - 外面加 lm_head 做 vocab 预测
    - 继承 PreTrainedModel，支持 save_pretrained / from_pretrained
    """

    config_class = Qwen2_5_VLTextConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen2_5_VLTextConfig):
        # 注意：一定要在 super().__init__ 之前改 config
        cfg = copy.deepcopy(config)

        # 对于这个自定义架构，强制使用 eager attention，禁止 SDPA
        # 这样就不会触发 _sdpa_can_dispatch 那段报错逻辑
        if not hasattr(cfg, "attn_implementation") or cfg.attn_implementation is None:
            cfg.attn_implementation = "eager"
        cfg._attn_implementation_internal = "eager"

        super().__init__(cfg)

        self.model = Qwen2_5_VLTextModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return logits
