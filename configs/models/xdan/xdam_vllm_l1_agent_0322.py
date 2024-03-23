from opencompass.models import HuggingFaceCausalLM
from opencompass.models import VLLM



_meta_template = dict(
    begin="<|im_start|>",
    round=[
        dict(role="HUMAN", begin="<|im_start|>", end="<|im_end|>"),
        dict(role="BOT", begin="<|im_start|>", end="<|im_end|>", generate=True),
    ],
    eos_token_id=32001
)

xdan_l1_agent_0322 = [
    dict(
        type=VLLM,
        abbr='xDAN-L1-MixChat-v2.1-Agent-0322-e2',
        path='xDAN2099/xDAN-L1-MixChat-v2.1-Agent-0322-e2',
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=4096,
        batch_size=2,
        generation_kwargs=dict(temperature=0.7),
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
