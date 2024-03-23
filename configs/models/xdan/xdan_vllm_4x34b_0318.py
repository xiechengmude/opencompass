from opencompass.models import HuggingFaceCausalLM
from opencompass.models import VLLM


_meta_template = dict(
    begin="<|im_start|>",
    round=[
        dict(role="HUMAN", begin="<|im_start|>", end="<|im_end|>"),
        dict(role="BOT", begin="<|im_start|>", end="<|im_end|>", generate=True),
    ],
    eos_token_id=7
)

moe4x34b_0318_vllm = [
    dict(
        type=VLLM,
        abbr='xDAN-L2-DPO-moe-4x34b-0318',
        path='xDAN2099/xDAN-L2-DPO-moe-4x34b-0318',
        model_kwargs=dict(tensor_parallel_size=8),
        meta_template=_meta_template,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=2,
        generation_kwargs=dict(temperature=0.7),
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=8, num_procs=1),
    )
]