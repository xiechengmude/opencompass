from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    begin="<|im_start|>",
    round=[
        dict(role="HUMAN", begin="<|im_start|>", end="<|im_end|>"),
        dict(role="BOT", begin="<|im_start|>", end="<|im_end|>", generate=True),
    ],
    eos_token_id=7
)

moe4x34b_0318_hf = [
    dict(
        abbr='xDAN-L2-DPO-moe-4x34b-0318',
        type=HuggingFaceCausalLM,
        path='xDAN2099/xDAN-L2-DPO-moe-4x34b-0318',
        tokenizer_path='xDAN2099/xDAN-L2-DPO-moe-4x34b-0318',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=_meta_template,
        max_out_len=2000,
        max_seq_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=8, num_procs=1),
        end_str='</s>',
    )
]

# models = [moe4x34b_0318_hf]