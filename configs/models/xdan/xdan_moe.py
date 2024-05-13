from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)


models = [
    dict(
        abbr='xDAN-L3-8x22B-MoE-DPO-0416',
        type=HuggingFaceCausalLM,
        path='xDAN2099/xDAN-L3-8x22B-MoE-DPO-0416',
        tokenizer_path='xDAN2099/xDAN-L3-8x22B-MoE-DPO-0416',
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
        max_out_len=1024,
        max_seq_len=8192,
        batch_size=4,
        run_cfg=dict(num_gpus=4, num_procs=1),
        batch_padding=True,
    )
]
