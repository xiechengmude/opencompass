from opencompass.models import HuggingFaceCausalLM

xdan_l1_agent_0322 = [
    # xDAN-L1-Agent-0322-e2
    dict(
        type=HuggingFaceCausalLM,
        abbr='xDAN-L1-Agent-0322-e2',
        path="xDAN2099/xDAN-L1-MixChat-v2.1-Agent-0322-e2",
        tokenizer_path='xDAN2099/xDAN-L1-MixChat-v2.1-Agent-0322-e2',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=4,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        run_cfg=dict(num_gpus=2, num_procs=1),
    ),
]