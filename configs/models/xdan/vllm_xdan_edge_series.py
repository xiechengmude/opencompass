from opencompass.models import VLLM

settings = [
    ('xdan-edge-3b-e3-vllm', '/data/vayu/train/LLaMA-Factory/output/xDAN-L1-Edge-3b-Allstar-Instruct-e3', 2),
    ('xdan-edge-3b-e2-vllm', '/data/vayu/train/LLaMA-Factory/output/xDAN-L1-Edge-3b-Allstar-0802/checkpoint-14112', 1),
    ('xdan-edge-3b-e15-vllm', '/data/vayu/train/LLaMA-Factory/output/xDAN-L1-Edge-3b-Allstar-0802/checkpoint-10080', 1),
    ('xdan-edge-3b-e1-vllm', '/data/vayu/train/LLaMA-Factory/output/xDAN-L1-Edge-3b-Allstar-0802/checkpoint-8064', 1),
        ]

models = []
for abbr, path, num_gpus in settings:
    models.append(
        dict(
            type=VLLM,
            abbr=abbr,
            path=path,
            model_kwargs=dict(tensor_parallel_size=num_gpus),
            max_out_len=1024,
            max_seq_len=16384,
            batch_size=16,
            generation_kwargs=dict(temperature=0),
            run_cfg=dict(num_gpus=num_gpus, num_procs=1),
        )
    )