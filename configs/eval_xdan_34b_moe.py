from mmengine.config import read_base

with read_base():
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.agieval.agieval_gen_64afd3 import agieval_datasets
    from .datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.math.math_evaluatorv2_gen_265cce import math_datasets
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from .datasets.mbpp.sanitized_mbpp_gen_1e1056 import sanitized_mbpp_datasets
    # from .datasets.subjective.alpaca_eval.alpacav1_judgeby_gpt4 import subjective_datasets as alpacav1
    # from .datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import subjective_datasets as alpacav2
    # from .datasets.subjective.alignbench.alignbench_judgeby_critiquellm import subjective_datasets

    from .models.xdan.xdan_hf_4x34b_0318 import moe4x34b_0318_hf
    from .models.xdan.xdan_vllm_4x34b_0318 import moe4x34b_0318_vllm
    from .models.xdan.xdam_vllm_l1_agent_0322 import xdan_l1_agent_0322

#datasets = [*gsm8k_datasets, *math_datasets, *ceval_datasets, *mmlu_datasets]
datasets = [*gsm8k_datasets]


models = [xdan_l1_agent_0322]

# models = [moe4x34b_0318_vllm]