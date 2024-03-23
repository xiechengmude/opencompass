from mmengine.config import read_base

with read_base():
    from ..datasets.gsm8k.gsm8k_gen import  gsm8k_datasets
    from ..datasets.math.math_gen import math_datasets
    from ..datasets.ceval.ceval_gen import ceval_datasets
    from ..datasets.mmlu.mmlu_gen import mmlu_datasets
    from ..models.xdan.xdan_hf_4x34b_0318 import moe4x34b_0318_hf
    from ..models.xdan.xdan_vllm_4x34b_0318 import moe4x34b_0318_vllm

#datasets = [*gsm8k_datasets, *math_datasets, *ceval_datasets, *mmlu_datasets]
datasets = [*gsm8k_datasets]


models = [moe4x34b_0318_vllm]