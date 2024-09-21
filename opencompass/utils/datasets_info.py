DATASETS_MAPPING = {
    # ADVGLUE Datasets
    "opencompass/advglue-dev": {
        "ms_id": None,
        "hf_id": None,
        "local": "./data/adv_glue/dev_ann.json",
    },
    # AGIEval Datasets
    "opencompass/agieval": {
        "ms_id": "opencompass/agieval",
        "hf_id": "opencompass/agieval",
        "local": "./data/AGIEval/data/v1/",
    },
    # ARC Datasets(Test)
    "opencompass/ai2_arc-test": {
        "ms_id": "opencompass/ai2_arc",
        "hf_id": "opencompass/ai2_arc",
        "local": "./data/ARC/ARC-c/ARC-Challenge-Test.jsonl",
    },
    "opencompass/ai2_arc-dev": {
        "ms_id": "opencompass/ai2_arc",
        "hf_id": "opencompass/ai2_arc",
        "local": "./data/ARC/ARC-c/ARC-Challenge-Dev.jsonl",
    },
    "opencompass/ai2_arc-easy-dev": {
        "ms_id": "opencompass/ai2_arc",
        "hf_id": "opencompass/ai2_arc",
        "local": "./data/ARC/ARC-e/ARC-Easy-Dev.jsonl",
    },
    # BBH
    "opencompass/bbh": {
        "ms_id": "opencompass/bbh",
        "hf_id": "opencompass/bbh",
        "local": "./data/BBH/data",
    },
    # C-Eval
    "opencompass/ceval-exam": {
        "ms_id": "opencompass/ceval-exam",
        "hf_id": "opencompass/ceval-exam",
        "local": "./data/ceval/formal_ceval",
    },
    # AFQMC
    "opencompass/afqmc-dev": {
        "ms_id": "opencompass/afqmc",
        "hf_id": "opencompass/afqmc",
        "local": "./data/CLUE/AFQMC/dev.json",
    },
    # CMNLI
    "opencompass/cmnli-dev": {
        "ms_id": "opencompass/cmnli",
        "hf_id": "opencompass/cmnli",
        "local": "./data/CLUE/cmnli/cmnli_public/dev.json",
    },
    # OCNLI
    "opencompass/OCNLI-dev": {
        "ms_id": "opencompass/OCNLI",
        "hf_id": "opencompass/OCNLI",
        "local": "./data/CLUE/OCNLI/dev.json",
    },
    # ChemBench
    "opencompass/ChemBench": {
        "ms_id": "opencompass/ChemBench",
        "hf_id": "opencompass/ChemBench",
        "local": "./data/ChemBench/",
    },
    # CMMLU
    "opencompass/cmmlu": {
        "ms_id": "opencompass/cmmlu",
        "hf_id": "opencompass/cmmlu",
        "local": "./data/cmmlu/",
    },
    # CommonsenseQA
    "opencompass/commonsense_qa": {
        "ms_id": "opencompass/commonsense_qa",
        "hf_id": "opencompass/commonsense_qa",
        "local": "./data/commonsenseqa",
    },
    # CMRC
    "opencompass/cmrc_dev": {
        "ms_id": "opencompass/cmrc_dev",
        "hf_id": "opencompass/cmrc_dev",
        "local": "./data/CLUE/CMRC/dev.json",
    },
    # DRCD_dev
    "opencompass/drcd_dev": {
        "ms_id": "opencompass/drcd_dev",
        "hf_id": "opencompass/drcd_dev",
        "local": "./data/CLUE/DRCD/dev.json",
    },
    # clozeTest_maxmin
    "opencompass/clozeTest_maxmin": {
        "ms_id": None,
        "hf_id": None,
        "local": "./data/clozeTest-maxmin/python/clozeTest.json",
    },
    # clozeTest_maxmin
    "opencompass/clozeTest_maxmin_answers": {
        "ms_id": None,
        "hf_id": None,
        "local": "./data/clozeTest-maxmin/python/answers.txt",
    },
    # Flores
    "opencompass/flores": {
        "ms_id": "opencompass/flores",
        "hf_id": "opencompass/flores",
        "local": "./data/flores_first100",
    },
    # MBPP
    "opencompass/mbpp": {
        "ms_id": "opencompass/mbpp",
        "hf_id": "opencompass/mbpp",
        "local": "./data/mbpp/mbpp.jsonl",
    },
    # 'opencompass/mbpp': {
    #     'ms_id': 'opencompass/mbpp',
    #     'hf_id': 'opencompass/mbpp',
    #     'local': './data/mbpp/mbpp.jsonl',
    # },
    "opencompass/sanitized_mbpp": {
        "ms_id": "opencompass/mbpp",
        "hf_id": "opencompass/mbpp",
        "local": "./data/mbpp/sanitized-mbpp.jsonl",
    },
    # GSM
    "opencompass/gsm8k": {
        "ms_id": "opencompass/gsm8k",
        "hf_id": "opencompass/gsm8k",
        "local": "./data/gsm8k/",
    },
    # HellaSwag
    "opencompass/hellaswag": {
        "ms_id": "opencompass/hellaswag",
        "hf_id": "opencompass/hellaswag",
        "local": "./data/hellaswag/hellaswag.jsonl",
    },
    # HellaSwagICE
    "opencompass/hellaswag_ice": {
        "ms_id": "opencompass/hellaswag",
        "hf_id": "opencompass/hellaswag",
        "local": "./data/hellaswag/",
    },
    # HumanEval
    "opencompass/humaneval": {
        "ms_id": "opencompass/humaneval",
        "hf_id": "opencompass/humaneval",
        "local": "./data/humaneval/human-eval-v2-20210705.jsonl",
    },
    # HumanEvalCN
    "opencompass/humaneval_cn": {
        "ms_id": "opencompass/humaneval",
        "hf_id": "opencompass/humaneval",
        "local": "./data/humaneval_cn/human-eval-cn-v2-20210705.jsonl",
    },
    # Lambada
    "opencompass/lambada": {
        "ms_id": "opencompass/lambada",
        "hf_id": "opencompass/lambada",
        "local": "./data/lambada/test.jsonl",
    },
    # LCSTS
    "opencompass/LCSTS": {
        "ms_id": "opencompass/LCSTS",
        "hf_id": "opencompass/LCSTS",
        "local": "./data/LCSTS",
    },
    # MATH
    "opencompass/math": {
        "ms_id": "opencompass/math",
        "hf_id": "opencompass/math",
        "local": "./data/math/math.json",
    },
    # MMLU
    "opencompass/mmlu": {
        "ms_id": "opencompass/mmlu",
        "hf_id": "opencompass/mmlu",
        "local": "./data/mmlu/",
    },
    # MMLU_PRO
    "opencompass/mmlu_pro": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/mmlu_pro",
    },
    # NQ
    "opencompass/natural_question": {
        "ms_id": "opencompass/natural_question",
        "hf_id": "opencompass/natural_question",
        "local": "./data/nq/",
    },
    # OpenBook QA-test
    "opencompass/openbookqa_test": {
        "ms_id": "opencompass/openbookqa",
        "hf_id": "opencompass/openbookqa",
        "local": "./data/openbookqa/Main/test.jsonl",
    },
    # OpenBook QA-fact
    "opencompass/openbookqa_fact": {
        "ms_id": "opencompass/openbookqa",
        "hf_id": "opencompass/openbookqa",
        "local": "./data/openbookqa/Additional/test_complete.jsonl",
    },
    # PIQA
    "opencompass/piqa": {
        "ms_id": "opencompass/piqa",
        "hf_id": "opencompass/piqa",
        "local": "./data/piqa",
    },
    # RACE
    "opencompass/race": {
        "ms_id": "opencompass/race",
        "hf_id": "opencompass/race",
        "local": "./data/race/",
    },
    # SIQA
    "opencompass/siqa": {
        "ms_id": "opencompass/siqa",
        "hf_id": "opencompass/siqa",
        "local": "./data/siqa",
    },
    # XStoryCloze
    "opencompass/xstory_cloze": {
        "ms_id": "opencompass/xstory_cloze",
        "hf_id": "opencompass/xstory_cloze",
        "local": "./data/xstory_cloze",
    },
    # StrategyQA
    "opencompass/strategy_qa": {
        "ms_id": "opencompass/strategy_qa",
        "hf_id": "opencompass/strategy_qa",
        "local": "./data/strategyqa/strategyQA_train.json",
    },
    # SummEdits
    "opencompass/summedits": {
        "ms_id": "opencompass/summedits",
        "hf_id": "opencompass/summedits",
        "local": "./data/summedits/summedits.jsonl",
    },
    # SuperGLUE
    "opencompass/boolq": {
        "ms_id": "opencompass/boolq",
        "hf_id": "opencompass/boolq",
        "local": "./data/SuperGLUE/BoolQ/val.jsonl",
    },
    # TriviaQA
    "opencompass/trivia_qa": {
        "ms_id": "opencompass/trivia_qa",
        "hf_id": "opencompass/trivia_qa",
        "local": "./data/triviaqa/",
    },
    # TydiQA
    "opencompass/tydiqa": {
        "ms_id": "opencompass/tydiqa",
        "hf_id": "opencompass/tydiqa",
        "local": "./data/tydiqa/",
    },
    # Winogrande
    "opencompass/winogrande": {
        "ms_id": "opencompass/winogrande",
        "hf_id": "opencompass/winogrande",
        "local": "./data/winogrande/",
    },
    # XSum
    "opencompass/xsum": {
        "ms_id": "opencompass/xsum",
        "hf_id": "opencompass/xsum",
        "local": "./data/Xsum/dev.jsonl",
    },
    # Longbench
    "opencompass/Longbench": {
        "ms_id": "",
        "hf_id": "THUDM/LongBench",
        "local": "./data/Longbench",
    },
    # Needlebench
    "opencompass/needlebench": {
        "ms_id": "",
        "hf_id": "opencompass/needlebench",
        "local": "./data/needlebench",
    }
}

DATASETS_URL = {
    "/mmlu/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip",
        "md5": "761310671509a239e41c4b717f7fab9c",
    },
    "/gpqa/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip",
        "md5": "2e9657959030a765916f1f2aca29140d",
    },
    "/CHARM/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/CHARM.zip",
        "md5": "fdf51e955d1b8e0bb35bc1997eaf37cb",
    },
    "/ifeval/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip",
        "md5": "64d98b6f36b42e7390c9cef76cace75f",
    },
    "/mbpp/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip",
        "md5": "777739c90f04bce44096a5bc96c8f9e5",
    },
    "/cmmlu/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip",
        "md5": "a59f4003d6918509a719ce3bc2a5d5bc",
    },
    "/math/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip",
        "md5": "8b1b897259684672055e6fd4fc07c808",
    },
    "/hellaswag/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip",
        "md5": "2b700a02ffb58571c7df8d8d0619256f",
    },
    "/BBH/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip",
        "md5": "60c49f9bef5148aa7e1941328e96a554",
    },
    "/compass_arena/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/compass_arena.zip",
        "md5": "cd59b54a179d16f2a858b359b60588f6",
    },
    "/TheoremQA/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/TheoremQA.zip",
        "md5": "f2793b07bc26510d507aa710d9bd8622",
    },
    "/mathbench_v1/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mathbench_v1.zip",
        "md5": "50257a910ca43d1f61a610a79fdb16b5",
    },
    "/gsm8k/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip",
        "md5": "901e5dc93a2889789a469da9850cdca8",
    },
    "/LCBench2023/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/LCBench2023.zip",
        "md5": "e1a38c94a42ad1809e9e0650476a9306",
    },
    "/humaneval/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip",
        "md5": "88b1b89dc47b7121c81da6bcd85a69c3",
    },
    "/humanevalx": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip",
        "md5": "22930355c03fb73fb5bae14b50f1deb9",
    },
    "/ds1000_data": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ds1000_data.zip",
        "md5": "1a4990aec04a2fd73ccfad12e2d43b43",
    },
    "/drop_simple_eval/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip",
        "md5": "c912afe5b4a63509851cf16e6b91830e",
    },
    "subjective/alignment_bench/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/alignment_bench.zip",
        "md5": "d8ae9a0398526479dbbcdb80fafabceb",
    },
    "subjective/alpaca_eval": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/alpaca_eval.zip",
        "md5": "d7399d63cb46c82f089447160ef49b6a",
    },
    "subjective/arena_hard": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/arena_hard.zip",
        "md5": "02cd09a482cb0f0cd9d2c2afe7a1697f",
    },
    "subjective/mtbench": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mtbench.zip",
        "md5": "d1afc0787aeac7f1f24872742e161069",
    },
    "subjective/fofo": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/fofo.zip",
        "md5": "8a302712e425e27e4292a9369df5b9d3",
    },
    "subjective/mtbench101": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mtbench101.zip",
        "md5": "5d80257bc9929ebe5cfbf6d11184b04c",
    },
    "subjective/WildBench": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/wildbench.zip",
        "md5": "b06252857f1f8f44a17b1bfca4888ff4",
    },
    "/ruler/": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ruler.zip",
        "md5": "c60bdfff3d02358067104cc1dea7c0f7",
    },
    "/scicode": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/scicode.zip",
        "md5": "9c6c64b8c70edc418f713419ea39989c",
    },
    "/commonsenseqa": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/commonsenseqa.zip",
        "md5": "c4a82fc07c81ae1462605f5d7fd2bb2e",
    },
    "FewCLUE": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/FewCLUE.zip",
        "md5": "7976e2bb0e9d885ffd3c55f7c5d4021e",
    },
    "/race": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/race.zip",
        "md5": "b758251764a264746cf45749c02363f9",
    },
    "/ARC": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ARC.zip",
        "md5": "d720629b69f1a51cfe78bf65b00b44f6",
    },
    "/SuperGLUE": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip",
        "md5": "b60904915b0b61d1a04ea52280169936",
    },
    "SQuAD2.0": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SQuAD2.0.zip",
        "md5": "1321cbf9349e1102a57d31d1b2bfdd7e",
    },
    "mmlu_pro": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu_pro.zip",
        "md5": "e3200c7380f4cea5f13c768f2815fabb",
    },
    "/Longbench": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/Longbench.zip",
        "md5": "ab0cb9e520ae5cfb899bf38b564249bb",
    },
    "/needlebench": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/needlebench.zip",
        "md5": "b546da0397746eaff4d3ff0f20d6ede2",
    },
    "/teval": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/teval.zip",
        "md5": "7628ab5891a26bf96ca17becfd044867",
    },
}
