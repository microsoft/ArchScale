# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""
import lighteval.tasks.default_prompts as prompt
import random
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
import sys
sys.set_int_max_str_digits(0)

latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)


# expr_gold_metric_pass1 = multilingual_extractive_match_metric(
#     language=Language.ENGLISH,
#     fallback_mode="first_match",
#     precision=5,
#     gold_extraction_target=(ExprExtractionConfig(),),
#     pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
#     aggregation_function=mean,
# )


indices_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    gold_extraction_target=(IndicesExtractionConfig(prefix_for_extraction="Letters"),),
    pred_extraction_target=(
        IndicesExtractionConfig(prefix_for_extraction="Letters"), LatexExtractionConfig(boxed_match_priority=0)
    ),
    aggregation_function=max,
)
gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)

MATH_QUERY_TEMPLATE="Please reason step by step, and put your final answer within \\boxed{{}}. {Question}"

def get_prompt(user):
    #return "<|system|>"+SYS_PROMPT+"<|end|><|user|>"+user+"<|end|><|assistant|>"
    #return "<|user|>"+user+"<|end|><|assistant|>"
    return user

def prompt_fn(line, task_name: str = None):
    """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
    problem = line["problem"]
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=problem),
        choices=[line["solution"]],
        gold_index=0,
    )


def aime_prompt_fn(line, task_name: str = None):
    problem = line["problem"]
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=problem), #f"{SYS_PROMPT}\n{problem}",
        choices=[line["answer"]],
        gold_index=0,
    )


ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*([A-D])"


GPQA_QUERY_TEMPLATE = """  
Please reason step by step, and put your final choice of one letter from A/B/C/D within \\boxed{{}}. {Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

def gpqa_prompt_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )


# Define tasks
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        #expr_gold_metric,
        Metrics.math_pass_at_1_64n,
    ],
    version=1,
)
aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        #expr_gold_metric,
        Metrics.math_pass_at_1_64n,
    ],
    version=1,
)

math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[Metrics.math_pass_at_1_8n],
    version=1,
)


# default gpqa config from lighteval
gpqa_lighteval = LightevalTaskConfig(
    name="gpqa",
    suite=["custom"],
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_main",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        #Metrics.loglikelihood_acc_single_token,
        Metrics.gpqa_instruct_pass_at_1_8n
    ],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)


# gpqa_diamond setting for R1 reproduction
gpqa_diamond = LightevalTaskConfig(
    name="gpqa_diamond",
    suite=["custom"],
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        #gpqa_metric,
        #indices_gold_metric,
        Metrics.gpqa_instruct_pass_at_1_8n
        ],
    stop_sequence=[], 
    trust_dataset=True,
    version=1,
)


# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime25)
TASKS_TABLE.append(gpqa_lighteval)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(gpqa_diamond)

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
