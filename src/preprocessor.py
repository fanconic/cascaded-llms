from typing import Dict, Callable
from datasets import Dataset
from omegaconf import DictConfig


class DatasetPreprocessor:
    """Base class for dataset-specific preprocessing."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:
        raise NotImplementedError("This method should be implemented by subclasses.")


class MedQAPreprocessor(DatasetPreprocessor):
    """Preprocessing logic for MedQA dataset."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:
        prompt_template = """You are a medical doctor taking the US Medical Licensing Examination. 
Answer the following multiple-choice question using step-by-step reasoning, then conclude with a final line stating the best answer.

Question: {question}

Choices:
{choice_0}
{choice_1}
{choice_2}
{choice_3}
{choice_4}

Let's reason step-by-step, then conclude with: "The best answer is: <X>"

Reasoning:
"""

        question_template="""Question: {question}

Choices:
{choice_0}
{choice_1}
{choice_2}
{choice_3}
{choice_4}
"""
        

        def preprocess_example(example):
            prompt = prompt_template.format(
                question=example['question'], 
                choice_0 = "A) " +example["options"]["A"],
                choice_1 = "B) " +example["options"]["B"],
                choice_2 = "C) " +example["options"]["C"],
                choice_3 = "D) " +example["options"]["D"],
                choice_4 = "E) " +example["options"]["E"])

            questions = question_template.format(
                question=example['question'], 
                choice_0 = "A) " +example["options"]["A"],
                choice_1 = "B) " +example["options"]["B"],
                choice_2 = "C) " +example["options"]["C"],
                choice_3 = "D) " +example["options"]["D"],
                choice_4 = "E) " +example["options"]["E"]
            )
            
            example["prompts"] = prompt
            example["questions"] = questions
            example["answer"] =example["answer_idx"]
            
            del example["question"]
            del example["answer_idx"]
            del example["meta_info"]
            del example["options"]
            return example

        return dataset.map(preprocess_example, batched=False)


class MedMCQAPreprocessor(DatasetPreprocessor):
    """Preprocessing logic for MedMCQA dataset."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:
        prompt_template = """You are a medical doctor answering realworld medical entrance exam questions.
Answer the following multiple-choice question using step-by-step reasoning, then conclude with a final line stating the best answer.

Question: {question}

Choices:
{choice_0}
{choice_1}
{choice_2}
{choice_3}

Let's reason step-by-step, then conclude with: "The best answer is: <X>"

Reasoning:
"""

        question_template="""Question: {question}

Choices:
{choice_0}
{choice_1}
{choice_2}
{choice_3}
"""
        class_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

        def preprocess_example(example):
            prompt = prompt_template.format(
                question=example['question'], 
                choice_0 = "A) " +example["opa"],
                choice_1 = "B) " +example["opb"],
                choice_2 = "C) " +example["opc"],
                choice_3 = "D) " +example["opd"]
            )

            questions = question_template.format(
                question=example['question'], 
                choice_0 = "A) " +example["opa"],
                choice_1 = "B) " +example["opb"],
                choice_2 = "C) " +example["opc"],
                choice_3 = "D) " +example["opd"]
            )
            
            example["prompts"] = prompt
            example["questions"] = questions
            example["answer"] = class_labels[example["cop"]]
            del example["question"]
            del example["id"]
            del example["cop"]
            del example["opa"]
            del example["opb"]
            del example["opc"]
            del example["opd"]
            del example["choice_type"]
            del example["exp"]
            del example["subject_name"]
            del example["topic_name"]
            return example

        return dataset.map(preprocess_example, batched=False)


class PubMedQAPreprocessor(DatasetPreprocessor):
    """Preprocessing logic for PubMedQA dataset."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:
        prompt_template = "“Your task is to answer biomedical questions using the given abstract. Only output 'yes' or 'no' as answer."

        def preprocess_example(example):
            context = (" ").join(example["context"]["contexts"])
            prompt = (
                f"{prompt_template}\n\nABSTRACT:{context}\n\n\INPUT: {example['question']}\n\n"
                + "OPTIONS: {yes, no}\n\nAnswer: "
            )

            questions = prompt
            example["prompts"] = prompt
            example["questions"] = questions
            example["answer"] = example["final_decision"]
            del example["question"]
            del example["final_decision"]
            del example["long_answer"]
            del example["pubid"]
            del example["context"]
            return example

        return dataset.map(preprocess_example, batched=False)


class ARC2AIPreprocessor(DatasetPreprocessor):
    """Preprocessing logic for ARC2AI dataset."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:

        prompt_template = """You are a helpful AI.
Answer the following multiple-choice question using step-by-step reasoning, then conclude with a final line stating the best answer.

Question: {question}

Choices:
{choice_0}
{choice_1}
{choice_2}
{choice_3}

Let's reason step-by-step, then conclude with: "The best answer is: <X>"

Reasoning:
"""

        question_template="""Question: {question}

Choices:
{choice_0}
{choice_1}
{choice_2}
{choice_3}
"""

        label2number = {
            "A": "A",
            "B": "B",
            "C": "C",
            "D": "D",
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
        }

        def preprocess_example(example):
            option = {
                label2number[label]: text
                for text, label in zip(
                    example["choices"]["text"], example["choices"]["label"]
                )
            }
            prompt = prompt_template.format(
                question=example['question'], 
                choice_0 = "A) " +option["A"],
                choice_1 = "B) " +option["B"],
                choice_2 = "C) " +option["C"],
                choice_3 = "D) " +option["D"]
            )

            questions = question_template.format(
                question=example['question'], 
                choice_0 = "A) " +option["A"],
                choice_1 = "B) " +option["B"],
                choice_2 = "C) " +option["C"],
                choice_3 = "D) " +option["D"]
            )
            example["prompts"] = prompt
            example["questions"] = questions
            example["answer"] = label2number[example["answerKey"]]
            del example["question"]
            del example["answerKey"]
            del example["choices"]
            del example["id"]
            return example

        return dataset.map(preprocess_example, batched=False)


class DefaultPreprocessor(DatasetPreprocessor):
    """Default preprocessing logic for unknown datasets."""

    prompt_template = "Please answer with one of the option in the bracket"

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:
        def preprocess_example(example):
            prompts = [f"{prompt_template}\n\nInput: {example['input']}\n\nOutput:"]
            example["prompts"] = prompts
            example["questions"] = [example["input"]]
            return example

        return dataset.map(preprocess_example, batched=False)


def get_preprocessor(dataset_name: str) -> Callable:
    """Factory to get the appropriate preprocessor class based on dataset name."""
    preprocessors = {
        "HuggingSara/medqa": MedQAPreprocessor,
        "allenai/ai2_arc": ARC2AIPreprocessor,
        "qiaojin/PubMedQA": PubMedQAPreprocessor,
        "openlifescienceai/medmcqa": MedMCQAPreprocessor,
        "Eladio/emrqa-msquad": DatasetPreprocessor,
    }
    return preprocessors.get(dataset_name, DefaultPreprocessor)
