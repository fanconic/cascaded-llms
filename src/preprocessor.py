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
        prompt_template = "You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines."

        def preprocess_example(example):
            option = str(example["options"]).replace("'", "")
            prompt = f"{prompt_template}\n\nQuestion: {example['question']}\n\n Options:{option}\n\nAnswer: "

            questions = f"Question: {example['question']}\n\n Options:{option}"
            example["prompts"] = prompt
            example["questions"] = questions
            example["answer"] = example["answer_idx"]
            del example["question"]
            del example["answer_idx"]
            del example["meta_info"]
            del example["options"]
            return example

        return dataset.map(preprocess_example, batched=False)


class ARC2AIPreprocessor(DatasetPreprocessor):
    """Preprocessing logic for ARC2AI dataset."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:

        prompt_template = "Please answer with one of the option in the bracket"

        def preprocess_example(example):
            option = {
                label: text
                for text, label in zip(
                    example["choices"]["text"], example["choices"]["label"]
                )
            }
            option = str(option).replace("'", "")
            prompt = f"{prompt_template}\n\nQuestion: {example['question']}\n\n Options:{option}\n\nAnswer: "

            questions = f"Question: {example['question']}\n\n Options:{option}"
            example["prompts"] = prompt
            example["questions"] = questions
            example["answer"] = example["answerKey"]
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
        "qiaojin/PubMedQA": DatasetPreprocessor,
    }
    return preprocessors.get(dataset_name, DefaultPreprocessor)
