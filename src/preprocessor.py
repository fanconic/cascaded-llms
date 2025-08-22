from typing import Dict, Callable
from datasets import Dataset, Value
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

        question_template = """Question: {question}

Choices:
{choice_0}
{choice_1}
{choice_2}
{choice_3}
{choice_4}
"""

        def preprocess_example(example):
            prompt = prompt_template.format(
                question=example["question"],
                choice_0="A) " + example["options"]["A"],
                choice_1="B) " + example["options"]["B"],
                choice_2="C) " + example["options"]["C"],
                choice_3="D) " + example["options"]["D"],
                choice_4="E) " + example["options"]["E"],
            )

            questions = question_template.format(
                question=example["question"],
                choice_0="A) " + example["options"]["A"],
                choice_1="B) " + example["options"]["B"],
                choice_2="C) " + example["options"]["C"],
                choice_3="D) " + example["options"]["D"],
                choice_4="E) " + example["options"]["E"],
            )

            example["prompts"] = prompt
            example["questions"] = questions
            example["answer"] = example["answer_idx"]

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
        prompt_template = """You are a medical doctor answering real world medical entrance exam questions.
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

        question_template = """Question: {question}

Choices:
{choice_0}
{choice_1}
{choice_2}
{choice_3}
"""
        class_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

        def preprocess_example(example):
            prompt = prompt_template.format(
                question=example["question"],
                choice_0="A) " + example["opa"],
                choice_1="B) " + example["opb"],
                choice_2="C) " + example["opc"],
                choice_3="D) " + example["opd"],
            )

            questions = question_template.format(
                question=example["question"],
                choice_0="A) " + example["opa"],
                choice_1="B) " + example["opb"],
                choice_2="C) " + example["opc"],
                choice_3="D) " + example["opd"],
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

        question_template = """Question: {question}

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
            "E": "E",
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
            "5": "E",
        }

        def preprocess_example(example):
            option = {
                label2number[label]: text
                for text, label in zip(
                    example["choices"]["text"], example["choices"]["label"]
                )
            }
            if len(option) == 3:
                prompt = prompt_template.format(
                    question=example["question"],
                    choice_0="A) " + option["A"],
                    choice_1="B) " + option["B"],
                    choice_2="C) " + option["C"],
                    choice_3="",
                    choice_4="",
                )

                questions = question_template.format(
                    question=example["question"],
                    choice_0="A) " + option["A"],
                    choice_1="B) " + option["B"],
                    choice_2="C) " + option["C"],
                    choice_3="",
                    choice_4="",
                )

            elif len(option) == 5:
                prompt = prompt_template.format(
                    question=example["question"],
                    choice_0="A) " + option["A"],
                    choice_1="B) " + option["B"],
                    choice_2="C) " + option["C"],
                    choice_3="D) " + option["D"],
                    choice_4="E) " + option["E"],
                )

                questions = question_template.format(
                    question=example["question"],
                    choice_0="A) " + option["A"],
                    choice_1="B) " + option["B"],
                    choice_2="C) " + option["C"],
                    choice_3="D) " + option["D"],
                    choice_4="E) " + option["E"],
                )

            else:
                prompt = prompt_template.format(
                    question=example["question"],
                    choice_0="A) " + option["A"],
                    choice_1="B) " + option["B"],
                    choice_2="C) " + option["C"],
                    choice_3="D) " + option["D"],
                    choice_4="",
                )

                questions = question_template.format(
                    question=example["question"],
                    choice_0="A) " + option["A"],
                    choice_1="B) " + option["B"],
                    choice_2="C) " + option["C"],
                    choice_3="D) " + option["D"],
                    choice_4="",
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


class MMLUPreprocessor(DatasetPreprocessor):
    """Preprocessing logic for MMLU (Massive Multitask Language Understanding) dataset."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:
        prompt_template = """You are an expert in {subject}.
Answer the following multiple-choice question using step-by-step reasoning, then conclude with a final line stating the best answer.

Question: {question}

Choices:
{choices}

Let's reason step-by-step, then conclude with: "The best answer is: <X>"

Reasoning:
"""

        question_template = """Question: {question}

Choices:
{choices}
"""
        
        label2number = {
            "A": "A",
            "B": "B",
            "C": "C",
            "D": "D",
            "E": "E",
            "0": "A",
            "1": "B",
            "2": "C",
            "3": "D",
            "4": "E",
        }
        
        dataset = dataset.cast_column("answer", Value("string"))
        
        def preprocess_example(example):
            choices_str = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(example["choices"])])
    
            answer_letter = label2number[str(example["answer"])]
            
            return {
                "prompts": prompt_template.format(
                    subject=example["subject"],
                    question=example["question"],
                    choices=choices_str,
                ),
                "questions": question_template.format(
                    question=example["question"],
                    choices=choices_str,
                ),
                "answer": answer_letter,
                "subject": example["subject"],
            }
            
        return dataset.map(preprocess_example, batched=False)


class MATHPreprocessor(DatasetPreprocessor):
    """Preprocessing logic for MATH dataset."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:
        prompt_template = """You are a mathematics expert.
Solve the following mathematics problem step by step, showing all your work clearly.

Problem: {problem}

Level: {level}
Subject: {subject}

Let's reason step-by-step, then conclude with: "The answer is: <X>"

Reasoning:
"""

        question_template = """Problem: {problem}"""

        def preprocess_example(example):
            prompt = prompt_template.format(
                problem=example["problem"],
                level=example["level"],
                subject=example["subject"]
            )
            
            example["prompts"] = prompt
            example["questions"] = question_template.format(
                problem=example["problem"]
            )
            example["answer"] = example["answer"]
            
            # Clean up original fields
            del example["problem"]
            del example["solution"]
            return example

        return dataset.map(preprocess_example, batched=False)


class AIMEPreprocessor(DatasetPreprocessor):
    """Preprocessing logic for AIME (American Invitational Mathematics Examination) dataset."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:
        prompt_template = """You are solving AIME (American Invitational Mathematics Examination) problems.

Problem: {problem}

Important: Always end your solution with the final answer in one of these two formats:

1. \\[
   \\boxed{{X}}.
   \\]

2. $n=\\boxed{{X}}$

where X is your integer answer between 0 and 999.

Reasoning:
"""

        question_template = """Problem: {problem}"""
        
        def preprocess_example(example):
            prompt = prompt_template.format(
                problem=example["problem"]
            )
            
            example["prompts"] = prompt
            example["questions"] = question_template.format(
                problem=example["problem"]
            )
            example["answer"] = str(example["answer"]).zfill(3)  # Ensure 3-digit format
            
            # Clean up original fields
            del example["problem"]
            del example["solution"]
            return example

        return dataset.map(preprocess_example, batched=False)


class DefaultPreprocessor(DatasetPreprocessor):
    """Default preprocessing logic for unknown datasets."""

    @staticmethod
    def preprocess(dataset: Dataset, cfg: DictConfig) -> Dataset:
        prompt_template = "Please answer with one of the option in the bracket"
        
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
        "cais/mmlu": MMLUPreprocessor,
        "nlile/hendrycks-MATH-benchmark": MATHPreprocessor,
        "AI-MO/aimo-validation-aime": AIMEPreprocessor,
    }
    return preprocessors.get(dataset_name, DefaultPreprocessor)
