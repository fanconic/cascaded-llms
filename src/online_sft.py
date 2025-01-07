import os
from torch.optim import AdamW
import torch
from peft import LoraConfig, get_peft_model, TaskType


class OnlineSFTTrainerLoRA:
    """
    A class to store data (pseudo-labeled or expert-labeled) and update the base model
    using LoRA (Low-Rank Adaptation).
    """

    def __init__(
        self,
        model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        learning_rate=1e-4,
        max_buffer_size=128,
        tokenizer=None,
        output_dir="lora_output",
    ):
        """
        Args:
            model: A Hugging Face model (e.g. GPT-2, LLaMA) to be LoRA-finetuned.
            lora_r: Rank for the LoRA adapters.
            lora_alpha: Scaling factor for LoRA.
            lora_dropout: Dropout rate used in LoRA layers.
            learning_rate: Learning rate for AdamW.
            max_buffer_size: Maximum # of examples in the replay buffer
                             before we do a training step.
            tokenizer: matching tokenizer.
            output_dir: Directory where the LoRA adapters will be saved.
        """
        self.output_dir = output_dir

        # Convert the base model into a LoRA-enabled model using PEFT
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=TaskType.CAUSAL_LM,  # Typically 'CAUSAL_LM' or 'SEQ_CLS' etc.
        )
        self.model = get_peft_model(model, lora_config)
        self.tokenizer = tokenizer

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Optional: a scheduler if you prefer
        self.scheduler = None

        # Replay buffer for training data
        # We store tuples of (prompt, label_text) or (prompt, pseudo_label)
        self.replay_buffer = []
        self.max_buffer_size = max_buffer_size

    def add_example(self, prompt_text, label_text, pseudo_label_weight=None):
        """
        Add a new training example to the replay buffer.
        label_text can be an expert label or a large-model pseudo-label.
        """
        self.replay_buffer.append((prompt_text, label_text, pseudo_label_weight))
        # If we exceed the buffer size, do a training step
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.train_step()
            self.replay_buffer = []

    def train_step(self):
        """
        Run a single training step on the replay buffer data, weighting pseudo-label loss.
        """
        if not self.replay_buffer:
            return

        self.base_model.train()

        # Separate pseudo-labeled and ground-truth examples
        pseudo_examples, pseudo_weights = [
            ((p, l), w) for (p, l, w) in self.replay_buffer if w
        ]
        pseudo_weights = torch.Tensor(pseudo_weights).to(self.model.device)
        gt_examples = [(p, l) for (p, l, w) in self.replay_buffer if not w]

        total_loss = 0.0
        total_examples = 0

        # 1. Pseudo-labeled sub-batch
        if pseudo_examples:
            pseudo_loss = self._compute_loss_for_examples(pseudo_examples)
            # Weight the pseudo-loss by self.pseudo_label_weight
            weighted_pseudo_loss = pseudo_weights * pseudo_loss
            total_loss += weighted_pseudo_loss.item()  # keep track for logging

            self.optimizer.zero_grad()
            weighted_pseudo_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_examples += len(pseudo_examples)
            print(
                f"[train_step] Pseudo-labeled sub-batch => Loss={pseudo_loss:.4f} (weight={self.pseudo_label_weight})"
            )

        # 2. Ground-truth sub-batch
        if gt_examples:
            gt_loss = self._compute_loss_for_examples(gt_examples)
            total_loss += gt_loss.item()

            self.optimizer.zero_grad()
            gt_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_examples += len(gt_examples)
            print(f"[train_step] Ground-truth sub-batch => Loss={gt_loss:.4f}")

        avg_loss = total_loss / max(total_examples, 1)
        print(f"[train_step] Combined Weighted Loss={avg_loss:.4f}")

    def _compute_loss_for_examples(self, examples):
        """
        Helper function that takes a list of (prompt, label_text) pairs,
        runs a forward pass, and returns the average loss.
        """
        if not examples or self.tokenizer is None:
            return torch.tensor(0.0, device=self.model.device)

        # Concatenate prompt + label for each example
        combined_texts = [p + l for (p, l) in examples]
        inputs = self.tokenizer(
            combined_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )
        return outputs.loss

    def finalize(self):
        """
        Perform a final training step if buffer is non-empty, then save LoRA adapters.
        """
        if self.replay_buffer:
            self.train_step()
            self.replay_buffer = []

        # Save the LoRA adapters
        os.makedirs(self.output_dir, exist_ok=True)
        self.base_model.save_pretrained(self.output_dir)
        print(f"[OnlineSFTTrainerLoRA] LoRA adapters saved to {self.output_dir}")
