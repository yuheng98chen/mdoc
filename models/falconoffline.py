from models.base_model import BaseModel
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer
)
from pathlib import Path
import torch
import os

from pathlib import Path
os.environ["TRANSFORMERS_OFFLINE"] = "1"



class Falcon(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        model_dir = Path(__file__).resolve().parent
        model_path = (model_dir / config.model_id).resolve()
        print(f"[DEBUG] Falcon model path: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            local_files_only=True
        )

        self.pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0
        )

        self.create_ask_message = lambda question: {
            "role": "user",
            "content": question
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": ans
        }

    def create_text_message(self, texts, question):
        prompt = "\n".join(texts)
        return {
            "role": "user",
            "content": f"{prompt}\n{question}"
        }

    @torch.no_grad()
    def predict(self, question, texts=None, images=None, history=None):
        self.clean_up()
        messages = self.process_message(question, texts, images, history)

        try:
            outputs = self.pipeline(
                messages,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )
        except Exception:
            if isinstance(messages, list):
                prompt = "\n".join([m["content"] for m in messages])
            elif isinstance(messages, dict):
                prompt = messages["content"]
            else:
                prompt = str(messages)

            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )

        self.clean_up()
        return outputs[0]["generated_text"], outputs[0]

    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], str):
                return False
        return True
