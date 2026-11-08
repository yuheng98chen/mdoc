from models.base_model import BaseModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info
from pathlib import Path
import torch
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

class Qwen2_5VL(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        model_dir = Path(__file__).resolve().parent
        model_path = (model_dir / config.model_id).resolve()
        print(f"[DEBUG] Qwen2.5-VL model path: {model_path}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced_low_0",
            local_files_only=True
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }

        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [{"type": "text", "text": ans}]
        }

    def create_text_message(self, texts, question):
        content = [{"type": "text", "text": t} for t in texts]
        content.append({"type": "text", "text": question})
        return {"role": "user", "content": content}

    def create_image_message(self, images, question):
        content = [{"type": "image", "image": p} for p in images]
        content.append({"type": "text", "text": question})
        return {"role": "user", "content": content}

    def has_format_directive(self, question):
        """
        Check if question specifies output format.
        Example keywords: json, table, markdown, bullet
        """
        lowered = question.lower()
        return any(x in lowered for x in ["json", "table", "markdown", "bullet"])

    @torch.no_grad()
    def predict(self, question, texts=None, images=None, history=None):
        self.clean_up()

        # Add natural language instruction if no format directive found
        if not self.has_format_directive(question):
            question = f"{question}\nPlease answer in natural language and plain sentences."

        messages = self.process_message(question, texts, images, history)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens
        )

        trimmed_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        messages.append(self.create_ans_message(output_text))
        self.clean_up()
        return output_text, messages

    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True
