import requests
from models.base_model import BaseModel

class Qwen2_5VL(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.api_url = "http://localhost:40911/v1/chat/completions"

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

    def has_format_directive(self, question: str) -> bool:
        import re
        q = question.lower()
        for p in (r"\bjson\b", r"\btable\b", r"\bmarkdown\b", r"\blist\b", r"\bbullet\b"):
            if re.search(p, q):
                return True
        return False

    def predict(self, question: str, texts: list[str]=None, images: list[str]=None, history: list[dict]=None):
        self.clean_up()

        # 1) Build prompt
        parts: list[str] = []

        if texts:
            for t in texts:
                parts.append(t.strip())

        if images:
            parts.append("<image> " * len(images))

        parts.append(question.strip())

        if not self.has_format_directive(question):
            parts.append("Please answer in natural language.")

        prompt = "\n\n".join(parts)

        # Prepend history if valid
        if history and self.is_valid_history(history):
            hist_lines = []
            for msg in history:
                role = msg["role"]
                text_parts = []
                for c in msg["content"]:
                    if c["type"] == "text":
                        text_parts.append(c["text"])
                    elif c["type"] == "image":
                        text_parts.append("<image>")
                 
                text = " ".join(text_parts)
                hist_lines.append(f"{role}: {text}")
            prompt = "\n\n".join(hist_lines + [prompt])

        # 2) Send request
        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_new_tokens
        }
        try:
            resp = requests.post(self.api_url, json=payload, timeout=30)
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            answer = "Sorry, I couldn't process the request right now."

        # 3) Return messages
        messages = history[:] if history else []
        messages.append(self.create_ask_message(prompt))
        messages.append(self.create_ans_message(answer))
        self.clean_up()
        return answer, messages
