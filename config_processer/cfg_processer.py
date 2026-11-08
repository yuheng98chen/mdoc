from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import os

@dataclass
class Content:
    image: Image
    image_path: str
    txt: str

class ConfigProcesser:
    def __init__(self, config, pdf_path: str, question: str):
        
        self.config = config
        self.pdf_path = pdf_path
        self.question = question
        self.doc_id = os.path.basename(pdf_path)
        self.doc_name = os.path.splitext(os.path.basename(pdf_path))[0]  
        self.extract_path = Path(config.extract.extract_dir) / self.doc_name
        self.IM_FILE = None
        self.TEXT_FILE = None
        self.sample = {
            "pdf_path": self.pdf_path,
            "question": self.question,
            "doc_id" : self.doc_id
        }
        
        
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
        
    def load_processed_content(self, disable_load_image=True) -> list[Content]:
        doc_name = self.doc_name
        extract_path = self.extract_path
        content_list = []

        for i in range(self.config.extract.max_page):
            txt_file = extract_path / f"{doc_name}_{i}.txt"
            img_file = extract_path / f"{doc_name}_{i}.png"

            if not txt_file.exists():
                break

            with open(txt_file, "r", encoding="utf-8") as f:
                txt = f.read()

            image = None
            if not disable_load_image and img_file.exists():
                image = Image.open(img_file).convert("RGB")

            content_list.append(Content(image=image, image_path=str(img_file), txt=txt))

        return content_list