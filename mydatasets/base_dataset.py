import json
import re
from dataclasses import dataclass
from PIL import Image
import os
from tqdm import tqdm
from datetime import datetime

@dataclass
class Content:
    image: Image
    image_path: str
    txt: str

class BaseDataset():
    def __init__(self, config, sample, extract_path):
        self.config = config
        self.sample = sample
        self.IM_FILE = lambda doc_name, index: extract_path / f"{doc_name}_{index}.png"
        self.TEXT_FILE = lambda doc_name, index: extract_path / f"{doc_name}_{index}.txt"
        self.EXTRACT_DOCUMENT_ID = lambda s: os.path.splitext(os.path.basename(s["doc_id"] if isinstance(s, dict) else s))[0]

        current_time = datetime.now()
        self.time = current_time.strftime("%Y-%m-%d-%H-%M")

    def load_data(self, use_retreival=True):
        return self.sample

    def load_sample_retrieval_data(self, sample):
        question: str = sample[self.config.question_key]
        texts = []
        images = []

        content_list = self.load_processed_content(sample, disable_load_image=True)

        # Helper function to directly extract text from various file types
        def try_load_text_from_file(file_path: str) -> str:
            if not os.path.exists(file_path):
                return ""

            ext = os.path.splitext(file_path)[-1].lower()
            try:
                if ext in [".txt", ".md"]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return f.read()
                elif ext == ".pdf":
                    from PyPDF2 import PdfReader
                    reader = PdfReader(file_path)
                    return "\n".join([page.extract_text() or "" for page in reader.pages])
                elif ext == ".docx":
                    from docx import Document
                    doc = Document(file_path)
                    return "\n".join([para.text for para in doc.paragraphs])
                elif ext == ".html":
                    from bs4 import BeautifulSoup
                    with open(file_path, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f, "html.parser")
                    return soup.get_text()
            except Exception as e:
                print(f"[ERROR] Failed to read {file_path}: {e}")
                return ""
            return ""

        # Primary logic for loading content based on configuration
        if self.config.use_mix:
            if self.config.r_mix_key in sample:
                for page in sample[self.config.r_mix_key][:self.config.top_k]:
                    if page in sample.get(self.config.r_image_key, []):
                        images.append(content_list[page].image_path)
                    if page in sample.get(self.config.r_text_key, []):
                        text = content_list[page].txt.strip()
                        texts.append(text.replace("\n", " "))
        else:
            if self.config.r_text_key in sample:
                for page in sample[self.config.r_text_key][:self.config.top_k]:
                    if page < len(content_list):
                        text = content_list[page].txt.strip()
                        texts.append(text.replace("\n", " "))

            if self.config.r_image_key in sample:
                for page in sample[self.config.r_image_key][:self.config.top_k]:
                    if page < len(content_list):
                        image_path = content_list[page].image_path
                        exists = os.path.exists(image_path)
                        print(f"[DEBUG] Checking image path: {image_path} | Exists: {exists}")
                        if exists:
                            images.append(image_path)


        # Fallback: If no texts extracted, try to read entire document directly
        if not texts:
            file_path = sample.get("pdf_path") or sample.get("doc_path")
            if file_path:
                print("[WARN] No text from content list. Trying fallback from:", file_path)
                fallback_text = try_load_text_from_file(file_path)
                if fallback_text.strip():
                    texts.append(fallback_text.replace("\n", " "))

        return question, texts, images

    def load_processed_content(self, sample: dict, disable_load_image=True) -> list[Content]:
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        doc_name, _ = os.path.splitext(doc_name) # remove extension if necessary
        content_list = []
        for page_idx in range(self.config.max_page):
            im_file = self.IM_FILE(doc_name, page_idx)
            text_file = self.TEXT_FILE(doc_name, page_idx)
            if not os.path.exists(im_file):
                break
            img = None
            if not disable_load_image:
                img = self.load_image(im_file)
            txt = self.load_txt(text_file)
            content_list.append(Content(image=img, image_path=im_file, txt=txt))
        return content_list

    def load_image(self, file):
        pil_im = Image.open(file)
        return pil_im

    def load_txt(self, file):
        max_length = self.config.max_character_per_page
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        return content[:max_length]
