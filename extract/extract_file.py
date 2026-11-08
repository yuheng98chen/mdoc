

import os
import os.path
from pathlib import Path
import pymupdf  
from typing import Optional
def extract(cfg_processer, resolution=144):
    pdf_path = cfg_processer.pdf_path
    extract_path = cfg_processer.extract_path
    extract_path.mkdir(parents=True, exist_ok=True)
    extract_path = str(extract_path)
    max_page = cfg_processer.config.extract.max_page

    doc_name = cfg_processer.doc_name
    image_list = []
    text_list = []

    with pymupdf.open(pdf_path) as pdf:
        for index, page in enumerate(pdf[:max_page]):
            im_file = f"{extract_path}/{doc_name}_{index}.png"
            txt_file = f"{extract_path}/{doc_name}_{index}.txt"

            if not os.path.exists(im_file):
                pix = page.get_pixmap(dpi=resolution)
                pix.save(im_file)
            image_list.append(im_file)

            if not os.path.exists(txt_file):
                text = page.get_text("text")
                with open(txt_file, "w") as f:
                    f.write(text)
            text_list.append(txt_file)

    cfg_processer.IM_FILE = lambda doc_name,index: f"{extract_path}/{doc_name}_{index}.png"
    cfg_processer.TEXT_FILE = lambda doc_name,index: f"{extract_path}/{doc_name}_{index}.txt"