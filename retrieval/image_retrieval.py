#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# Force Hugging Face to offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from peft import PeftModel

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries

from .base_retrieval import BaseRetrieval

class Content:
    def __init__(self, image, image_path, txt):
        self.image = image
        self.image_path = image_path
        self.txt = txt

class ColpaliRetrieval(BaseRetrieval):
    def __init__(self, cfg_processer):
        super().__init__(cfg_processer)
        self.cfg_processer = cfg_processer
        self.cfg = cfg_processer.config
        self.pdf_path = cfg_processer.pdf_path
        self.question = cfg_processer.question
        self.sample = cfg_processer.sample
        self.doc_id = cfg_processer.doc_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_embed = None  

        # Define local paths for the base model and adapter
        current_dir = os.path.dirname(__file__)
        self.model_root = os.path.join(current_dir, "local_colpaligemma")
        self.adapter_root = os.path.join(current_dir, "local_colpali_adapter")

        # Load the base ColPali model from local files
        base_model = ColPali.from_pretrained(
            self.model_root,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
        ).eval()

        # Patch: Remove any 'output_hidden_states' argument to avoid duplicate injections
        orig_forward = base_model.forward
        def patched_forward(*args, **kwargs):
            kwargs.pop('output_hidden_states', None)
            return orig_forward(*args, **kwargs)
        base_model.forward = patched_forward

        # Load the adapter model locally and move to device
        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_root,
            local_files_only=True
        ).to(base_model.device)

        # Load the processor from local files
        self.processor = AutoProcessor.from_pretrained(
            self.adapter_root,
            local_files_only=True
        )

    def prepare(self):
        # Load processed content, including images
        content_list = self.cfg_processer.load_processed_content(disable_load_image=False)
        images = [content.image for content in content_list]

        # Collate function to process and move batches to device
        def collate_fn(batch):
            processed = process_images(self.processor, batch)
            return {k: v.to(self.device) for k, v in processed.items()}

        # Create DataLoader for image embedding
        dataloader = DataLoader(
            images,
            batch_size=self.cfg.image_retrieval.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        image_embeds = []
        for batch_image in tqdm(dataloader, desc="Encoding images"):
            with torch.no_grad():
                output = self.model(**batch_image)
                image_embeds.extend(output)

        # Stack embeddings into a single tensor
        try:
            self.image_embed = torch.stack(image_embeds, dim=0)
        except Exception as e:
            print(f"[prepare] stack error: {e}")
            self.image_embed = None

        return self.sample

    def find_top_k(self) -> Dict[str, Any]:
        # Check that embeddings have been prepared
        if self.image_embed is None:
            print("[find_top_k] Missing image embedding. Did you run prepare()?")
            return self.sample

        # Retrieve top_k and key names from config
        top_k = self.cfg.retrieval.get("top_k", 10)
        page_id_key = self.cfg.image_retrieval.get("page_id_key", "page_id")
        image_question_key = self.cfg.image_retrieval.get("image_question_key", "question")
        r_image_key = self.cfg.image_retrieval.get("r_image_key", "image-top")

        sample = self.sample
        # If already retrieved, return sample
        if r_image_key in sample:
            return sample

        # Prepare query with an image placeholder prefix
        query_text = sample[image_question_key]
        query = ["<image> " + query_text]
        dummy_image = Image.new("RGB", (448, 448), (255, 255, 255))
        batch_queries = process_queries(self.processor, query, dummy_image).to(self.device)

        # Encode query
        with torch.no_grad():
            query_embed = self.model(**batch_queries)

        # Compute similarity scores
        evaluator = CustomEvaluator(is_multi_vector=True)
        scores = evaluator.evaluate(query_embed, self.image_embed)

        scores_tensor = torch.tensor(scores)
        # If page IDs are provided, mask out others
        if page_id_key in sample:
            mask = torch.zeros_like(scores_tensor, dtype=torch.bool)
            for idx in sample[page_id_key]:
                mask[0, idx] = True
            scores_tensor = torch.where(mask, scores_tensor, torch.full_like(scores_tensor, float('-inf')))

        # Select top-k indices and scores
        values, indices = torch.topk(scores_tensor, min(top_k, scores_tensor.shape[-1]), dim=-1)

        sample[r_image_key] = indices.tolist()[0]
        sample[r_image_key + "_score"] = values.tolist()[0]
        self.cfg_processer.sample = sample
        return sample
