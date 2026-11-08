import os
import json
from tqdm import tqdm
# offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


from ragatouille import RAGPretrainedModel
from .base_retrieval import BaseRetrieval

class TextRetrieval(BaseRetrieval):
    def __init__(self, cfg_processer):
        super().__init__(cfg_processer)
        self.cfg_processer = cfg_processer
        self.cfg = cfg_processer.config
        self.sample = cfg_processer.sample
        self.pdf_path = cfg_processer.pdf_path
        self.question = cfg_processer.question
        self.doc_id = cfg_processer.doc_id
        self.doc_index = {}
      
        self.model_root = os.path.join(os.path.dirname(__file__), "local_colbert")
        self.index_root = os.path.join(self.model_root, "indexes")
        os.makedirs(self.index_root, exist_ok=True)

    def prepare(self):

        RAG = RAGPretrainedModel.from_pretrained(self.model_root)


        doc_id = self.doc_id
        index_key = self.cfg.text_retrieval.r_text_index_key

        try:
            content_list = self.cfg_processer.load_processed_content(disable_load_image=True)
            text_list = [
                content.txt.replace("\n", " ").replace("\r", "")[:self.cfg.text_retrieval.text_max_character_per_page]
                for content in content_list
            ]
        except Exception as e:
            print(f"[prepare] Failed loading processed content: {e}")
            self.sample[index_key] = ""
            return self.sample

      
        index_name = f"text-{self.cfg.text_retrieval.text_question_key}-{doc_id}"
        index_path = os.path.join(self.index_root, index_name)
        os.makedirs(index_path, exist_ok=True)

        try:
       
            created_path = RAG.index(
                index_name=index_path,
                collection=text_list,
                use_faiss=False
            )
            self.doc_index[doc_id] = created_path
            self.sample[index_key] = created_path
        except Exception as e:
            print(f"[prepare] Indexing failed: {e}")
            self.sample[index_key] = ""

        return self.sample

    def find_top_k(self):
        top_k = self.cfg.retrieval.top_k
        index_key = self.cfg.text_retrieval.r_text_index_key
        question_key = self.cfg.text_retrieval.text_question_key
        result_key = self.cfg.text_retrieval.r_text_key

        index_path = self.sample.get(index_key)
        map_file = os.path.join(index_path, "pid_docid_map.json")
        if not os.path.exists(map_file):
            print(f"[find_top_k] Missing index or pid_docid_map.json at: {index_path}")
            self.sample[result_key] = []
            self.sample[result_key + "_score"] = []
            return self.sample

        try:
            with open(map_file, "r", encoding="utf-8") as f:
                pid_map_data = json.load(f)

            unique_pages = list(dict.fromkeys(pid_map_data.values()))
            page_to_idx = {v: i for i, v in enumerate(unique_pages)}
            pid_map = {int(k): page_to_idx[v] for k, v in pid_map_data.items()}

            question = self.sample[question_key]
            RAG = RAGPretrainedModel.from_index(index_path)
            results = RAG.search(question, k=len(pid_map))

            top_indices = [pid_map[r['passage_id']] for r in results][:top_k]
            top_scores = [r['score'] for r in results][:top_k]

            self.sample[result_key] = top_indices
            self.sample[result_key + "_score"] = top_scores
            self.cfg_processer.sample = self.sample
        except Exception as e:
            print(f"[find_top_k] Retrieval failed: {e}")
            self.sample[result_key] = []
            self.sample[result_key + "_score"] = []

        return self.sample
