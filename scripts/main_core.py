import os
import json
from pathlib import Path
import pymupdf
from omegaconf import OmegaConf
from ragatouille import RAGPretrainedModel
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval import TextRetrieval, ColpaliRetrieval
from config_processer import ConfigProcesser
from extract import extract
import shutil
import hydra
from hydra.utils import get_original_cwd
from predict import predict


def run_core(cfg):
    pdf_path = cfg.input.pdf_path
    question = cfg.input.question

    cfg_processer = ConfigProcesser(cfg, str(pdf_path), question)
    
    # extract
    extract(cfg_processer)
    
    # retrieval
    text_retrieval = TextRetrieval(cfg_processer)
    text_retrieval.retrieval()

    image_retrieval = ColpaliRetrieval(cfg_processer)
    image_retrieval.retrieval()

    sample = cfg_processer.sample 

    # predict
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.mdoc_agent.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    
    for agent_config in cfg.mdoc_agent.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_cfg = hydra.compose(config_name="agent/" + agent_name, overrides=[]).agent
        model_cfg = hydra.compose(config_name="model/" + model_name, overrides=[]).model
        agent_config.agent = agent_cfg
        agent_config.model = model_cfg

    cfg.mdoc_agent.sum_agent.agent = hydra.compose(
        config_name="agent/" + cfg.mdoc_agent.sum_agent.agent, overrides=[]
    ).agent
    cfg.mdoc_agent.sum_agent.model = hydra.compose(
        config_name="model/" + cfg.mdoc_agent.sum_agent.model, overrides=[]
    ).model

    sample = predict(cfg, sample, cfg_processer.extract_path)
    print("sample is ", sample)
    answer = sample[cfg.mdoc_agent.ans_key]
    return answer
  
