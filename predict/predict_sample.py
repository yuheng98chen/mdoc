import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from mydatasets.base_dataset import BaseDataset
from agents.mdoc_agent import MDocAgent




def predict(cfg, sample, extract_path):
    dataset = BaseDataset(cfg.predict, sample, extract_path)
    mdoc_agent = MDocAgent(cfg.mdoc_agent, dataset)
    sample = mdoc_agent.predict_single_sample(sample)
    return sample
