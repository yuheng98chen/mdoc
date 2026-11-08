from abc import ABC, abstractmethod
from omegaconf import OmegaConf
import os

class BaseRetrieval(ABC):
    def __init__(self,cfg_processer):
        self.cfg_processer = cfg_processer

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def find_top_k(self):
        pass
    
    def retrieval(self):
        self.prepare()
        self.find_top_k()
        return self.sample