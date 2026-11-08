from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset
from tqdm import tqdm
import importlib
import json
import torch
from typing import List
import os


class MultiAgentSystem:
    def __init__(self, config, dataset: BaseDataset): 
        self.config = config
        self.dataset = dataset
        self.agents: List[Agent] = []
        self.models: dict = {}

        for agent_config in self.config.agents:
            if agent_config.model.class_name not in self.models:
                module = importlib.import_module(agent_config.model.module_name)
                model_class = getattr(module, agent_config.model.class_name)
                print("Create model: ", agent_config.model.class_name)
                self.models[agent_config.model.class_name] = model_class(agent_config.model)
            self.add_agent(agent_config, self.models[agent_config.model.class_name])

        if config.sum_agent.model.class_name not in self.models:
            module = importlib.import_module(config.sum_agent.model.module_name)
            model_class = getattr(module, config.sum_agent.model.class_name)
            self.models[config.sum_agent.model.class_name] = model_class(config.sum_agent.model)
        self.sum_agent = Agent(config.sum_agent, self.models[config.sum_agent.model.class_name])

        print("MultiAgentSystem successfully initialized with dataset:", type(self.dataset))

    def add_agent(self, agent_config, model):
        module = importlib.import_module(agent_config.agent.module_name)
        agent_class = getattr(module, agent_config.agent.class_name)
        agent: Agent = agent_class(agent_config, model)
        self.agents.append(agent)

    def predict(self, question, texts, images):
        '''This should be implemented in the subclass (e.g., MDocAgent).'''
        raise NotImplementedError("Subclasses should implement `predict` method.")

    def sum(self, sum_question):
        ans, all_messages = self.sum_agent.predict(sum_question)
        def extract_final_answer(agent_response):
            try:
                response_dict = json.loads(agent_response)
                return response_dict.get("Answer", agent_response)
            except:
                return agent_response
        final_ans = extract_final_answer(ans)
        return final_ans, all_messages


    def predict_single_sample(self, sample: dict):
        print("[DEBUG] Sample type:", type(sample))
        print("[DEBUG] Sample content:", sample)

        question, texts, images = self.dataset.load_sample_retrieval_data(sample)
        print("[DEBUG] Texts passed to agents:", texts)
        print("[DEBUG] Images passed to agents:", images)
        try:
            final_ans, final_messages = self.predict(question, texts, images)
        except RuntimeError as e:
            print(e)
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            final_ans, final_messages = None, None

        sample[self.config.ans_key] = final_ans
        if self.config.save_message:
            sample[self.config.ans_key + "_message"] = final_messages

        torch.cuda.empty_cache()
        self.clean_messages()
        return sample

    def clean_messages(self):
        for agent in self.agents:
            agent.clean_messages()
        self.sum_agent.clean_messages()
