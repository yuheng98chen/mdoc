from tqdm import tqdm
import importlib
import json
import torch
import os
from agents.multi_agent_system import MultiAgentSystem
from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset

class MDocAgent(MultiAgentSystem):
    def __init__(self, config, dataset: BaseDataset):  
        super().__init__(config, dataset)

    def predict(self, question, texts, images):


        # Call general agent to generate an initial response
        general_agent = self.agents[-1]
        general_response, messages = general_agent.predict(
            question, texts, images, with_sys_prompt=True
        )
  

        # Let general agent perform self-reflection using a critical prompt
        critical_info = general_agent.self_reflect(
            prompt=general_agent.config.agent.critical_prompt,
            add_to_message=False
        )
        print(f"[DEBUG] Self-reflect raw output: {critical_info}")

        # Validate and extract JSON content from reflection
        if not critical_info or '{' not in critical_info or '}' not in critical_info:
            print("[WARN] self_reflect returned empty or invalid content.")
            critical_info = '{"text": "", "image": ""}'
        else:
            start_index = critical_info.find('{')
            end_index = critical_info.find('}') + 1
            critical_info = critical_info[start_index:end_index]

        text_reflection = ""
        image_reflection = ""

        # Try parsing JSON and extracting clues
        try:
            critical_info = json.loads(critical_info)
            text_reflection = critical_info.get("text", "")
            image_reflection = critical_info.get("image", "")

        except Exception as e:
            print(f"[ERROR] Failed to parse critical_info JSON: {e}")
            text_reflection = ""
            image_reflection = ""

        # Call text agent with text clue
        text_agent = self.agents[1]
        reflect_prompt = "\nYou may use the given clue:\n"
        text_input = question + reflect_prompt + text_reflection
        text_response, _ = text_agent.predict(
            text_input,
            texts=texts,
            images=None,
            with_sys_prompt=True
        )
        if not text_response.strip():
            print("[WARN] Text agent produced empty output.")
            text_response = "No text insight generated."

        # Call image agent with image clue
        image_agent = self.agents[0]
        image_input = question + reflect_prompt + image_reflection
        image_response, _ = image_agent.predict(
            image_input,
            texts=None,
            images=images,
            with_sys_prompt=True
        )
        if not image_response.strip():
            print("[WARN] Image agent produced empty output.")
            image_response = "No image insight generated."

        # Combine all messages
        all_messages = ""
        all_messages += "General Agent:\n" + general_response + "\n"
        all_messages += "Text Agent:\n" + text_response + "\n"
        all_messages += "Image Agent:\n" + image_response + "\n"

        # Summarize final answer
        final_ans, final_messages = self.sum(all_messages)


        return final_ans, final_messages
