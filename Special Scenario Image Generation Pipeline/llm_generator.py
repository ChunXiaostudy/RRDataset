import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class QwenPromptGenerator:
    def __init__(self, config, gpu_id: int = 0):
        self.config = config
        self.device = f"cuda:{gpu_id}"
        self.model_path = config.get_model_path("qwen2.5_7b")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
    def generate_rich_prompt(self, theme: str, scene: str) -> str:
        # Build system prompt
        system_prompt = """You are a professional prompt engineer for image generation. 
        Please expand the given theme and scene into a detailed, rich prompt that will generate 
        high-quality, photorealistic images. Include details about lighting, composition, 
        atmosphere, and technical aspects. The output should be in English."""
        
        # Build user input
        user_input = f"Theme: {theme}\nScene: {scene}\nPlease generate a detailed prompt."
        
        # Combine full input
        full_prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
        
        # Generate expanded prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        expanded_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_prompt(expanded_prompt)
    
    def _clean_prompt(self, prompt: str) -> str:
        # Clean generated text, keep only the actual prompt part
        try:
            return prompt.split("Assistant:")[-1].strip()
        except:
            return prompt.strip() 