import json
import os
from typing import Dict, List

class Config:
    def __init__(self):
        # Load model paths configuration
        self.model_paths = {}
        try:
            with open('path.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            model_name = parts[0].strip()
                            model_path = ':'.join(parts[1:]).strip()
                            self.model_paths[model_name] = model_path
        except FileNotFoundError:
            raise FileNotFoundError("Model path configuration file not found: path.txt")
        
        # Ensure required model paths exist
        required_models = ["qwen2.5_7b", "stable-diffusion-3.5-large", "FLUX.1-dev"]
        missing_models = [model for model in required_models if model not in self.model_paths]
        if missing_models:
            raise ValueError(f"Missing path configuration for models: {', '.join(missing_models)}")
        
        # Load themes and scenes configuration
        try:
            with open('base_prompt.json', 'r') as f:
                self.themes = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Theme configuration file not found: base_prompt.json")
        except json.JSONDecodeError:
            raise ValueError("Theme configuration file is not in valid JSON format")
        
        # GPU configuration
        self.num_gpus = 8
        self.gpu_ids = list(range(self.num_gpus))
        
        # Output path configuration
        self.output_base_dir = "generated_images"
        
    def get_model_path(self, model_name: str) -> str:
        if model_name not in self.model_paths:
            raise KeyError(f"Path configuration not found for model {model_name}")
        return self.model_paths[model_name]
    
    def get_themes(self) -> Dict:
        return self.themes