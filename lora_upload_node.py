import io
import torch
from comfy.model_management import load_lora_model
from comfy.custom_nodes import NODE_REGISTRY, CustomNodeBase

class LoRAUploadNode(CustomNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_file_1": ("FILE", {"label": "LoRA File 1"}),
                "lora_1_switch": ("BOOL", {"default": True}),
                "model_weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "clip_weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                
                "lora_file_2": ("FILE", {"label": "LoRA File 2"}),
                "lora_2_switch": ("BOOL", {"default": False}),
                "model_weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "clip_weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                
                "lora_file_3": ("FILE", {"label": "LoRA File 3"}),
                "lora_3_switch": ("BOOL", {"default": False}),
                "model_weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "clip_weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "Custom Nodes"

    def load_loras(self, lora_file_1, lora_1_switch, model_weight_1, clip_weight_1,
                          lora_file_2, lora_2_switch, model_weight_2, clip_weight_2,
                          lora_file_3, lora_3_switch, model_weight_3, clip_weight_3):
        
        models = []
        
        # Load and apply LoRA models if enabled
        if lora_1_switch and lora_file_1 is not None:
            model_1 = self.load_lora_from_file(lora_file_1, model_weight_1, clip_weight_1)
            models.append(model_1)
        
        if lora_2_switch and lora_file_2 is not None:
            model_2 = self.load_lora_from_file(lora_file_2, model_weight_2, clip_weight_2)
            models.append(model_2)
        
        if lora_3_switch and lora_file_3 is not None:
            model_3 = self.load_lora_from_file(lora_file_3, model_weight_3, clip_weight_3)
            models.append(model_3)
        
        return models

    def load_lora_from_file(self, file_path, model_weight, clip_weight):
        """Loads a LoRA model from a file and applies the given weights."""
        # Load the LoRA model from the given file path
        lora_model = load_lora_model(file_path)
        
        # Apply model and clip weights
        lora_model.adjust_weights(model_weight, clip_weight)
        
        return lora_model

# Register the node with ComfyUI
NODE_REGISTRY.register_node("LoRAUploadNode", LoRAUploadNode)
