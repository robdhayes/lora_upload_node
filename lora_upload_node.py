import io

class LoRAUploadNode:
    def __init__(self):
        # Store LoRA data and settings for up to three LoRA models
        self.lora_data = [None, None, None]
        self.lora_switches = [False, False, False]
        self.lora_names = [None, None, None]
        self.model_weights = [1.0, 1.0, 1.0]
        self.clip_weights = [1.0, 1.0, 1.0]

    def upload_lora(self, index, file_content, file_name):
        """Uploads a LoRA file to memory."""
        self.lora_data[index] = io.BytesIO(file_content)
        self.lora_names[index] = file_name
        print(f"Loaded LoRA model {index + 1} ({file_name}) into system RAM.")

    def set_lora_switch(self, index, state):
        """Enables or disables a LoRA."""
        self.lora_switches[index] = state
        status = "On" if state else "Off"
        print(f"LoRA model {index + 1} is {status}.")

    def set_model_weight(self, index, weight):
        """Sets the model weight for a LoRA."""
        self.model_weights[index] = weight
        print(f"Model weight for LoRA model {index + 1} set to {weight}.")

    def set_clip_weight(self, index, weight):
        """Sets the clip weight for a LoRA."""
        self.clip_weights[index] = weight
        print(f"Clip weight for LoRA model {index + 1} set to {weight}.")

    def process_loras(self):
        """Processes the active LoRAs based on their settings."""
        for i in range(3):
            if self.lora_switches[i] and self.lora_data[i]:
                # Simulate processing the LoRA with its weights
                print(f"Processing LoRA model {i + 1}: {self.lora_names[i]}")
                print(f"  - Model Weight: {self.model_weights[i]}")
                print(f"  - Clip Weight: {self.clip_weights[i]}")
            elif not self.lora_switches[i]:
                print(f"LoRA model {i + 1} is disabled.")
            else:
                print(f"No LoRA data for model {i + 1}.")

# Example usage of the node:
node = LoRAUploadNode()

# Simulate file uploads (this would be handled by the ComfyUI web interface)
file_content_1 = b"binary content of LoRA file 1"
file_name_1 = "LoRA_1.safetensors"
node.upload_lora(0, file_content_1, file_name_1)

file_content_2 = b"binary content of LoRA file 2"
file_name_2 = "LoRA_2.safetensors"
node.upload_lora(1, file_content_2, file_name_2)

# Enable LoRAs and set weights
node.set_lora_switch(0, True)
node.set_model_weight(0, 0.8)
node.set_clip_weight(0, 1.0)

node.set_lora_switch(1, True)
node.set_model_weight(1, 0.9)
node.set_clip_weight(1, 0.7)

# Process active LoRAs
node.process_loras()
