# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import AutoProcessor, MusicgenForConditionalGeneration

def download_model():
    AutoProcessor.from_pretrained("facebook/musicgen-medium")
    MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium", torch_dtype=torch.float16)

if __name__ == "__main__":
    download_model()