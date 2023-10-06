# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from audiocraft.models import MusicGen

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    MusicGen.get_pretrained('facebook/musicgen-melody')

if __name__ == "__main__":
    download_model()