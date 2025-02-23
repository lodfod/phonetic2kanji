from huggingface_hub import login
from datasets import load_dataset
import os
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
login(token=token)

    
def load_dataset():
    ds = load_dataset("reazon-research/reazonspeech", "tiny", trust_remote_code=True)
    return ds

