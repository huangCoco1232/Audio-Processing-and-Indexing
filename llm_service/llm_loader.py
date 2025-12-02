# model_loader.py
"""
torch==2.5.1
transformers==4.49.0
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_PATH = "microsoft/Phi-4-mini-instruct"

def load_model():
    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    return pipe

# 全局只加载一次
pipe = load_model()