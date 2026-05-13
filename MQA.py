import torch
import time
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
num_heads = model.config.num_attention_heads
hidden_size = model.config.hidden_size
head_dim = hidden_size // num_heads
def load_wikitext():
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
    return [t for t in dataset["text"] if len(t.strip()) > 200][:5] 
import types
def apply_mqa_patch(model):
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        attn = layer.attention
        original_forward = attn.forward
        def new_forward(self, *args, _original_forward=original_forward, **kwargs):
            outputs = _original_forward(*args, **kwargs)
            if len(outputs) < 2:
                return outputs
            attn_output = outputs[0]
            present = outputs[1]
            if present is not None:
                key, value = present
                key_shared = key.mean(dim=1, keepdim=True)
                value_shared = value.mean(dim=1, keepdim=True)
                key = key_shared.expand_as(key)
                value = value_shared.expand_as(value)
                present = (key, value)
            new_outputs = (attn_output, present)
            if len(outputs) > 2:
                new_outputs += outputs[2:]
            return new_outputs
        attn.forward = types.MethodType(new_forward, attn)       
apply_mqa_patch(model)
@torch.no_grad()
def compute_ppl(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    loss = model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(loss).item()
@torch.no_grad()
def test_speed(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    start = time.time()
    model.generate(**inputs, max_new_tokens=256, do_sample=False)
    end = time.time()
    mem = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0
    return end - start, mem
if __name__ == "__main__":
    wikitext_samples = load_wikitext()
    print("WikiText PPL 测试")
    ppls = [compute_ppl(model, tokenizer, t) for t in wikitext_samples]
    print(f"平均 PPL: {sum(ppls)/len(ppls):.2f}")
    print("推理速度")
    test_text = wikitext_samples[0][:2000]
    t, mem = test_speed(model, tokenizer, test_text)
    print(f"生成时间: {t:.2f} s")
    
