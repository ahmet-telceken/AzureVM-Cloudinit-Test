import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/mnt/datadisk/hf_models")
model = AutoModel.from_pretrained(MODEL_NAME, cache_dir="/mnt/datadisk/hf_models")

st.title("🔎 Metin Embedding Uygulaması")

text = st.text_area("Bir metin girin:", "")

if st.button("Embedding Oluştur"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        st.write("✅ Embedding vektörü:")
        st.json(embedding.tolist())
    else:
        st.warning("Lütfen bir metin giriniz.")