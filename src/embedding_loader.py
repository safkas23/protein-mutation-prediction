import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
 
# esm model
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
 
def load_esm_model(model_name=ESM_MODEL_NAME):
    """
    Load ESM-2 tokenizer and model onto the best available device.
    Returns (tokenizer, model, device).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device
 
def get_embedding(seq, tokenizer, model, device):
    """
    Compute a sequence-level ESM-2 embedding for a single protein sequence
    via mean pooling over token hidden states.
    """
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
 
def get_embeddings_batch(sequences, tokenizer, model, device):
    """
    Compute ESM-2 embeddings for a list of sequences.
    Returns an ndarray of shape (n_sequences, embedding_dim).
    """
    return np.array([get_embedding(seq, tokenizer, model, device) for seq in sequences])
 
def combine_features(phys_features, embeddings):
    """
    Horizontally joins physicochemical features and ESM embeddings
    into a single hybrid feature matrix.
    """
    return np.hstack([phys_features, embeddings])
