# embedding loader script
# Week 4 embedding loader

import numpy as np

def load_embeddings(file_path):
    embeddings = np.load(file_path)
    return embeddings

def combine_features(bio_features, embeddings):
    combined = np.concatenate([bio_features, embeddings], axis=1)
    return combined
