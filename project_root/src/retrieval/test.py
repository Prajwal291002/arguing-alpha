import numpy as np
import json

embeddings = np.load("data/embeddings/embeddings.npy")
print("Shape of embeddings.npy", embeddings.shape)

with open("data/embeddings/metadata.json") as f:
    metadata = json.load(f)
print("Length of metadata: ",len(metadata))