import ujson as json
from clip import ClipVitLargePatch14
from data import load
import numpy as np

model = ClipVitLargePatch14()
model.load_model()
load("data.json")

from data import queries, image_embeddings, captions, asr_transcripts, timestamps


def get_score(first_features, second_features):
    first_features = np.array(first_features)
    second_features = np.array(second_features)
    # if the features are 1d arrays, convert them to 2d arrays
    if len(first_features.shape) == 1:
        first_features = first_features[np.newaxis, :]
    if len(second_features.shape) == 1:
        second_features = second_features[np.newaxis, :]
    # normalize features
    first_features /= np.linalg.norm(first_features, axis=1, keepdims=True)
    second_features /= np.linalg.norm(second_features, axis=1, keepdims=True)
    return (first_features @ second_features.T).squeeze()

if __name__ == "__main__":
    first_query = queries[0]
    first_query_text = first_query.text
    first_query_embedding = model.text_embedding(first_query_text)[0]
    
    score = get_score(first_query_embedding, image_embeddings[0])
    
    print("done")