import ujson as json
from clip import ClipVitLargePatch14
from data import load
import numpy as np
import torch
from torchmetrics.retrieval import RetrievalMRR
from torchmetrics.functional.retrieval import retrieval_reciprocal_rank

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


def score_segments(descriptor, embedded=False):
    scores = np.empty([len(queries),len(descriptor)])

    for i in range(len(queries)):
        query_embedding = model.text_embedding(queries[i].text)

        query_scores = np.empty(len(descriptor))
        for j in range(len(descriptor)):
            if embedded:
                score = get_score(query_embedding, descriptor[j])
            else:
                score = get_score(query_embedding, model.text_embedding(descriptor[j]))
            
            query_scores[j] = score
        
        scores[i] = query_scores

    return scores


def mean_reciprocal_rank(scores, ground_truth):
    # compute reciprocal rank individually for each query, then apply the mean
    all_rr = []
    for i in range(len(queries)):
        score_tensor = torch.from_numpy(scores[i])
        ground_truth_tensor = torch.from_numpy(ground_truth[i])
        rr = retrieval_reciprocal_rank(score_tensor, ground_truth_tensor)
        all_rr.append(rr)
    mrr = torch.mean(torch.stack(all_rr))
    return mrr


if __name__ == "__main__":
    ground_truth = np.zeros([len(queries),len(image_embeddings)])
    for i in range(len(queries)):
        matching_documents = queries[i].correct_segments
        for match in matching_documents:
            ground_truth[i][match] = 1

    # CLIP image embeddings
    scores_image = score_segments(image_embeddings, True)
    mrr = mean_reciprocal_rank(scores_image, ground_truth)
    print(f"Mean Reciprocal Rank for image embeddings: {mrr}")

    # ASR transcript embedding
    #scores_asr = score_segments(asr_transcripts, False)

    # Image captions
    scores_caption = score_segments(captions, False)
    mrr = mean_reciprocal_rank(scores_caption, ground_truth)
    print(f"Mean Reciprocal Rank for image captions: {mrr}")
