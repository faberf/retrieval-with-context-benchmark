import ujson as json
from clip import ClipVitLargePatch14
from data import load
import numpy as np
import torch
from torchmetrics.functional.retrieval import retrieval_reciprocal_rank
from torchmetrics.classification import Precision, Recall

model = ClipVitLargePatch14()
model.load_model()
load("data.json")

from data import queries, image_embeddings, captions, asr_transcripts, timestamps, caption_embeddings, asr_embeddings, query_embeddings, query_embeddings_clip


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


def score_segments(descriptor, query_embeddings):
    scores = np.empty([len(query_embeddings),len(descriptor)])

    for i in range(len(query_embeddings)):
        query_embedding = query_embeddings[i]

        query_scores = np.empty(len(descriptor))
        for j in range(len(descriptor)):
            score = get_score(query_embedding, descriptor[j])
            
            query_scores[j] = score
        
        scores[i] = query_scores

    return scores

def save_ranks(scores, filename):
    # sort indexes by score
    sorted_indexes = np.argsort(scores, axis=1)
    # reverse the indexes so that the highest score is first
    sorted_indexes = np.flip(sorted_indexes, axis=1)
    # save the indexes to a file
    with open(filename, "w") as f:
        json.dump(sorted_indexes.tolist(), f)
    

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


def precision_recall_per_query(scores, ground_truth, threshold=0.25):
    for i in range(len(queries)):
        # Convert to binary predictions based on the threshold
        predicted = (torch.from_numpy(scores[i]) > threshold).float()

        # Convert ground truth to binary tensor
        target = torch.tensor(ground_truth[i], dtype=torch.float32)

        # Initialize Precision and Recall metrics
        precision_metric = Precision(task="binary")
        recall_metric = Recall(task="binary")

        # Compute Precision and Recall
        precision = precision_metric(predicted, target)
        recall = recall_metric(predicted, target)

        print(f"Query {i}: precision: {precision.item()}, recall: {recall.item()}")


if __name__ == "__main__":
    ground_truth = np.zeros([len(queries),len(image_embeddings)])
    for i in range(len(queries)):
        matching_documents = queries[i].correct_segments
        for match in matching_documents:
            ground_truth[i][match] = 1

    # CLIP image embeddings
    scores_image = score_segments(image_embeddings, query_embeddings_clip)
    save_ranks(scores_image, "image_ranks.json")
    # mrr_image = mean_reciprocal_rank(scores_image, ground_truth)
    # print(f"Mean Reciprocal Rank for image embeddings: {mrr_image}")

    # precision_recall_per_query(scores_image, ground_truth)

    scores_asr = score_segments(asr_embeddings, query_embeddings)
    save_ranks(scores_asr, "asr_ranks.json")
    # mrr_asr = mean_reciprocal_rank(scores_asr, ground_truth)
    # print(f"Mean Reciprocal Rank for ASR embeddings: {mrr_asr}")

    # Image captions
    scores_caption = score_segments(caption_embeddings, query_embeddings)
    save_ranks(scores_caption, "caption_ranks.json")
    # mrr = mean_reciprocal_rank(scores_caption, ground_truth)
    # print(f"Mean Reciprocal Rank for image captions: {mrr}")

    # precision_recall_per_query(scores_caption, ground_truth)
    
    # COSMIR
    
    # Load COSMIR embeddings
    with open("cosmir_descriptions-ws_4-t_basic-embeddings.json", "r") as f:
        cosmir_embeddings = json.load(f)
    scores_cosmir = score_segments(cosmir_embeddings, query_embeddings)
    save_ranks(scores_cosmir, "cosmir_ranks.json")
    # mrr = mean_reciprocal_rank(scores_cosmir, ground_truth)
    # print(f"Mean Reciprocal Rank for COSMIR embeddings: {mrr}")
    
    # precision_recall_per_query(scores_cosmir, ground_truth)
