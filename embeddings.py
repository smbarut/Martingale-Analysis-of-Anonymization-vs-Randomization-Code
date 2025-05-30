
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from typing import List, Tuple
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from transformers import BertTokenizer, BertModel
import torch


def get_senctence_embeddings(sentences: List[str], tokenizer, model) -> np.ndarray:
    
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)

def get_bert_embeddings(sentences: List[str], model) -> np.ndarray:
    
    embeddings = []
    for sentence in sentences:
        embedding = model.encode(sentence)
        embeddings.append(embedding)
    return np.array(embeddings)

def calculate_distance(original, anonymized, randomized) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    dist_original_anonymized = pairwise_distances(original, anonymized)
    dist_original_randomized = pairwise_distances(original, randomized)
    dist_anonymized_randomized = pairwise_distances(anonymized, randomized)
    
    return dist_original_anonymized, dist_original_randomized, dist_anonymized_randomized

def calculate_euclidean_distance_single_sentence(original, anonymized, randomized) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    dist_original_anonymized = euclidean_distances(original, anonymized)
    dist_original_randomized = euclidean_distances(original, randomized)
    dist_anonymized_randomized = euclidean_distances(anonymized, randomized)
    
    return dist_original_anonymized, dist_original_randomized, dist_anonymized_randomized

def calculate_cosine_distance_single_sentence(original, anonymized, randomized) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    dist_original_anonymized = cosine_similarity(original, anonymized)
    dist_original_randomized = cosine_similarity(original, randomized)
    dist_anonymized_randomized = cosine_similarity(anonymized, randomized)
    
    return dist_original_anonymized, dist_original_randomized, dist_anonymized_randomized
def plot_distance_matrix(distances: np.ndarray, title: str, output_path: str) -> None:
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, cmap='viridis', annot=True)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def main():
    sentences_with_names = [
        "John Doe is a software engineer.",
        "Jane Smith is a data scientist.",
        "Alice Johnson is a product manager."
    ]
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    anonymized_sentences = ["[Name] is a software engineer.",
                            "[Name] is a data scientist.",
                            "[Name] is a product manager."]
    randomized_sentences = ["Michael Brown is a software engineer.",
                            "Emily Davis is a data scientist.",
                            "David Wilson is a product manager."]
    original_embeddings = get_senctence_embeddings(sentences_with_names, tokenizer, model)
    anonymized_embeddings = get_senctence_embeddings(anonymized_sentences, tokenizer, model)
    randomized_embeddings = get_senctence_embeddings(randomized_sentences, tokenizer, model)
    for i in range(len(original_embeddings)):
        distances = calculate_cosine_distance_single_sentence(
            original_embeddings[i].reshape(1, -1),
            anonymized_embeddings[i].reshape(1, -1),
            randomized_embeddings[i].reshape(1, -1)
        )
        dist_original_anonymized = distances[0]
        dist_original_randomized = distances[1]
        dist_anonymized_randomized = distances[2]
        print(f"Distances for sentence {i}:")
        print("Original vs Anonymized:", dist_original_anonymized)
        print("Original vs Randomized:", dist_original_randomized)
        print("Anonymized vs Randomized:", dist_anonymized_randomized)

    # print("Distance matrices calculated.")
    # print("Original vs Anonymized Distance Matrix:\n", euclidean_distance[0])
    # print("Original vs Randomized Distance Matrix:\n", euclidean_distance[1])
    # print("Anonymized vs Randomized Distance Matrix:\n", euclidean_distance[2])
    # plot_distance_matrix(dist_original_anonymized, "Original vs Anonymized", "original_anonymized.png")
    # plot_distance_matrix(dist_original_randomized, "Original vs Randomized", "original_randomized.png")
    # plot_distance_matrix(dist_anonymized_randomized, "Anonymized vs Randomized", "anonymized_randomized.png")
    # print("Distance matrices plotted and saved.")

if __name__ == "__main__":
    main()
