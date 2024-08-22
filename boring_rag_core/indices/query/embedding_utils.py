"""
For generally embedding comparison, checkout boring_rag_core/embeddings/base.py
"""

import numpy as np
from typing import List, Tuple, Any, Optional
from enum import Enum
from sklearn import svm, linear_model
from boring_rag_core.vector_stores.types import VectorStoreQueryMode


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_top_k_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    similarity_top_k: int,
    embedding_ids: Optional[List] = None,
) -> Tuple[List[float], List]:
    query_embedding = np.array(query_embedding)
    embeddings = np.array(embeddings)
    
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    
    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))
    
    sorted_indices = np.argsort(similarities)[::-1][:similarity_top_k]
    
    top_similarities = [similarities[i] for i in sorted_indices]
    top_ids = [embedding_ids[i] for i in sorted_indices]
    
    return top_similarities, top_ids


def get_top_k_embeddings_learner(
    query_embedding: List[float],
    embeddings: List[List[float]],
    similarity_top_k: int,
    embedding_ids: Optional[List] = None,
    query_mode: VectorStoreQueryMode = VectorStoreQueryMode.SVM,
) -> Tuple[List[float], List]:
    query_embedding = np.array(query_embedding)
    embeddings = np.array(embeddings)
    
    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))
    
    X = np.vstack([query_embedding, embeddings])
    y = np.zeros(len(X))
    y[0] = 1  # The query embedding is the positive example
    
    if query_mode == VectorStoreQueryMode.SVM:
        clf = svm.LinearSVC(class_weight='balanced', max_iter=10000)
    elif query_mode == VectorStoreQueryMode.LINEAR_REGRESSION:
        clf = linear_model.LinearRegression()
    elif query_mode == VectorStoreQueryMode.LOGISTIC_REGRESSION:
        clf = linear_model.LogisticRegression(class_weight='balanced')
    else:
        raise ValueError(f"Unsupported learner mode: {query_mode}")
    
    clf.fit(X, y)
    
    # Get decision values for all embeddings
    decision_values = clf.decision_function(embeddings)
    
    sorted_indices = np.argsort(decision_values)[::-1][:similarity_top_k]
    
    top_similarities = decision_values[sorted_indices].tolist()
    top_ids = [embedding_ids[i] for i in sorted_indices]
    
    return top_similarities, top_ids


def get_top_k_mmr_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    similarity_top_k: int,
    embedding_ids: Optional[List] = None,
    mmr_threshold: float = 0.5,
) -> Tuple[List[float], List]:
    query_embedding = np.array(query_embedding)
    embeddings = np.array(embeddings)
    
    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))
    
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    
    selected_indices = []
    unselected_indices = list(range(len(embeddings)))
    
    for _ in range(similarity_top_k):
        if not unselected_indices:
            break
        
        mmr_scores = []
        for i in unselected_indices:
            if not selected_indices:
                mmr_scores.append(similarities[i])
            else:
                penalty = max(cosine_similarity(embeddings[i], embeddings[j]) for j in selected_indices)
                mmr_scores.append(mmr_threshold * similarities[i] - (1 - mmr_threshold) * penalty)
        
        best_index = unselected_indices[np.argmax(mmr_scores)]
        selected_indices.append(best_index)
        unselected_indices.remove(best_index)
    
    top_similarities = [similarities[i] for i in selected_indices]
    top_ids = [embedding_ids[i] for i in selected_indices]
    
    return top_similarities, top_ids
