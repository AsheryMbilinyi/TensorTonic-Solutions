def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    hits = list(set(recommended[:k]) & set(relevant))
    precision = len(hits)/k
    recall = len(hits)/len(relevant)
    return [precision,recall]
    