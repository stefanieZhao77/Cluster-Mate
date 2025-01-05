from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

def evaluate_clustering(data, labels, model_type):
    """
    Calculate multiple clustering evaluation metrics for the given clustering results.
    
    Args:
        data: The preprocessed data matrix used for clustering
        labels: Cluster labels assigned by the clustering algorithm
        model_type: Type of clustering algorithm used (for logging purposes)
        
    Returns:
        dict: Dictionary containing the following evaluation metrics:
            - silhouette_score: Measure of how similar an object is to its own cluster
                compared to other clusters (-1 to 1, higher is better)
            - calinski_harabasz_score: Ratio of between-cluster variance to within-cluster
                variance (higher is better)
            - davies_bouldin_score: Average similarity measure of each cluster with its
                most similar cluster (lower is better)
                
    Note:
        Returns NaN values for all metrics if less than 2 clusters are found (excluding noise points)
    """
    # Check number of unique labels (excluding noise points labeled as -1)
    unique_labels = len(set(labels) - {-1})

    if unique_labels < 2:
        print(
            f"Warning: {model_type} produced only {unique_labels} cluster(s). Metrics cannot be calculated."
        )
        return {
            "silhouette_score": float("nan"),
            "calinski_harabasz_score": float("nan"),
            "davies_bouldin_score": float("nan"),
        }

    metrics = {
        "silhouette_score": silhouette_score(data, labels),
        "calinski_harabasz_score": calinski_harabasz_score(data, labels),
        "davies_bouldin_score": davies_bouldin_score(data, labels),
    }
    return metrics 