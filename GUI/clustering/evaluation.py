from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

def evaluate_clustering(data, labels, model_type):
    """Calculate multiple clustering evaluation metrics"""
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