import os
import shutil
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import joblib
import matplotlib.pyplot as plt

from .preprocessing import preprocess_data
from .evaluation import evaluate_clustering
from .visualization import plot_pca_components, plot_feature_importance

def train_clustering_model(data, model_type, params, feature_names, original_data):
    """Train clustering model and evaluate results"""
    if model_type == "kmeans":
        model = KMeans(**params, random_state=42)
    elif model_type == "dbscan":
        model = DBSCAN(**params)
    else:  # hierarchical
        model = AgglomerativeClustering(**params)

    # Fit the model
    model.fit(data)

    # Get metrics
    metrics = evaluate_clustering(data, model.labels_, model_type)

    # Create metrics dataframe
    metrics_df = pd.DataFrame(
        {
            "model_type": [model_type],
            **{k: [v] for k, v in metrics.items()},
            "params": [str(params)],
        }
    )

    # Save metrics
    metrics_df.to_csv(f"metrics/{model_type}_metrics.csv", index=False)

    # Plot PCA components
    plot_pca_components(data, model.labels_, model_type)

    # Save clustered data with labels and all original features
    clustered_data = original_data.copy()
    clustered_data["cluster_label"] = model.labels_
    n_components = data.shape[1]
    # Add PCA components as additional columns
    for i in range(data.shape[1]):
        clustered_data[f"component_{i+1}"] = data[:, i]
    clustered_data.to_csv(
        f"metrics/{model_type}_clustered_data_pc{n_components}.csv", index=False
    )

    return model, metrics

def main(n_components=3, model_type="kmeans", model_params={"n_clusters": 5}, random_state=42, selected_features=None):
    # Create directories
    os.makedirs("models", exist_ok=True)
    # Remove existing metrics directory if it exists
    if os.path.exists("metrics"):
        shutil.rmtree("metrics")
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("metrics/cluster_plots", exist_ok=True)

    # Load training data
    train_data = pd.read_csv("./cluster_data.csv")

    # Store all results
    all_results = []

    print(f"\nTraining models with {n_components} PCA components")

    # Preprocess data with current PCA configuration
    X_pca, scaler, pca, features, valid_indices = preprocess_data(
        train_data, n_components=n_components, random_state=42, all_features=selected_features
    )

    # Save PCA model
    joblib.dump(pca, f"models/pca_model_{n_components}comp.pkl")

    # Use valid_indices when creating original_data
    original_data = train_data.loc[valid_indices, features]
    
    # Train and evaluate model
    print(f"Training {model_type} with parameters: {model_params}")
    model, metrics = train_clustering_model(
        X_pca,
        model_type,
        model_params,
        [f"PC{i+1}" for i in range(n_components)],
        original_data,
    )

    # Save model
    joblib.dump(
        model, f"models/{model_type}_{n_components}comp_{str(model_params)}_model.pkl"
    )

    # Store results
    all_results.append(
        {
            "n_components": n_components,
            "model_type": model_type,
            "params": model_params,
            "explained_variance": sum(pca.explained_variance_ratio_),
            **metrics,
        }
    )

    # Save overall results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("metrics/all_models_comparison.csv", index=False)

    # Plot feature importance
    plot_feature_importance(pca, features)

    print("\nTraining completed. Results saved in metrics/all_models_comparison.csv")
    print("Performance comparison plots saved in metrics/cluster_plots/") 