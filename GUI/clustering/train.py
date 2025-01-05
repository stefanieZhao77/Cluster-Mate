import os
import shutil
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import joblib
import matplotlib.pyplot as plt

from .preprocessing import preprocess_data
from .evaluation import evaluate_clustering
from .visualization import plot_pca_components, plot_feature_importance, plot_feature_correlation, plot_pca_explained_variance

def train_clustering_model(data, model_type, params, feature_names, original_data, pca):
    """
    Train a clustering model and evaluate its performance.
    
    Args:
        data: Preprocessed data matrix for clustering
        model_type: Type of clustering algorithm ('kmeans', 'dbscan', or 'hierarchical')
        params: Dictionary of parameters for the chosen algorithm
        feature_names: List of feature names for visualization
        original_data: Original dataset with all features
        pca: Fitted PCA model for feature importance visualization
        
    Returns:
        tuple: (trained_model, evaluation_metrics)
    """
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

    # Create visualization directory if it doesn't exist
    os.makedirs("metrics/cluster_plots", exist_ok=True)

    # Generate and save all visualizations
    
    # 1. PCA components visualization
    plot_pca_components(data, model.labels_, model_type)
    
    # 2. Feature correlation matrix for original features
    plot_feature_correlation(original_data, original_data.columns)
    
    # 3. Feature importance in PCA
    plot_feature_importance(pca, feature_names)

    # Save clustered data with labels and all original features
    clustered_data = original_data.copy()
    
    # Add cluster labels
    clustered_data["cluster"] = model.labels_
    
    # Add PCA components
    n_components = data.shape[1]
    for i in range(n_components):
        clustered_data[f"PC{i+1}"] = data[:, i]
    
    # Reorder columns to have cluster information at the start
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    ordered_cols = ['cluster'] + pc_cols + [col for col in clustered_data.columns 
                                          if col not in ['cluster'] + pc_cols]
    clustered_data = clustered_data[ordered_cols]
    
    # Save the complete clustered data
    clustered_data.to_csv(
        f"metrics/{model_type}_clustered_data_pc{n_components}.csv", index=False
    )
    
    # Save additional analysis files
    
    # 1. Cluster statistics
    # Filter numeric columns only for statistics
    numeric_cols = clustered_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        cluster_stats = clustered_data[numeric_cols].groupby(clustered_data['cluster']).agg(['mean', 'std', 'min', 'max'])
        cluster_stats.to_csv(f"metrics/{model_type}_cluster_statistics.csv")
    
    # 2. Correlation matrix (only for numeric columns)
    if len(numeric_cols) > 0:
        correlation_matrix = clustered_data[numeric_cols].corr()
        correlation_matrix.to_csv(f"metrics/{model_type}_correlation_matrix.csv")

    return model, metrics

def main(n_components=3, model_type="kmeans", model_params={"n_clusters": 5}, random_state=42, selected_features=None, all_features=None):
    """
    Main function to run the clustering pipeline.
    
    Args:
        n_components: Number of PCA components to use
        model_type: Type of clustering algorithm to use
        model_params: Parameters for the chosen clustering algorithm
        random_state: Random seed for reproducibility
        selected_features: List of features to use for clustering
        all_features: List of all selected features to include in output
        
    The function performs the following steps:
    1. Sets up directory structure for outputs
    2. Loads and preprocesses the data
    3. Trains the clustering model
    4. Saves the model and results
    5. Generates visualization plots
    """
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

    # Preprocess data with current PCA configuration using selected_features for clustering
    X_pca, scaler, pca, features, valid_indices = preprocess_data(
        train_data, n_components=n_components, random_state=42, all_features=selected_features
    )

    # Save PCA model
    joblib.dump(pca, f"models/pca_model_{n_components}comp.pkl")

    # Create original_data with all features (not just clustering features)
    if all_features is None:
        all_features = selected_features
    original_data = train_data.loc[valid_indices, all_features]
    
    # Train and evaluate model
    print(f"Training {model_type} with parameters: {model_params}")
    model, metrics = train_clustering_model(
        X_pca,
        model_type,
        model_params,
        features,  # Pass the original feature names instead of PC labels
        original_data,
        pca       # Pass the PCA model
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

    print("\nTraining completed. Results saved in metrics/all_models_comparison.csv")
    print("Performance comparison plots saved in metrics/cluster_plots/") 