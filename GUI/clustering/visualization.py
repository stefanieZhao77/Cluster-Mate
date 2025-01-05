import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_pca_components(X_pca, labels, model_type):
    """
    Create a grid of scatter plots showing pairwise relationships between PCA components.
    
    Args:
        X_pca: PCA-transformed data matrix
        labels: Cluster labels for coloring the points
        model_type: Type of clustering algorithm used (for plot title)
        
    The function creates a figure with n_components x n_components subplots where:
    - Diagonal plots show histograms of individual components
    - Off-diagonal plots show scatter plots of component pairs
    - Points are colored according to their cluster labels
    
    The plot is saved to metrics/cluster_plots/ directory.
    """
    n_components = X_pca.shape[1]
    fig, axes = plt.subplots(n_components, n_components, figsize=(15, 15))

    scatter = None  # Initialize scatter variable

    # Create pairwise scatter plots for all components
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                scatter = axes[i, j].scatter(
                    X_pca[:, j], X_pca[:, i], c=labels, cmap="viridis", s=50
                )
                axes[i, j].set_xlabel(f"Principal Component {j+1}")
                axes[i, j].set_ylabel(f"Principal Component {i+1}")
            else:
                # On diagonal, show component distribution
                axes[i, i].hist(X_pca[:, i], bins=20)
                axes[i, i].set_xlabel(f"Principal Component {i+1}")

    plt.suptitle(
        f"Principal Components Analysis\nClustering with {model_type.capitalize()}"
    )

    # Add colorbar only if scatter exists
    if scatter is not None:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatter, cax=cbar_ax)

    plt.savefig(
        f"metrics/cluster_plots/{model_type}_pc{n_components}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

def plot_pca_explained_variance(pca, features):
    """
    Create plots showing the explained variance of PCA components.
    
    Args:
        pca: Fitted PCA model
        features: List of original feature names
        
    Creates two subplots:
    1. Bar plot of explained variance ratio for each component
    2. Line plot of cumulative explained variance
    
    The plot is saved to metrics/cluster_plots/ directory.
    """
    plt.figure(figsize=(12, 5))

    # Plot 1: Explained variance ratio
    plt.subplot(1, 2, 1)
    plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance Ratio by Principal Component")

    # Plot 2: Cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        "bo-",
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance vs. Number of Components")

    plt.tight_layout()
    plt.savefig(
        f"metrics/cluster_plots/pca_explained_variance_pc{pca.n_components_}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

def plot_feature_correlation(data, features):
    """
    Create a heatmap showing correlations between original features.
    
    Args:
        data: DataFrame containing the original features
        features: List of feature names to include
        
    The correlation matrix is visualized using a heatmap with:
    - Color scale from -1 (negative correlation) to 1 (positive correlation)
    - Numerical correlation values annotated in each cell
    
    The plot is saved to metrics/cluster_plots/ directory.
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="coolwarm", 
        center=0, 
        fmt=".2f"
    )
    
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(
        f"metrics/cluster_plots/feature_correlation.png", bbox_inches="tight", dpi=300
    )
    plt.close()

def plot_feature_importance(pca, features):
    """
    Create a heatmap showing the importance of original features in PCA components.
    
    Args:
        pca: Fitted PCA model
        features: List of original feature names
        
    The function:
    1. Extracts absolute values of PCA loadings
    2. Creates a heatmap showing feature importance in each component
    3. Saves both the visualization and raw loadings data
    
    Returns:
        DataFrame: PCA loadings for each feature and component
    """
    # Get the absolute value of loadings
    loadings = np.abs(pca.components_)

    # Create a DataFrame of loadings
    loadings_df = pd.DataFrame(
        loadings.T,
        columns=[f"PC{i+1}" for i in range(loadings.shape[0])],
        index=features,
    )

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings_df, annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title("Feature Importance in Principal Components")
    plt.ylabel("Features")
    plt.xlabel("Principal Components")

    plt.tight_layout()
    plt.savefig(
        f"metrics/cluster_plots/feature_importance_heatmap_pc{pca.n_components_}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Save the loadings to CSV for further analysis
    loadings_df.to_csv("metrics/pca_feature_loadings.csv")

    return loadings_df 