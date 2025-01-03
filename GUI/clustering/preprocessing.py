import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .visualization import plot_feature_correlation, plot_pca_explained_variance

def preprocess_data(data, n_components=3, random_state=42, all_features=None):
    """Preprocess the data by selecting relevant features, scaling, and applying PCA"""
    # Drop rows with missing values and convert to numeric, replacing invalid values with NaN
    all_features_data = data[all_features].apply(pd.to_numeric, errors="coerce")

    # Drop any rows that have NaN values after conversion
    all_features_data = all_features_data.dropna()
    X_original = all_features_data[all_features]

    # Plot feature correlations with cleaned data
    plot_feature_correlation(all_features_data, all_features)

    # Scale the features
    scaler_original = StandardScaler()
    X_original_scaled = scaler_original.fit_transform(X_original)

    # Apply PCA
    pca_original = PCA(n_components=n_components, random_state=random_state)
    X_pca_original = pca_original.fit_transform(X_original_scaled)

    # Plot PCA explained variance
    plot_pca_explained_variance(pca_original, all_features)

    # Print explained variance ratio
    print("\nPCA Explained variance ratio:", pca_original.explained_variance_ratio_)
    print("Number of components:", pca_original.n_components_)
    print("Total explained variance:", sum(pca_original.explained_variance_ratio_))

    # Keep track of valid indices after dropping NA values
    valid_indices = all_features_data.index

    return X_pca_original, X_original_scaled, pca_original, all_features, valid_indices 