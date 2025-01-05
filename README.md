# ClusterMate


ClusterMate is an open-source data clustering and visualization tool designed for easy analysis of multi-dimensional datasets.

## Features

- Import and merge multiple datasets
- Apply various clustering algorithms (K-means, DBSCAN, Hierarchical)
- Visualize clustering results with interactive plots
- Perform Principal Component Analysis (PCA)
- Evaluate clustering performance with multiple metrics

## Usage

### Operation Manual

#### 1. Data Import and Linking
1. Launch the application to see two default dataset frames
2. For each dataset:
   - Click "Browse" to select your CSV or Excel file
   - Select the link column (common identifier between datasets)
   - Select the date column for temporal alignment
   - Choose features you want to include in the analysis
   - For additional datasets (after baseline), set the time bias in days
3. Click "+ Add Dataset" if you need to add more datasets
4. The first dataset is considered the baseline - its timestamps will be used as reference points

#### 2. Data Preview and Feature Engineering
1. Click "Preview Combined Data" to see the merged datasets
2. In the preview window, you can:
   - View the combined data with all selected features
   - Create new features using the operation panel:
     * Select an operation (mean, sum, min, max)
     * Choose the features to combine
     * Enter a name for the new feature
     * Click "Calculate" to create the new feature
   - Navigate through the data using pagination controls

#### 3. Clustering Configuration
1. In the side menu, configure your clustering parameters:
   - Select the clustering algorithm (K-means, DBSCAN, or Hierarchical)
   - Set algorithm-specific parameters:
     * K-means: Number of clusters
     * DBSCAN: Epsilon and minimum samples
     * Hierarchical: Number of clusters
   - Set the number of PCA components
   - Select which features to use for clustering

#### 4. Model Training and Results
1. Click "Train Model" to start the clustering process
2. The training window will show:
   - Training progress
   - Visualization plots:
     * PCA components visualization
     * Feature correlation matrix
     * Feature importance in PCA
     * Cluster distribution plots
   - Evaluation metrics
3. After training:
   - View the generated plots and metrics
   - Click "Download Metrics" to save all results
   - Results include:
     * Clustered data with labels
     * Evaluation metrics
     * Visualization plots
     * Cluster statistics
     * Correlation matrices

#### Tips and Best Practices
- Ensure your datasets have consistent identifiers in the link columns
- Start with the most complete dataset as your baseline
- Use appropriate time bias based on your data's temporal characteristics
- Create combined features before training if needed
- Experiment with different clustering algorithms and parameters
- Review the evaluation metrics to assess clustering quality

## Download

You can download the latest release from the [Releases](https://github.com/stefanieZhao77/Cluster-Mate/releases) page.

## Running from Source

To run ClusterMate from source:

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run:
```python
python -m GUI.main
```

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CustomTkinter for the modern UI components
- Scikit-learn for machine learning algorithms
