import customtkinter as ctk

class SideMenu(ctk.CTkFrame):
    """
    A side menu frame that contains clustering algorithm parameters and controls.
    Provides UI elements for selecting algorithms and configuring their parameters.
    """
    def __init__(self, master, **kwargs):
        """
        Initialize the side menu frame.
        
        Args:
            master: Parent widget
            **kwargs: Additional keyword arguments for CTkFrame
        """
        super().__init__(master, width=250, fg_color=("gray95", "gray13"), **kwargs)
        self.grid_rowconfigure(10, weight=1)  # Push everything up
        
        # Title with enhanced styling
        title = ctk.CTkLabel(
            self,
            text="Training Parameters",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("gray25", "gray90")
        )
        title.pack(pady=15, padx=10)
        
        # Create parameters
        self.create_parameter_widgets()
        
    def create_parameter_widgets(self):
        """
        Creates all parameter widgets including algorithm selection,
        algorithm-specific parameters, and PCA components input.
        Sets up the initial state of the UI.
        """
        # Clustering Algorithm Selection with better styling
        algo_label = ctk.CTkLabel(
            self, 
            text="Clustering Algorithm:",
            font=ctk.CTkFont(size=14),
            text_color=("gray25", "gray90")
        )
        algo_label.pack(pady=(15,0), padx=15, anchor="w")
        
        self.algo_var = ctk.StringVar(value="kmeans")
        self.algo_menu = ctk.CTkOptionMenu(
            self,
            values=["kmeans", "dbscan", "hierarchical"],
            variable=self.algo_var,
            command=self.on_algorithm_change,
            width=220,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=13)
        )
        self.algo_menu.pack(pady=(5,15), padx=15)
        
        # Create frames for different algorithm parameters
        self.create_kmeans_frame()
        self.create_dbscan_frame()
        self.create_hierarchical_frame()
        
        # Show initial algorithm parameters
        self.on_algorithm_change(self.algo_var.get())
        
        # PCA components with enhanced styling
        pca_label = ctk.CTkLabel(
            self, 
            text="Number of PCA Components:",
            font=ctk.CTkFont(size=14),
            text_color=("gray25", "gray90")
        )
        pca_label.pack(pady=(15,0), padx=15, anchor="w")
        
        self.pca_entry = ctk.CTkEntry(
            self,
            width=220,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=13)
        )
        self.pca_entry.pack(pady=(5,15), padx=15)
        self.pca_entry.insert(0, "3")
        
        # Feature Selection for Clustering with enhanced styling
        features_label = ctk.CTkLabel(
            self, 
            text="Features for Clustering:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("gray25", "gray90")
        )
        features_label.pack(pady=(15,5), padx=15, anchor="w")
        
        # Create scrollable frame for feature selection
        self.features_frame = ctk.CTkScrollableFrame(
            self,
            height=180,
            fg_color=("gray90", "gray17"),
            corner_radius=8
        )
        self.features_frame.pack(fill="x", expand=True, padx=15, pady=(0,15))
        
        # Dictionary to store feature checkboxes
        self.feature_vars = {}

    def update_feature_selection(self, features):
        """
        Updates the feature selection area with checkboxes for available features.
        
        Args:
            features: List of feature names available for clustering
        """
        # Clear existing checkboxes
        for widget in self.features_frame.winfo_children():
            widget.destroy()
        self.feature_vars.clear()
        
        # Add new checkboxes with enhanced styling
        for feature in features:
            var = ctk.BooleanVar(value=True)
            checkbox = ctk.CTkCheckBox(
                self.features_frame,
                text=feature,
                variable=var,
                font=ctk.CTkFont(size=12),
                height=24,
                corner_radius=6
            )
            checkbox.pack(anchor="w", padx=10, pady=2)
            self.feature_vars[feature] = var

    def get_clustering_features(self):
        """
        Returns the list of features selected for clustering.
        
        Returns:
            list: Names of features selected for clustering
        """
        return [feature for feature, var in self.feature_vars.items() if var.get()]

    def create_kmeans_frame(self):
        """Creates the frame for K-means parameters."""
        self.kmeans_frame = ctk.CTkFrame(
            self,
            fg_color=("gray90", "gray17"),
            corner_radius=8
        )
        
        # Number of clusters
        clusters_label = ctk.CTkLabel(
            self.kmeans_frame,
            text="Number of Clusters:",
            font=ctk.CTkFont(size=13),
            text_color=("gray25", "gray90")
        )
        clusters_label.pack(pady=(10,0), padx=10, anchor="w")
        
        self.clusters_entry = ctk.CTkEntry(
            self.kmeans_frame,
            width=200,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=13)
        )
        self.clusters_entry.pack(pady=(5,10), padx=10)
        self.clusters_entry.insert(0, "3")  # Default value

    def create_dbscan_frame(self):
        """Creates the frame for DBSCAN parameters."""
        self.dbscan_frame = ctk.CTkFrame(
            self,
            fg_color=("gray90", "gray17"),
            corner_radius=8
        )
        
        # Epsilon
        eps_label = ctk.CTkLabel(
            self.dbscan_frame,
            text="Epsilon:",
            font=ctk.CTkFont(size=13),
            text_color=("gray25", "gray90")
        )
        eps_label.pack(pady=(10,0), padx=10, anchor="w")
        
        self.eps_entry = ctk.CTkEntry(
            self.dbscan_frame,
            width=200,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=13)
        )
        self.eps_entry.pack(pady=(5,10), padx=10)
        self.eps_entry.insert(0, "0.5")  # Default value
        
        # Min samples
        min_samples_label = ctk.CTkLabel(
            self.dbscan_frame,
            text="Min Samples:",
            font=ctk.CTkFont(size=13),
            text_color=("gray25", "gray90")
        )
        min_samples_label.pack(pady=(5,0), padx=10, anchor="w")
        
        self.min_samples_entry = ctk.CTkEntry(
            self.dbscan_frame,
            width=200,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=13)
        )
        self.min_samples_entry.pack(pady=(5,10), padx=10)
        self.min_samples_entry.insert(0, "5")  # Default value

    def create_hierarchical_frame(self):
        """Creates the frame for Hierarchical Clustering parameters."""
        self.hierarchical_frame = ctk.CTkFrame(
            self,
            fg_color=("gray90", "gray17"),
            corner_radius=8
        )
        
        # Number of clusters
        clusters_label = ctk.CTkLabel(
            self.hierarchical_frame,
            text="Number of Clusters:",
            font=ctk.CTkFont(size=13),
            text_color=("gray25", "gray90")
        )
        clusters_label.pack(pady=(10,0), padx=10, anchor="w")
        
        self.h_clusters_entry = ctk.CTkEntry(
            self.hierarchical_frame,
            width=200,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=13)
        )
        self.h_clusters_entry.pack(pady=(5,10), padx=10)
        self.h_clusters_entry.insert(0, "3")  # Default value

    def on_algorithm_change(self, algorithm):
        """
        Handles the algorithm selection change event.
        Shows/hides relevant parameter frames based on selected algorithm.
        
        Args:
            algorithm: String identifier of the selected algorithm
        """
        # Hide all parameter frames
        self.kmeans_frame.pack_forget()
        self.dbscan_frame.pack_forget()
        self.hierarchical_frame.pack_forget()
        
        # Show relevant parameter frame
        if algorithm == "kmeans":
            self.kmeans_frame.pack(fill="x", padx=5, pady=5)
        elif algorithm == "dbscan":
            self.dbscan_frame.pack(fill="x", padx=5, pady=5)
        elif algorithm == "hierarchical":
            self.hierarchical_frame.pack(fill="x", padx=5, pady=5)

    def get_parameters(self):
        """
        Retrieves all currently set parameters for the selected algorithm.
        
        Returns:
            tuple: (algorithm_name, n_components, algorithm_specific_params)
        """
        algorithm = self.algo_var.get()
        n_components = int(self.pca_entry.get())
        
        params = {}
        if algorithm == "kmeans":
            params["n_clusters"] = int(self.clusters_entry.get())
        elif algorithm == "dbscan":
            params["eps"] = float(self.eps_entry.get())
            params["min_samples"] = int(self.min_samples_entry.get())
        else:  # hierarchical
            params["n_clusters"] = int(self.h_clusters_entry.get())
            
        return algorithm, n_components, params 