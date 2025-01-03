import customtkinter as ctk

class SideMenu(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=200, **kwargs)
        self.grid_rowconfigure(10, weight=1)  # Push everything up
        
        # Title
        title = ctk.CTkLabel(
            self,
            text="Training Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(pady=10, padx=10)
        
        # Create parameters
        self.create_parameter_widgets()
        
    def create_parameter_widgets(self):
        # Clustering Algorithm Selection
        algo_label = ctk.CTkLabel(self, text="Clustering Algorithm:")
        algo_label.pack(pady=(10,0), padx=10, anchor="w")
        
        self.algo_var = ctk.StringVar(value="kmeans")
        self.algo_menu = ctk.CTkOptionMenu(
            self,
            values=["kmeans", "dbscan", "hierarchical"],
            variable=self.algo_var,
            command=self.on_algorithm_change
        )
        self.algo_menu.pack(pady=(5,10), padx=10)
        
        # Create frames for different algorithm parameters
        self.create_kmeans_frame()
        self.create_dbscan_frame()
        self.create_hierarchical_frame()
        
        # Show initial algorithm parameters
        self.on_algorithm_change(self.algo_var.get())
        
        # PCA components
        pca_label = ctk.CTkLabel(self, text="Number of PCA Components:")
        pca_label.pack(pady=(10,0), padx=10, anchor="w")
        
        self.pca_entry = ctk.CTkEntry(self)
        self.pca_entry.pack(pady=(5,10), padx=10)
        self.pca_entry.insert(0, "3")  # Default value

    def create_kmeans_frame(self):
        self.kmeans_frame = ctk.CTkFrame(self)
        clusters_label = ctk.CTkLabel(self.kmeans_frame, text="Number of Clusters:")
        clusters_label.pack(pady=(10,0), padx=10, anchor="w")
        self.clusters_entry = ctk.CTkEntry(self.kmeans_frame)
        self.clusters_entry.pack(pady=(5,10), padx=10)
        self.clusters_entry.insert(0, "3")

    def create_dbscan_frame(self):
        self.dbscan_frame = ctk.CTkFrame(self)
        eps_label = ctk.CTkLabel(self.dbscan_frame, text="Epsilon (eps):")
        eps_label.pack(pady=(10,0), padx=10, anchor="w")
        self.eps_entry = ctk.CTkEntry(self.dbscan_frame)
        self.eps_entry.pack(pady=(5,10), padx=10)
        self.eps_entry.insert(0, "0.5")
        
        min_samples_label = ctk.CTkLabel(self.dbscan_frame, text="Minimum Samples:")
        min_samples_label.pack(pady=(10,0), padx=10, anchor="w")
        self.min_samples_entry = ctk.CTkEntry(self.dbscan_frame)
        self.min_samples_entry.pack(pady=(5,10), padx=10)
        self.min_samples_entry.insert(0, "5")

    def create_hierarchical_frame(self):
        self.hierarchical_frame = ctk.CTkFrame(self)
        clusters_label = ctk.CTkLabel(self.hierarchical_frame, text="Number of Clusters:")
        clusters_label.pack(pady=(10,0), padx=10, anchor="w")
        self.hierarchical_clusters_entry = ctk.CTkEntry(self.hierarchical_frame)
        self.hierarchical_clusters_entry.pack(pady=(5,10), padx=10)
        self.hierarchical_clusters_entry.insert(0, "3")

    def on_algorithm_change(self, algorithm):
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
        algorithm = self.algo_var.get()
        n_components = int(self.pca_entry.get())
        
        params = {}
        if algorithm == "kmeans":
            params["n_clusters"] = int(self.clusters_entry.get())
        elif algorithm == "dbscan":
            params["eps"] = float(self.eps_entry.get())
            params["min_samples"] = int(self.min_samples_entry.get())
        else:  # hierarchical
            params["n_clusters"] = int(self.hierarchical_clusters_entry.get())
            
        return algorithm, n_components, params 