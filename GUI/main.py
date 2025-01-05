import os
import sys
import shutil
import customtkinter as ctk
from tkinter import messagebox, filedialog
from pathlib import Path
import pandas as pd
import numpy as np

# Add the current directory to Python path
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle
    application_path = sys._MEIPASS
else:
    # If the application is run from a Python interpreter
    application_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(application_path))

from GUI.frames.side_menu import SideMenu
from GUI.frames.dataset_frame import DatasetFrame
from GUI.clustering.train import main
from PIL import Image, ImageTk

class ClusteringGUI(ctk.CTk):
    """
    Main application window for the Clustering Training Tool.
    Handles the overall GUI layout and coordination between different components.
    """
    def __init__(self):
        """
        Initialize the main application window with default settings and layout.
        Sets up the side menu and main container for dataset frames.
        """
        super().__init__()
        self.dataset_frames = []
        self.selected_features = []
        
        # Configure window
        self.title("Clustering Training Tool")
        self.geometry("1200x800")
        
        # Set default appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Configure window style
        self.configure(fg_color=("gray92", "gray14"))
        
        # Create main layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create side menu with shadow effect
        side_menu_container = ctk.CTkFrame(self, fg_color=("gray85", "gray17"))
        side_menu_container.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        self.side_menu = SideMenu(side_menu_container)
        self.side_menu.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Create main container with shadow effect
        container_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray17"))
        container_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        
        self.container = ctk.CTkFrame(container_frame, fg_color=("gray95", "gray13"))
        self.container.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Create data section
        self.create_data_section()
        
    def create_data_section(self):
        """
        Creates the main data section of the application.
        Sets up the frame for dataset entries, including the header with title and add button,
        and a scrollable container for multiple dataset frames.
        """
        # Data Frame with enhanced styling
        data_frame = ctk.CTkFrame(self.container, fg_color=("gray90", "gray15"))
        data_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Title and Add Button row with better styling
        header_frame = ctk.CTkFrame(data_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        title_label = ctk.CTkLabel(
            header_frame, 
            text="Data Import", 
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("gray25", "gray90")
        )
        title_label.pack(side="left", pady=5, padx=5)
        
        add_button = ctk.CTkButton(
            header_frame,
            text="+ Add Dataset",
            command=self.add_dataset_frame,
            font=ctk.CTkFont(size=13),
            height=32,
            corner_radius=8
        )
        add_button.pack(side="right", pady=5, padx=5)
        
        # Scrollable frame with enhanced styling
        self.scrollable_frame = ctk.CTkScrollableFrame(
            data_frame,
            fg_color=("gray95", "gray13"),
            corner_radius=10
        )
        self.scrollable_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        
        # Create button frame with better styling
        self.buttons_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        self.buttons_frame.pack(pady=10)
        
        preview_button = ctk.CTkButton(
            self.buttons_frame,
            text="Preview Combined Data",
            command=self.preview_combined_data,
            font=ctk.CTkFont(size=13),
            height=32,
            corner_radius=8
        )
        preview_button.pack(side="left", padx=5)
        
        self.train_button = ctk.CTkButton(
            self.buttons_frame,
            text="Train Model",
            command=self.train_model,
            state="disabled",
            font=ctk.CTkFont(size=13),
            height=32,
            corner_radius=8
        )
        self.train_button.pack(side="left", padx=5)
        
        # Add initial dataset frames
        self.add_dataset_frame()
        self.add_dataset_frame()

    def create_button_frame(self):
        """
        Creates the frame containing the Preview and Train buttons.
        These buttons are used to preview combined data and initiate model training.
        """
        button_frame = ctk.CTkFrame(self.scrollable_frame)
        button_frame.pack(pady=10)
        
        preview_button = ctk.CTkButton(
            button_frame,
            text="Preview Combined Data",
            command=self.preview_combined_data
        )
        preview_button.pack(side="left", padx=5)
        
        self.train_button = ctk.CTkButton(
            button_frame,
            text="Train Model",
            command=self.train_model,
            state="disabled"
        )
        self.train_button.pack(side="left", padx=5)

    def add_dataset_frame(self):
        """
        Adds a new dataset frame to the scrollable container.
        Each frame allows users to load and configure a dataset for clustering.
        """
        # Create new dataset frame
        new_frame = DatasetFrame(self, frame_id=len(self.dataset_frames), on_remove=self.remove_dataset_frame)
        
        # Pack it before the buttons_frame
        new_frame.repack_before(self.buttons_frame)  # assuming buttons_frame is your frame containing preview/train buttons
        
        # Add to list of frames
        self.dataset_frames.append(new_frame)

    def remove_dataset_frame(self, frame):
        """
        Removes a dataset frame from the GUI and updates the remaining frames.
        
        Args:
            frame: The DatasetFrame instance to be removed
        """
        self.dataset_frames.remove(frame)
        frame.destroy()
        # Update remaining frames
        self.update_dataset_frames()

    def update_dataset_frames(self):
        """
        Updates the IDs and appearance of all dataset frames after a frame is removed.
        Ensures the first frame cannot be removed as it serves as the baseline.
        """
        for i, frame in enumerate(self.dataset_frames):
            frame.frame_id = i
            # Update frame appearance/styling if needed
            if i == 0:  # First frame should not have remove button
                for child in frame.file_frame.winfo_children():
                    if isinstance(child, ctk.CTkButton) and child.cget("text") == "×":
                        child.destroy()

    def preview_combined_data(self):
        """
        Processes and combines data from all loaded datasets.
        Shows a loading window during processing and displays the combined data in a preview window.
        Handles data alignment based on link columns and time bias settings.
        """
        try:
            # Create loading window with progress circle
            loading_window = ctk.CTkToplevel(self)
            loading_window.title("Loading")
            loading_window.geometry("300x150")
            loading_window.transient(self)  # Make it float on top of main window
            
            # Center the loading window
            loading_window.update_idletasks()
            x = self.winfo_x() + (self.winfo_width() // 2) - (loading_window.winfo_width() // 2)
            y = self.winfo_y() + (self.winfo_height() // 2) - (loading_window.winfo_height() // 2)
            loading_window.geometry(f"+{x}+{y}")
            
            # Add progress circle and label
            progress = ctk.CTkProgressBar(loading_window, mode="indeterminate")
            progress.pack(pady=20)
            progress.start()
            
            label = ctk.CTkLabel(loading_window, text="Processing data...\nPlease wait...")
            label.pack(pady=10)
            
            # Update GUI
            loading_window.update()
            
            # Get data from all frames
            valid_frames = [frame for frame in self.dataset_frames if frame.file_entry.get()]
            
            if len(valid_frames) < 2:
                loading_window.destroy()
                messagebox.showwarning("Warning", "Please load at least two datasets")
                return
            
            # Process baseline dataset
            baseline_frame = valid_frames[0]
            baseline_data = baseline_frame.get_data()
            
            # Load and process baseline
            baseline_df = pd.read_csv(baseline_data['file']) if baseline_data['file'].endswith('.csv') \
                        else pd.read_excel(baseline_data['file'])
            
            # Process baseline date
            baseline_df[baseline_data['date_column']] = pd.to_datetime(baseline_df[baseline_data['date_column']]).dt.tz_localize(None)
            
            # Get baseline rows based on type
            if baseline_data['baseline_type'] == "Latest":
                idx = baseline_df.groupby(baseline_data['link_column'])[baseline_data['date_column']].idxmax()
            else:
                idx = baseline_df.groupby(baseline_data['link_column'])[baseline_data['date_column']].idxmin()
            baseline_df = baseline_df.loc[idx]
            
            # Select features and rename columns
            baseline_features = [baseline_data['link_column'], baseline_data['date_column']] + baseline_data['selected_features']
            result_df = baseline_df[baseline_features].copy()
            
            baseline_name = Path(baseline_data['file']).name
            result_df.columns = [col if col in [baseline_data['link_column'], baseline_data['date_column']] 
                               else f"{baseline_name}_{col}" for col in baseline_features]
            
            # Store selected features
            self.selected_features = [f"{baseline_name}_{col}" for col in baseline_data['selected_features']]
            
            # Process additional datasets
            for frame in valid_frames[1:]:
                frame_data = frame.get_data()
                df = pd.read_csv(frame_data['file']) if frame_data['file'].endswith('.csv') \
                    else pd.read_excel(frame_data['file'])
                
                # Process date column
                df[frame_data['date_column']] = pd.to_datetime(df[frame_data['date_column']]).dt.tz_localize(None)
                
                # Get selected features
                selected_features = [frame_data['link_column'], frame_data['date_column']] + frame_data['selected_features']
                temp_df = df[selected_features].copy()
                
                # Create new column names
                dataset_name = Path(frame_data['file']).name
                new_columns = [f"{dataset_name}_{col}" for col in frame_data['selected_features']]
                self.selected_features.extend(new_columns)
                
                # Add columns with NaN values
                for col in new_columns:
                    result_df[col] = np.nan
                
                # Add bias days column
                bias_col = f"{dataset_name}_bias_days"
                result_df[bias_col] = np.nan
                
                # Process each baseline row
                try:
                    time_bias = int(frame_data['time_bias'] or 0)
                except (ValueError, TypeError):
                    time_bias = 0
                    messagebox.showwarning("Warning", 
                                         f"Invalid time bias value for {dataset_name}. Using 0 days instead.")
                
                for idx, base_row in result_df.iterrows():
                    matching_rows = temp_df[temp_df[frame_data['link_column']] == base_row[baseline_data['link_column']]]
                    
                    if not matching_rows.empty:
                        matching_rows.loc[:, 'date_diff'] = abs(matching_rows[frame_data['date_column']] - base_row[baseline_data['date_column']])
                        closest_row = matching_rows.nsmallest(1, 'date_diff').iloc[0]
                        
                        days_diff = (closest_row[frame_data['date_column']] - base_row[baseline_data['date_column']]).total_seconds() / (24*3600)
                        days_diff = int(round(days_diff))  # Convert to integer
                        
                        if abs(days_diff) <= time_bias:
                            for feature in frame_data['selected_features']:
                                new_col = f"{dataset_name}_{feature}"
                                result_df.loc[idx, new_col] = closest_row[feature]
                            result_df.loc[idx, bias_col] = days_diff
            
            # Drop rows with any empty values
            result_df = result_df.dropna()
            
            # Save combined data
            result_df.to_csv('./cluster_data.csv', index=False)
            
            # After processing is complete
            loading_window.destroy()
            
            # Update feature selection in side menu
            numeric_features = self.selected_features
            self.side_menu.update_feature_selection(numeric_features)
            
            # Show preview window
            self.show_preview_window(result_df)
            
            # Enable train button
            self.train_button.configure(state="normal")
            
        except Exception as e:
            if 'loading_window' in locals():
                loading_window.destroy()
            messagebox.showerror("Error", f"Error previewing data: {str(e)}")

    def show_preview_window(self, df):
        """
        Creates and displays a window showing the preview of combined dataset.
        
        Args:
            df: pandas DataFrame containing the combined data to be displayed
        """
        preview_window = ctk.CTkToplevel(self)
        preview_window.title("Combined Data Preview")
        preview_window.geometry("1000x700")
        preview_window.state("zoomed")
        
        # Add preview info
        info_text = (
            f"Combined Shape: {df.shape}\n"
            f"Total Features: {len(df.columns)}"
        )
        info_label = ctk.CTkLabel(
            preview_window,
            text=info_text,
            wraplength=950,
            justify="left"
        )
        info_label.pack(pady=(10,5), padx=10)
        
        # Show columns
        columns_frame = ctk.CTkFrame(preview_window)
        columns_frame.pack(fill="x", padx=10, pady=5)
        
        columns_label = ctk.CTkLabel(
            columns_frame,
            text="Columns: " + ", ".join(df.columns),
            wraplength=950,
            justify="left"
        )
        columns_label.pack(pady=5, padx=5)
        
        # Create table preview
        self.create_table_preview(preview_window, df)

    def create_table_preview(self, window, df):
        """
        Creates an interactive table preview with pagination and column operations.
        
        Args:
            window: The parent window to contain the table
            df: pandas DataFrame to be displayed in the table
        """
        # Create main container for table and controls
        main_container = ctk.CTkFrame(window)
        main_container.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Add controls frame
        controls_frame = ctk.CTkFrame(main_container)
        controls_frame.pack(fill="x", padx=2, pady=5)
        
        # Left side: Operation controls
        left_controls = ctk.CTkFrame(controls_frame)
        left_controls.pack(side="left", fill="x", expand=True, padx=5)
        
        # Operation selection row
        operation_row = ctk.CTkFrame(left_controls)
        operation_row.pack(fill="x", pady=(0, 2))  # Reduced bottom padding
        
        operation_label = ctk.CTkLabel(operation_row, text="Operation:", width=80)
        operation_label.pack(side="left", padx=5)
        
        operation_var = ctk.StringVar(value="mean")
        operation_dropdown = ctk.CTkOptionMenu(
            operation_row,
            values=["mean", "sum", "min", "max"],
            variable=operation_var,
            width=100
        )
        operation_dropdown.pack(side="left", padx=5)
        
        name_label = ctk.CTkLabel(operation_row, text="New column:", width=80)
        name_label.pack(side="left", padx=5)
        
        name_entry = ctk.CTkEntry(operation_row, placeholder_text="New column name", width=150)
        name_entry.pack(side="left", padx=5)
        
        # Create scrollable frame for feature selection
        features_label = ctk.CTkLabel(
            left_controls, 
            text="Select features for operation:",
            font=ctk.CTkFont(size=13)  # Slightly smaller font
        )
        features_label.pack(anchor="w", padx=5, pady=(2,0))  # Reduced top padding
        
        features_frame = ctk.CTkScrollableFrame(left_controls, height=100)
        features_frame.pack(fill="x", expand=True, padx=5, pady=(2,5))  # Reduced top padding
        
        # Create grid layout for checkboxes
        num_columns = 3  # Number of columns for checkboxes
        selected_columns = []
        column_vars = {}
        
        def update_selection():
            selected_columns.clear()
            selected_columns.extend([col for col, var in column_vars.items() if var.get()])
            calculate_button.configure(state="normal" if len(selected_columns) > 1 else "disabled")
        
        # Add checkboxes in a grid layout
        for i, col in enumerate(df.columns):
            var = ctk.BooleanVar(value=False)
            checkbox = ctk.CTkCheckBox(
                features_frame,
                text=col,
                variable=var,
                command=update_selection,
                width=200
            )
            row = i // num_columns
            col_pos = i % num_columns
            checkbox.grid(row=row, column=col_pos, padx=5, pady=2, sticky="w")
            column_vars[col] = var
            
        # Configure grid columns to be evenly spaced
        for i in range(num_columns):
            features_frame.grid_columnconfigure(i, weight=1)
        
        # Right side: Calculate button
        right_controls = ctk.CTkFrame(controls_frame)
        right_controls.pack(side="right", padx=5)
        
        def calculate_new_column():
            if not name_entry.get():
                messagebox.showwarning("Warning", "Please enter a name for the new column")
                return
                
            try:
                # Get selected columns data
                selected_data = df[selected_columns]
                
                # Perform calculation
                operation = operation_var.get()
                if operation == "mean":
                    new_values = selected_data.mean(axis=1)
                elif operation == "sum":
                    new_values = selected_data.sum(axis=1)
                elif operation == "min":
                    new_values = selected_data.min(axis=1)
                elif operation == "max":
                    new_values = selected_data.max(axis=1)
                
                new_column_name = name_entry.get()
                
                # Add new column
                df[new_column_name] = new_values
                
                # Update selected_features
                # Remove old features and add new one
                self.selected_features = [f for f in self.selected_features if f not in selected_columns]
                self.selected_features.append(new_column_name)
                
                # Remove old columns
                df.drop(columns=selected_columns, inplace=True)
                
                # Save updated data to CSV
                df.to_csv('./cluster_data.csv', index=False)
                
                # Destroy old window and create new one
                window.destroy()
                self.show_preview_window(df)
                
            except Exception as e:
                messagebox.showerror("Error", f"Calculation failed: {str(e)}")
        
        # Add calculate button
        calculate_button = ctk.CTkButton(
            right_controls,
            text="Calculate",
            command=calculate_new_column,
            state="disabled",
            width=120
        )
        calculate_button.pack(pady=5)
        
        # Create table content
        self.create_table_content(main_container, df)
    
    def create_table_content(self, container, df):
        """
        Creates the actual table content with headers and data cells.
        Implements pagination for large datasets.
        
        Args:
            container: The container widget to hold the table
            df: pandas DataFrame to be displayed
        """
        # Create scrollable frame for table
        table_frame = ctk.CTkScrollableFrame(container, width=980, height=500)
        table_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Pagination settings
        self.rows_per_page = 30
        self.current_page = 0
        self.total_pages = (len(df) - 1) // self.rows_per_page + 1
        
        # Create header
        header_frame = ctk.CTkFrame(table_frame)
        header_frame.pack(fill="x", padx=2, pady=(0, 5))
        
        # Calculate optimal column widths
        col_widths = {}
        max_width = 300  # Maximum column width
        min_width = 100  # Minimum column width
        padding = 20     # Extra padding for text
        
        for col in df.columns:
            # Get maximum width of column name and values
            col_width = max(
                len(str(col)),
                df[col].astype(str).str.len().max()
            )
            # Convert character count to pixels (approximate)
            col_widths[col] = min(max(col_width * 8 + padding, min_width), max_width)
        
        # Add headers
        for i, col in enumerate(df.columns):
            header_label = ctk.CTkLabel(
                header_frame,
                text=str(col),
                width=col_widths[col],
                height=25,
                fg_color=("gray85", "gray25"),
                corner_radius=4,
                wraplength=col_widths[col] - padding  # Enable text wrapping
            )
            header_label.grid(row=0, column=i, padx=1, pady=1, sticky="ew")
            header_frame.grid_columnconfigure(i, weight=0)
        
        def update_table(page):
            # Clear existing rows
            for widget in table_frame.winfo_children():
                if widget != header_frame:
                    widget.destroy()
            
            # Calculate start and end indices
            start_idx = page * self.rows_per_page
            end_idx = min(start_idx + self.rows_per_page, len(df))
            
            # Add data rows for current page
            for row_idx, (_, row) in enumerate(df.iloc[start_idx:end_idx].iterrows()):
                row_frame = ctk.CTkFrame(table_frame)
                row_frame.pack(fill="x", padx=2, pady=1)
                
                for col_idx, (col, value) in enumerate(row.items()):
                    formatted_value = self.format_cell_value(value)
                    
                    cell = ctk.CTkLabel(
                        row_frame,
                        text=formatted_value,
                        width=col_widths[col],
                        height=25,
                        fg_color=("gray95" if row_idx % 2 == 0 else "gray90", 
                                 "gray20" if row_idx % 2 == 0 else "gray15"),
                        corner_radius=4,
                        wraplength=col_widths[col] - padding  # Enable text wrapping
                    )
                    cell.grid(row=0, column=col_idx, padx=1, pady=1, sticky="ew")
                    row_frame.grid_columnconfigure(col_idx, weight=0)
            
            # Update page info label
            page_info.configure(text=f"Page {page + 1} of {self.total_pages}")
            
            # Update button states
            prev_button.configure(state="normal" if page > 0 else "disabled")
            next_button.configure(state="normal" if page < self.total_pages - 1 else "disabled")
        
        # Create pagination controls
        pagination_frame = ctk.CTkFrame(container)
        pagination_frame.pack(fill="x", padx=10, pady=5)
        
        prev_button = ctk.CTkButton(
            pagination_frame,
            text="Previous",
            width=100,
            command=lambda: update_table(self.current_page - 1)
        )
        prev_button.pack(side="left", padx=5)
        
        page_info = ctk.CTkLabel(
            pagination_frame,
            text=f"Page 1 of {self.total_pages}"
        )
        page_info.pack(side="left", padx=5)
        
        next_button = ctk.CTkButton(
            pagination_frame,
            text="Next",
            width=100,
            command=lambda: update_table(self.current_page + 1)
        )
        next_button.pack(side="left", padx=5)
        
        # Show initial page
        update_table(0)

    @staticmethod
    def format_cell_value(value):
        """
        Formats cell values for display in the table.
        
        Args:
            value: The value to be formatted
            
        Returns:
            str: Formatted string representation of the value
        """
        if pd.isna(value):
            return ""
        elif isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        elif isinstance(value, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S')
        return str(value)

    def train_model(self):
        """
        Initiates the model training process using the combined dataset.
        Creates a progress window to show training status and handles the download
        of training metrics upon completion.
        """
        try:
            # Get training parameters from side menu
            algorithm, n_components, params = self.side_menu.get_parameters()
            
            # Create training window
            training_window = ctk.CTkToplevel(self)
            training_window.title("Model Training")
            training_window.geometry("900x600")  # Increased window size
            training_window.minsize(900, 600)    # Set minimum size
            training_window.resizable(True, True)  # Allow resizing
            
            # Make window maximizable
            training_window.state('zoomed')  # For Windows
            
            # Center the window on screen
            screen_width = training_window.winfo_screenwidth()
            screen_height = training_window.winfo_screenheight()
            x = (screen_width - 900) // 2
            y = (screen_height - 600) // 2
            training_window.geometry(f"+{x}+{y}")
            
            # Add loading label and progress bar
            loading_label = ctk.CTkLabel(
                training_window,
                text=f"Training {algorithm.upper()} model...\nPlease wait...",
                font=ctk.CTkFont(size=16, weight="bold")
            )
            loading_label.pack(pady=20)
            
            progress = ctk.CTkProgressBar(training_window, mode="indeterminate")
            progress.pack(pady=10)
            progress.start()
            
            # Update GUI
            training_window.update()
            
            # Get features for clustering from side menu
            clustering_features = self.side_menu.get_clustering_features()
            if not clustering_features:
                clustering_features = self.selected_features
            
            # Train model with both clustering features and all selected features
            main(
                n_components=n_components,
                model_type=algorithm,
                model_params=params,
                selected_features=clustering_features,  # Features used for clustering
                all_features=self.selected_features     # All features to include in output
            )
            
            # Stop progress bar and update label
            progress.stop()
            progress.destroy()
            loading_label.configure(text="Training completed successfully!")
            
            # Create scrollable frame for plots
            scroll_frame = ctk.CTkScrollableFrame(training_window)
            scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Load and display plots
            metrics_dir = Path("metrics/cluster_plots")
            self.plot_images = []  # Store references to prevent garbage collection
            
            for plot_file in metrics_dir.glob("*.png"):
                # Create frame for each plot
                plot_frame = ctk.CTkFrame(scroll_frame)
                plot_frame.pack(pady=10, fill="x", padx=10)
                
                # Add plot title
                title = plot_file.stem.replace('_', ' ').title()
                title_label = ctk.CTkLabel(
                    plot_frame,
                    text=title,
                    font=ctk.CTkFont(size=14, weight="bold")
                )
                title_label.pack(pady=5)
                
                # Load image
                img = Image.open(plot_file)
                
                # Calculate new dimensions while maintaining aspect ratio
                max_width = 800  # Maximum width for the plot
                max_height = 500  # Maximum height for the plot
                
                # Calculate scaling factor
                width_ratio = max_width / img.width
                height_ratio = max_height / img.height
                scale_factor = min(width_ratio, height_ratio)
                
                # Calculate new dimensions
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                
                # Resize image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.plot_images.append(img)  # Store reference
                
                # Create canvas for the image
                canvas = ctk.CTkCanvas(
                    plot_frame,
                    width=new_width,
                    height=new_height,
                    bg=training_window._apply_appearance_mode(training_window._fg_color),
                    highlightthickness=0  # Remove canvas border
                )
                canvas.pack(pady=5)
                
                # Convert PIL image to PhotoImage and store reference
                photo = ImageTk.PhotoImage(img)
                self.plot_images.append(photo)  # Store reference
                
                # Create image on canvas
                canvas.create_image(
                    new_width // 2,
                    new_height // 2,
                    image=photo,
                    anchor="center"
                )

            # Add download button
            def download_metrics():
                save_dir = filedialog.askdirectory(title="Select Download Location")
                if save_dir:
                    try:
                        shutil.make_archive(
                            os.path.join(save_dir, "clustering_metrics"),
                            'zip',
                            "metrics"
                        )
                        messagebox.showinfo(
                            "Success",
                            f"Metrics downloaded to {save_dir}/clustering_metrics.zip"
                        )
                    except Exception as e:
                        messagebox.showerror("Error", f"Download failed: {str(e)}")

            download_btn = ctk.CTkButton(
                training_window,
                text="Download Metrics",
                command=download_metrics
            )
            download_btn.pack(pady=10)
            
        except Exception as e:
            if 'loading_window' in locals():
                loading_window.destroy()
            messagebox.showerror("Error", f"Error during training: {str(e)}")

if __name__ == "__main__":
    app = ClusteringGUI()
    app.mainloop() 