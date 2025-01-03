import customtkinter as ctk
from pathlib import Path
import pandas as pd
from tkinter import filedialog
from ..widgets.scrollable_menu import CTkScrollableOptionMenu

class DatasetFrame(ctk.CTkFrame):
    def __init__(self, master, frame_id, on_remove=None, **kwargs):
        super().__init__(master, **kwargs)
        self.frame_id = frame_id
        self.on_remove = on_remove
        self.feature_vars = {}
        self.setup_ui()
        
    def setup_ui(self):
        # File selection row
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(fill="x", padx=5, pady=2)
        
        # Remove button for non-first frames
        if self.frame_id > 1:
            remove_btn = ctk.CTkButton(
                self.file_frame,
                text="×",
                width=30,
                command=self.remove_frame
            )
            remove_btn.pack(side="left", padx=(5,0))
        
        file_label = ctk.CTkLabel(self.file_frame, text=f"Dataset {self.frame_id + 1}:")
        file_label.pack(side="left", padx=5)
        
        self.file_entry = ctk.CTkEntry(self.file_frame, width=300)
        self.file_entry.pack(side="left", padx=5)
        
        # Features selection frame
        self.features_frame = ctk.CTkFrame(self)
        self.features_frame.pack(fill="x", padx=5, pady=2)
        
        # Split into two parts: link column and features
        selection_frame = ctk.CTkFrame(self.features_frame)
        selection_frame.pack(fill="x", padx=5, pady=2)
        
        # Link column selection
        link_label = ctk.CTkLabel(selection_frame, text="Link Column:")
        link_label.pack(side="left", padx=5)
        
        self.link_combobox = CTkScrollableOptionMenu(
            selection_frame,
            values=[],
            width=200
        )
        self.link_combobox.pack(side="left", padx=5)
        
        # Time selection frame
        self.time_frame = ctk.CTkFrame(selection_frame)
        self.time_frame.pack(fill="x", padx=5, pady=2)
        
        # Date column selection
        baseline_label = ctk.CTkLabel(self.time_frame, text="Date Column:")
        baseline_label.pack(side="left", padx=(20,5))
        
        self.baseline_combobox = CTkScrollableOptionMenu(
            self.time_frame,
            values=[],
            width=200
        )
        self.baseline_combobox.pack(side="left", padx=5)
        
        # Time bias input
        bias_label = ctk.CTkLabel(self.time_frame, text="Time Bias (days):")
        bias_label.pack(side="left", padx=(20,5))
        
        self.bias_entry = ctk.CTkEntry(
            self.time_frame,
            width=80,
            placeholder_text="0"
        )
        self.bias_entry.pack(side="left", padx=5)

        # Add baseline type selection for first dataset only
        if self.frame_id == 0:
            baseline_type_label = ctk.CTkLabel(self.time_frame, text="Baseline Type:")
            baseline_type_label.pack(side="left", padx=(20,5))
            
            self.baseline_type_var = ctk.StringVar(value="Earliest")
            self.baseline_type_menu = ctk.CTkOptionMenu(
                self.time_frame,
                values=["Earliest", "Latest"],
                variable=self.baseline_type_var,
                width=100
            )
            self.baseline_type_menu.pack(side="left", padx=5)
        else:
            self.baseline_type_var = None

        # Browse button
        browse_button = ctk.CTkButton(
            self.file_frame,
            text="Browse",
            width=100,
            command=self.browse_file
        )
        browse_button.pack(side="left", padx=5)

        # Features box
        features_label = ctk.CTkLabel(self.features_frame, text="Select Features:")
        features_label.pack(anchor="w", padx=5, pady=(5,0))
        
        self.features_box = ctk.CTkScrollableFrame(self.features_frame, height=150)
        self.features_box.pack(fill="x", expand=True, padx=5, pady=5)
        
        # Configure grid for features_box
        self.features_box.grid_columnconfigure((0,1,2), weight=1)

    def browse_file(self):
        filetypes = (
            ('Excel files', '*.xlsx'),
            ('CSV files', '*.csv')
        )
        
        filename = filedialog.askopenfilename(
            title='Select a file',
            filetypes=filetypes
        )
        
        if filename:
            self.file_entry.delete(0, ctk.END)
            self.file_entry.insert(0, filename)
            
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filename, low_memory=False)
                else:
                    df = pd.read_excel(filename, low_memory=False)
                
                # Update link column options
                columns = df.columns.tolist()
                self.link_combobox.configure(values=columns)
                if columns:
                    self.link_combobox.set(columns[0])
                
                # Update baseline column options
                self.baseline_combobox.configure(values=columns)
                if columns:
                    self.baseline_combobox.set(columns[0])
                
                # Update feature checkboxes
                self.update_feature_checkboxes(df.columns)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {str(e)}")

    def update_feature_checkboxes(self, columns):
        # Clear existing checkboxes
        for widget in self.features_box.winfo_children():
            widget.destroy()
        self.feature_vars.clear()
        
        # Calculate layout
        num_features = len(columns)
        num_cols = min(4, max(3, num_features // 8))
        
        # Create new checkboxes
        for i, col in enumerate(columns):
            var = ctk.BooleanVar()
            checkbox = ctk.CTkCheckBox(
                self.features_box,
                text=col,
                variable=var,
                width=180
            )
            row = i // num_cols
            col_pos = i % num_cols
            checkbox.grid(row=row, column=col_pos, padx=5, pady=1, sticky="w")
            self.feature_vars[col] = var

    def remove_frame(self):
        if self.on_remove:
            self.on_remove(self)

    def get_data(self):
        return {
            'file': self.file_entry.get(),
            'link_column': self.link_combobox.get(),
            'date_column': self.baseline_combobox.get(),
            'time_bias': self.bias_entry.get(),
            'baseline_type': self.baseline_type_var.get() if self.baseline_type_var else None,
            'selected_features': [col for col, var in self.feature_vars.items() if var.get()]
        } 