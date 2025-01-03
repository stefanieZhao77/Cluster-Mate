import customtkinter as ctk

class CTkScrollableOptionMenu(ctk.CTkFrame):
    def __init__(self, master, values=[], command=None, width=200, height=32):
        super().__init__(master, height=height)
        
        self.values = values
        self.command = command
        self.width = width
        
        # Create the main button
        self.button = ctk.CTkButton(
            self, 
            text="Select Column",
            width=width,
            height=height,
            command=self.show_dropdown
        )
        self.button.pack(fill="x")
        
        # Create dropdown window
        self.dropdown = None
        
    def show_dropdown(self):
        if self.dropdown is None:
            # Create dropdown window
            self.dropdown = ctk.CTkToplevel(self)
            self.dropdown.withdraw()  # Hide initially
            self.dropdown.overrideredirect(True)
            
            # Create search entry
            self.search_var = ctk.StringVar()
            self.search_var.trace_add('write', lambda *args: self.filter_options())
            search_entry = ctk.CTkEntry(
                self.dropdown,
                placeholder_text="Search...",
                textvariable=self.search_var,
                width=self.width-4
            )
            search_entry.pack(padx=2, pady=2)
            
            # Create scrollable frame
            self.scroll_frame = ctk.CTkScrollableFrame(
                self.dropdown, 
                width=self.width,
                height=min(200, len(self.values) * 35)
            )
            self.scroll_frame.pack(fill="both", expand=True)
            
            # Add options
            self.update_options(self.values)
            
            # Position dropdown below button
            x = self.button.winfo_rootx()
            y = self.button.winfo_rooty() + self.button.winfo_height()
            self.dropdown.geometry(f"{self.width}x{min(240, len(self.values) * 35 + 40)}+{x}+{y}")
            
            self.dropdown.deiconify()  # Show dropdown
            
            # Set focus to search entry
            search_entry.focus_set()
            
            # Bind click outside to close dropdown
            self.dropdown.bind('<FocusOut>', self.check_focus_out)
            # Bind escape key to close dropdown
            self.dropdown.bind('<Escape>', lambda e: self.hide_dropdown())
            
        else:
            self.hide_dropdown()
    
    def check_focus_out(self, event):
        # Get the current focused widget
        focused = self.dropdown.focus_get()
        
        # Check if the mouse is inside the dropdown window
        x, y = self.dropdown.winfo_pointerxy()
        widget_under_mouse = self.dropdown.winfo_containing(x, y)
        
        # Don't hide if mouse is inside dropdown or its children
        if widget_under_mouse and (
            widget_under_mouse == self.dropdown or 
            widget_under_mouse.winfo_toplevel() == self.dropdown or
            widget_under_mouse == self.button or
            'scrollbar' in str(widget_under_mouse).lower()  # Also handle scrollbar
        ):
            return
            
        # Hide if focus is completely lost
        if focused is None:
            self.hide_dropdown()
    
    def filter_options(self, *args):
        search_text = self.search_var.get().lower()
        filtered_values = [v for v in self.values if search_text in v.lower()]
        self.update_options(filtered_values)
    
    def update_options(self, values):
        # Clear existing options
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        
        # Add filtered options
        for value in values:
            btn = ctk.CTkButton(
                self.scroll_frame,
                text=value,
                command=lambda v=value: self.select_value(v)
            )
            btn.pack(fill="x", padx=2, pady=1)
            
    def hide_dropdown(self):
        if self.dropdown:
            self.dropdown.destroy()
            self.dropdown = None
            
    def select_value(self, value):
        self.button.configure(text=value)
        self.hide_dropdown()
        if self.command:
            self.command(value)
            
    def get(self):
        return self.button.cget("text")
    
    def set(self, value):
        self.button.configure(text=value)
        
    def configure(self, **kwargs):
        if "values" in kwargs:
            self.values = kwargs["values"]
            if self.dropdown:
                self.hide_dropdown() 