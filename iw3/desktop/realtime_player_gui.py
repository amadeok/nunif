import shlex
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
import subprocess
import sys
import threading
from tkinterdnd2 import DND_FILES, TkinterDnD

class TextWidgetWithUndo:
    def __init__(self, parent):
        args_frame = ttk.LabelFrame(parent, text="Command Line Arguments")
        args_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.args_text = scrolledtext.ScrolledText(args_frame, height=10, wrap=tk.WORD, undo=True)
        self.args_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Enable undo/redo functionality
        self.args_text.config(undo=True)
        
        # Bind Ctrl+Z for undo and Ctrl+Y for redo
        self.args_text.bind('<Control-z>', self.undo)
        self.args_text.bind('<Control-Shift-Z>', self.redo)
        
        # Add some common argument examples as placeholder
        self.args_text.insert(tk.END, "# Enter your command line arguments here\n")
        self.args_text.insert(tk.END, "# Example: --input file.txt --verbose --output results/")
        
        # Set up tags for text coloring if desired
        self.args_text.tag_configure("comment", foreground="gray")
        self.args_text.tag_add("comment", "1.0", "2.0")
        
    def undo(self, event):
        try:
            self.args_text.edit_undo()
        except tk.TclError:
            # Nothing to undo
            pass
        return "break"  # Prevent default behavior
    
    def redo(self, event):
        try:
            self.args_text.edit_redo()
        except tk.TclError:
            # Nothing to redo
            pass
        return "break"  # Prevent default behavior

class ArgumentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PyArgument - GUI for argparse")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)
        
        # Variables
        self.profiles = {}
        self.current_profile = None
        self.process = None
        self.dropped_file = None

        self.create_widgets()
        self.load_profiles()
        
    def create_widgets(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=2)

        self.build_left_panel(left_frame)
        
        self.build_right_panel(left_frame)
    
    
    def build_left_panel(self, parent):
        # Profile section
        profile_frame = ttk.LabelFrame(parent, text="Profiles")
        profile_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(profile_frame, text="Profile:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(profile_frame, textvariable=self.profile_var, state="readonly")
        self.profile_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.profile_combo.bind('<<ComboboxSelected>>', self.profile_selected)
        
        profile_buttons = ttk.Frame(profile_frame)
        profile_buttons.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(profile_buttons, text="Save", command=self.save_profile).pack(side=tk.LEFT, padx=2)
        ttk.Button(profile_buttons, text="Delete", command=self.delete_profile).pack(side=tk.LEFT, padx=2)
        ttk.Button(profile_buttons, text="New", command=self.new_profile).pack(side=tk.LEFT, padx=2)
        
        profile_frame.columnconfigure(1, weight=1)
        
        # # Arguments section
        # args_frame = ttk.LabelFrame(parent, text="Command Line Arguments")
        # args_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.args_text_obj = TextWidgetWithUndo(parent)#, height=10, wrap=tk.WORD)
        self.args_text = self.args_text_obj.args_text
        # self.args_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # # Add some common argument examples as placeholder
        # self.args_text.insert(tk.END, "# Enter your command line arguments here\n")
        # self.args_text.insert(tk.END, "# Example: --input file.txt --verbose --output results/")
        
        # self.args_text.bind("<Control-z>", self.undo)
        # self.args_text.bind("<Control-y>", self.redo)
        # # For Linux/Windows compatibility (some systems use <Control-Z>)
        # self.args_text.bind("<Control-Z>", self.undo)
        # self.args_text.bind("<Control-Y>", self.redo)
        
        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_process)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_arguments).pack(side=tk.RIGHT, padx=5)
        
        # Output section
        output_frame = ttk.LabelFrame(parent, text="Output")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def build_right_panel(self, parent):

        
        self.drop_zone = tk.Text(parent, height=5, bg="#555555", relief=tk.SUNKEN, 
                                highlightthickness=1, highlightbackground="gray")
        self.drop_zone.pack(fill=tk.X, expand=False, padx=5, pady=5)
        self.drop_zone.insert(tk.END, "Drag and drop files here or click to browse")
        self.drop_zone.config(state=tk.DISABLED)
        
        self.drop_zone.drop_target_register('DND_Files')
        self.drop_zone.dnd_bind('<<Drop>>', self.on_drop)
        
        self.drop_zone.bind("<Button-1>", self.open_file_dialog)
        def clear_dropped_file(a):
            self.dropped_file =None
            self.drop_zone.config(state=tk.NORMAL)
            self.drop_zone.delete(1.0, tk.END)  # Delete all content
            self.drop_zone.insert(1.0, f"Drag and drop files here or click to browse")
            self.drop_zone.config(state=tk.DISABLED)

        self.drop_zone.bind("<Button-3>", clear_dropped_file)
        
        
        # self.file_listbox = tk.Listbox(parent, height=8)
        # self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        self.dropped_file = files[0]
        print("on drop", self.dropped_file)
        self.drop_zone.config(state=tk.NORMAL)
        self.drop_zone.delete(1.0, tk.END)  # Delete all content
        self.drop_zone.insert(tk.END, f"Drag and drop files here or click to browse ({self.dropped_file})")
        self.drop_zone.config(state=tk.DISABLED)
        

        # self.add_files_to_list(files)
        
    def open_file_dialog(self, event=None):
        files = filedialog.askopenfilenames(
            title="Select files",
            filetypes=[("All files", "*.*")]
        )
        print("open file dialog")
        # if files:
        #     self.add_files_to_list(files)
            
    # def add_files_to_list(self, files):
    #     return
    #     self.file_listbox.config(state=tk.NORMAL)
    #     for file in files:
    #         self.file_listbox.insert(tk.END, file)
    #     self.file_listbox.config(state=tk.DISABLED)
        
    # def remove_selected_files(self):
    #     return
    #     selected = self.file_listbox.curselection()
    #     if selected:
    #         self.file_listbox.config(state=tk.NORMAL)
    #         for index in selected[::-1]:  # Reverse to avoid index issues
    #             self.file_listbox.delete(index)
    #         self.file_listbox.config(state=tk.DISABLED)
            
    # def clear_all_files(self):
    #     return
    #     self.file_listbox.config(state=tk.NORMAL)
    #     self.file_listbox.delete(0, tk.END)
    #     self.file_listbox.config(state=tk.DISABLED)
        
    def load_profiles(self):
        # Load saved profiles
        try:
            if os.path.exists("profiles.json"):
                with open("profiles.json", "r") as f:
                    self.profiles = json.load(f)
                self.profile_combo['values'] = list(self.profiles.keys())
                if self.profiles:
                    self.profile_combo.current(0)
                    self.profile_selected()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load profiles: {e}")
            
    def save_profiles(self):
        # Save profiles to file
        try:
            with open("profiles.json", "w") as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profiles: {e}")
            
    def profile_selected(self, event=None):
        # Load selected profile
        profile_name = self.profile_var.get()
        if profile_name in self.profiles:
            self.current_profile = profile_name
            self.args_text.delete(1.0, tk.END)
            self.args_text.insert(tk.END, self.profiles[profile_name].get("arguments", ""))
            
            # files = self.profiles[profile_name].get("files", [])

            
    def save_profile(self):
        # Save current settings as a profile
        profile_name = self.profile_var.get()
        if not profile_name:
            profile_name = simpledialog.askstring("Profile Name", "Enter a name for this profile:")
            if not profile_name:
                return
        
        # return
        # Get all files from listbox
        # files = []
        # self.file_listbox.config(state=tk.NORMAL)
        # for i in range(self.file_listbox.size()):
        #     files.append(self.file_listbox.get(i))
        # self.file_listbox.config(state=tk.DISABLED)
        
        # Save profile
        self.profiles[profile_name] = {
            "arguments": self.args_text.get(1.0, tk.END).strip(),
            # "files": files
        }
        
        # Update combobox
        self.profile_combo['values'] = list(self.profiles.keys())
        self.profile_var.set(profile_name)
        self.current_profile = profile_name
        
        # Save to file
        self.save_profiles()
        messagebox.showinfo("Success", f"Profile '{profile_name}' saved successfully!")
        
    def delete_profile(self):
        # Delete current profile
        profile_name = self.profile_var.get()
        if profile_name and profile_name in self.profiles:
            if messagebox.askyesno("Confirm", f"Are you sure you want to delete profile '{profile_name}'?"):
                del self.profiles[profile_name]
                self.profile_combo['values'] = list(self.profiles.keys())
                if self.profiles:
                    self.profile_combo.current(0)
                    self.profile_selected()
                else:
                    self.profile_var.set("")
                    self.current_profile = None
                    self.args_text.delete(1.0, tk.END)
                self.save_profiles()
                
    def new_profile(self):
        # Create a new empty profile
        self.profile_var.set("")
        self.args_text.delete(1.0, tk.END)
        
    def clear_arguments(self):
        # Clear arguments text area
        self.args_text.delete(1.0, tk.END)
        
    def start_process(self):
        # Start the process with the given arguments
        args = self.args_text.get(1.0, tk.END).strip()
        
        if not args:
            messagebox.showwarning("Warning", "Please enter some arguments first!")
            return
            
        # Disable start button, enable stop button
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Clear output
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Running: {args}\n\n")
        self.output_text.config(state=tk.DISABLED)
        
        # Run process in a separate thread to avoid blocking the GUI
        thread = threading.Thread(target=self.run_process, args=(args,))
        thread.daemon = True
        thread.start()
        
    def run_process(self, args):
        try:
            # args_list = args.split()
            args_list = shlex.split(args)
            if self.dropped_file:
                print("Using dropped file", self.dropped_file)
                assert os.path.isfile(self.dropped_file)
                ind = args_list.index("--input_file")
                args_list[ind+1] = self.dropped_file
            self.process = subprocess.Popen(
                args_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in iter(self.process.stdout.readline, ''):
                self.update_output(line)
                
            # return_code = self.process.wait()
            # self.root.after(0, self.process_completed, return_code)
            
        except Exception as e:
            self.root.after(0, self.process_error, str(e))
        finally:
            if self.process:
                self.process_completed(self.process.wait())
            
    def update_output(self, line):
        # Update output text widget (thread-safe)
        self.root.after(0, self._update_output, line)
        
    def _update_output(self, line):
        # Actually update the output text widget
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, line)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        
    def process_completed(self, return_code):
        # Process completed
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"\nProcess completed with return code: {return_code}\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        
        # Reset buttons
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.process = None
        
    def process_error(self, error_msg):
        # Process error occurred
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"\nError: {error_msg}\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        
        # Reset buttons
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.process = None
        
    def stop_process(self):
        # Stop the running process
        
        if self.process:
            self.process.stdin.write("___exit___\n")
            # self.process.terminate()
            print("waiting...")
            # ret = self.process.wait()
            # self.process_completed(ret)
            print("Process ended")
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, "\nProcess stopped by user\n")
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
            
            # Reset buttons
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.process = None

# For the missing simpledialog reference
from tkinter import simpledialog

if __name__ == "__main__":
    root = TkinterDnD.Tk()  #tk.Tk()
    theme_path = r"F:\all\GitHub\Azure-ttk-theme\azure.tcl"
    if os.path.isfile(theme_path):
        root.call('source', theme_path)
        root.call("set_theme", "dark")
    app = ArgumentGUI(root)
    root.mainloop()