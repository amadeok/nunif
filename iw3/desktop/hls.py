
import os
import time
from .utils import (
    init_win32,
    set_state_args,
    iw3_desktop_main_hls
)

from os import path
from packaging.version import Version
import torch
from torchvision.io import encode_jpeg
from .. import utils as IW3U

from nunif.device import create_device
from nunif.models import compile_model
from nunif.models.data_parallel import DeviceSwitchInference
from nunif.initializer import gc_collect
from .decode_encode import HLSEncoder
from performanceTimer import Counter

TORCH_VERSION = Version(torch.__version__)
ENABLE_GPU_JPEG = (TORCH_VERSION.major, TORCH_VERSION.minor) >= (2, 7)
TORCH_NUM_THREADS = torch.get_num_threads()

from .utils import init_num_threads, get_local_address, is_private_address


def iw3_desktop_main_hls(args):
    if not path.isfile(args.input_file):
        print("File not found", args.input_file)
        exit()
    init_num_threads(args.gpu[0])
    c = Counter()


    # if args.bind_addr is None:
    #     args.bind_addr = get_local_address()
    # if args.bind_addr == "0.0.0.0":
    #     pass  # Allows specifying undefined addresses
    # elif args.bind_addr == "127.0.0.1" or not is_private_address(args.bind_addr):
    #     raise RuntimeError(f"Detected IP address({args.bind_addr}) is not Local Area Network Address."
    #                        " Specify --bind-addr option")

    args.device = create_device(args.gpu)

    depth_model = args.state["depth_model"]
    print("Model: ", depth_model.model_type)
    if not depth_model.loaded():
        depth_model.load(gpu=args.gpu, resolution=args.resolution)

    # Use Flicker Reduction to prevent 3D sickness
    depth_model.enable_ema(args.ema_decay, buffer_size=1)
    args.mapper = IW3U.resolve_mapper_name(mapper=args.mapper, foreground_scale=args.foreground_scale,
                                           metric_depth=depth_model.is_metric())

    # TODO: For mlbw, it is better to switch models when the divergence value dynamically changes
    side_model = IW3U.create_stereo_model(
        args.method,
        divergence=args.divergence * (2.0 if args.synthetic_view in {"right", "left"} else 1.0),
        device_id=args.gpu[0],
    )

    vp = HLSEncoder(args.input_file, args.segment_folder, args=args)
    vp.start()
    
    try:
        if args.compile:
            depth_model.compile()
            if side_model is not None and not isinstance(side_model, DeviceSwitchInference):
                side_model = compile_model(side_model)
        count = 0

        output_queue = vp.interpolate_input_queue if vp.interpolation_multiplier > 1 else vp.encode_video_queue
        while True:
            with args.state["args_lock"]:
                

                frame =  vp.decode_video_queue.get()

                if type(frame) != torch.Tensor and not frame:
                    print("Decode terminated")
                    break
                # c.pt()

                sbs = IW3U.process_image(frame, args, depth_model, side_model)
                # c.ct(1)
                if vp.b_print_debug:vp.print_debug()

                output_queue.put(sbs)

                # time.sleep(0.001)

                if count % (30 * vp.video_info[2]) == 0:
                    gc_collect()
                    
            count += 1
            if args.state["stop_event"] and args.state["stop_event"].is_set():
                break
    finally:
        # server.stop()
        # screenshot_thread.stop()
        depth_model.clear_compiled_model()
        depth_model.reset()
        gc_collect()

    if args.state["stop_event"] and args.state["stop_event"].is_set():
        args.state["stop_event"].clear()

    return args

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import argparse
import threading
import time

class SeekBarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Seek Bar with Argument Parser")
        self.root.geometry("800x500")
        self.root.minsize(700, 400)
        
        # Variables
        self.is_processing = False
        self.current_value = tk.DoubleVar(value=50.0)  # Default value for seek bar
        self.min_value = 0.0
        self.max_value = 100.0
        
        # Setup the GUI
        self.setup_gui()
        
        # Set default arguments in the text field
        self.set_default_arguments()
        
    def setup_gui(self):
        # Create main frames
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Argument input section
        ttk.Label(main_frame, text="Command Line Arguments:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.argument_text = scrolledtext.ScrolledText(main_frame, height=5, wrap=tk.WORD)
        self.argument_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Seek bar section
        ttk.Label(main_frame, text="Seek Bar Value:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky=tk.W, pady=(10, 5))
        
        # Current value display
        self.value_label = ttk.Label(main_frame, text=f"{self.current_value.get():.1f}")
        self.value_label.grid(row=2, column=1, sticky=tk.E, pady=(10, 5))
        
        # Seek bar
        self.seek_bar = ttk.Scale(
            main_frame, 
            from_=self.min_value, 
            to=self.max_value,
            variable=self.current_value,
            orient=tk.HORIZONTAL,
            command=self.on_seek_bar_change
        )
        self.seek_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Min/Max labels
        min_max_frame = ttk.Frame(main_frame)
        min_max_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        ttk.Label(min_max_frame, text=f"Min: {self.min_value}").pack(side=tk.LEFT)
        ttk.Label(min_max_frame, text=f"Max: {self.max_value}").pack(side=tk.RIGHT)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        self.start_button = ttk.Button(
            button_frame, 
            text="Start Processing", 
            command=self.start_processing
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(
            button_frame, 
            text="Stop Processing", 
            command=self.stop_processing,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT)
        
        # Status section
        ttk.Label(main_frame, text="Status:", font=('Arial', 10, 'bold')).grid(
            row=6, column=0, sticky=tk.W, pady=(10, 5))
        
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=6, column=1, sticky=tk.W, pady=(10, 5))
        
        # Configure row and column weights for resizing
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def set_default_arguments(self):
        """Set some default arguments in the text field"""
        default_args = "--input sample.txt --output result.txt --verbose --count 5 --delay 0.5"
        self.argument_text.delete(1.0, tk.END)
        self.argument_text.insert(tk.END, default_args)
        
    def on_seek_bar_change(self, value):
        """Callback for when the seek bar value changes"""
        current_val = float(value)
        self.value_label.config(text=f"{current_val:.1f}")
        
        # You can add additional functionality here that responds to seek bar changes
        # For example: update a visualization, modify parameters, etc.
        
    def parse_arguments(self):
        """Parse the arguments from the text field using argparse"""
        arg_text = self.argument_text.get(1.0, tk.END).strip()
        
        if not arg_text:
            messagebox.showwarning("Warning", "Please enter some arguments")
            return None
        
        parser = argparse.ArgumentParser(description="GUI Argument Parser")
        
        # Define the arguments your application supports
        parser.add_argument('--input', '-i', type=str, help='Input file path')
        parser.add_argument('--output', '-o', type=str, help='Output file path')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')
        parser.add_argument('--count', '-c', type=int, default=10, help='Number of iterations')
        parser.add_argument('--delay', '-d', type=float, default=0.1, help='Delay between iterations')
        
        try:
            # Split the argument string and parse
            args = arg_text.split()
            parsed_args = parser.parse_args(args)
            return parsed_args
        except SystemExit:
            # argparse calls sys.exit() on error, we catch it here
            messagebox.showerror("Error", "Invalid arguments provided")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse arguments: {str(e)}")
            return None
    
    def start_processing(self):
        """Start the processing with the provided arguments"""
        if self.is_processing:
            return
            
        # Parse arguments
        parsed_args = self.parse_arguments()
        if parsed_args is None:
            return
        
        # Get the current seek bar value
        seek_value = self.current_value.get()
        
        # Update UI
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Processing with seek value: {seek_value:.1f}")
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self.process_data,
            args=(parsed_args, seek_value),
            daemon=True
        )
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the processing"""
        self.is_processing = False
        self.status_label.config(text="Stopping...")
    
    def process_data(self, args, seek_value):
        """Main processing function that uses the parsed arguments and seek value"""
        try:
            # Display the parsed arguments
            status_text = (f"Processing: input={args.input}, output={args.output}, "
                          f"count={args.count}, delay={args.delay}, verbose={args.verbose}, "
                          f"seek_value={seek_value:.1f}")
            
            self.root.after(0, lambda: self.status_label.config(text=status_text))
            
            # Simulate processing
            for i in range(args.count):
                if not self.is_processing:
                    break
                
                # Simulate some work based on the seek value
                time.sleep(args.delay)
                
                # Update status
                progress = (i + 1) / args.count * 100
                current_status = f"Processing item {i+1}/{args.count} ({(progress):.1f}%)"
                self.root.after(0, lambda s=current_status: self.status_label.config(text=s))
            
            # Final update
            self.root.after(0, self.processing_complete)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing error: {str(e)}"))
            self.root.after(0, self.processing_complete)
    
    def processing_complete(self):
        """Clean up after processing completes"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Ready")

def main():
    root = tk.Tk()
    theme_path= r"F:\all\GitHub\Azure-ttk-theme\azure.tcl"
    if os.path.isfile(theme_path):
        root.tk.call('source', theme_path)
        root.tk.call("set_theme", "dark")
    app = SeekBarApp(root)
    
    root.mainloop()

# if __name__ == "__main__":
#     main()

# from gooey import Gooey
# import argparse

# @Gooey  # ← Just add this decorator
def create_parser():

    parser = IW3U.create_parser(required_true=False)
    parser.add_argument("--port", type=int, default=8123,
                        help="HTTP listen port")

    parser.add_argument("--full-sbs", action="store_true", help="Use Full SBS for Pico4")
    parser.add_argument("--input_file", type=str, help="input_file")
    parser.add_argument("--segment_folder", type=str, help="output for the video segment files ", default="hls_out")
    parser.add_argument("--nvenc-preset", type=str, help="nvenc preset", default="p1")
    parser.add_argument("--cli-mode", type=int, help="cli mode", default=False)


    parser.set_defaults(
        input="dummy",
        output="dummy",
        depth_model="Any_V2_S",
        divergence=1.0,
        convergence=1.0,
        ema_normalize=True,
    )
    return parser


def cli_main():
    import sys
    init_win32()
    if  "cli-mode" in sys.argv or 1:        
        parser = create_parser()
        args = parser.parse_args()
        set_state_args(args)
        iw3_desktop_main_hls(args)    

if __name__ == "__main__":
    cli_main()
