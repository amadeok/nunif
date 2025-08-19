import gc
import os
import subprocess
import time
import threading
import av
import queue
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Tuple
import cv2
import numpy as np
import torch 
from concurrent.futures import  Future
from performanceTimer import Counter


from torchvision.transforms import (
    functional as TF,
    InterpolationMode)


import re
from pynput import mouse
import win32gui
import win32api
import pygetwindow as gw
import os
import time



def remove_last_segments(playlist_path, segments_to_remove=5):
    with open(playlist_path, 'r') as f:
        lines = f.readlines()
    
    segment_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
    
    if len(segment_lines) <= segments_to_remove:
        print("Warning: Playlist has fewer segments than requested to remove")
        return
    
    last_segment_to_keep = segment_lines[-(segments_to_remove + 1)]
    
    last_index_to_keep = len(lines) - 1 - lines[::-1].index(last_segment_to_keep)
    
    new_lines = lines[:last_index_to_keep + 1]
    
    with open(playlist_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Removed the last {segments_to_remove} segments from the playlist")

class Context:
    def __init__(self):
        
        
        
        self.output_dir = os.path.abspath("hls_output")
        self.init_dir()
        
        self.ff_hls_time = 2
        self.ff_hls_list_size = 2
        
        self.queue_seek = None
        self.seg_delta = None
        self.is_paused = False
        self.seg_delta_pause_thres = 2
        
        # self.start_frames = 24*15
                # Then where you originally had the code, replace it with:
        thread = threading.Thread( target=self.check_segment_delta)
        thread.daemon = True  # Makes the thread exit when main program exits
        thread.start()
        
        self.last_req_seg_n = 0
        self.newest_seg_n = 0
        
        import pygetwindow as gw, math
        def my_click_callback(x, y):
            perc = (x / notepad_hwnd.width)*100
            self.queue_seek =  perc
        
        self.handler = WindowClickHandler()
        
        
        notepad_hwnd :gw.Window = find_window_by_title("Notepad++")
        if not notepad_hwnd:
            print("Notepad++ window not found!")
            exit()
        
        self.handler.set_click_callback(notepad_hwnd, my_click_callback)
        
    def deinit(self):
        self.handler.remove_click_callback()
        print("Listener stopped.")
            

    def init_dir(self):
        import shutil
    
        # Clean up previous output if it exists
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                print("Error cleaning up:", e)
        while os.path.isdir(self.output_dir):#or len(os.listdir(self.output_dir)):
            print("Wait for delete")
            time.sleep(0.01)
        os.makedirs(self.output_dir, exist_ok=True)


    def seek(self):
        print("seek")
        self.queue_seek = 4
        
    def check_segment_delta(self):
        while 1:
            self.newest_seg_n = get_most_recent_seg_n(self.output_dir)

            if self.newest_seg_n is not None and self.last_req_seg_n is not None:
                self.seg_delta = self.newest_seg_n - self.last_req_seg_n
                
                if self.seg_delta >= self.seg_delta_pause_thres:
                    print("--------Pausing", self.seg_delta)
                    self.is_paused = True
                else:
                    print("--------Resuming", self.seg_delta)
                    self.is_paused = False
            time.sleep(1)




class WindowClickHandler:
    def __init__(self):
        self.target_hwnd = None
        self.click_callback = None
        self.mouse_listener = None
    
    def set_click_callback(self, window_hwnd, callback):
        """Set up a click listener for a specific window"""
        self.target_hwnd = window_hwnd
        self.click_callback = callback
        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.mouse_listener.start()
    
    def remove_click_callback(self):
        """Stop the click listener"""
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
    
    def _on_click(self, x, y, button, pressed):
        """Internal handler for mouse clicks"""
        if button == mouse.Button.left and pressed:
            hwnd = win32gui.WindowFromPoint((x, y))
            if hwnd == self.target_hwnd._hWnd or win32gui.IsChild(self.target_hwnd._hWnd, hwnd):
                client_pt = win32gui.ScreenToClient(self.target_hwnd._hWnd, (int(x), int(y)))
                if self.click_callback:
                    self.click_callback(client_pt[0], client_pt[1])
                # print(f"Clicked at client coordinates: {client_pt}")


def find_window_by_title(title_substring):
    """Find a window by title substring"""
    try:
        h = None
        def callback(hwnd, regex):
            nonlocal h
            title = win32gui.GetWindowText(hwnd)
            if regex in title:
                h = hwnd
        win32gui.EnumWindows(callback, "Notepad++")
        for x in range(10):
            if h: break
            time.sleep(0.01)
        hwnd = gw.Window(h)
        
    except Exception as e:
        print("e", e)
        exit()
    
    return hwnd

def get_most_recent_seg(folder_path):
    if not os.path.isdir(folder_path): return None
    try:
        newest_file = max(
            (f for f in os.listdir(folder_path) if f.endswith('.ts')),
            key=lambda f: os.path.getmtime(os.path.join(folder_path, f))
        )
        # print(f"Most recent .ts file: {newest_file}")
        return newest_file
    except ValueError:
        print("No .ts files found in the directory.")
        
def extract_n(file):
            # Using regular expression to extract the number
        match = re.search(r'segment_(\d+)\.ts', file)
        if match:
            number = int(match.group(1))
            # print(f"Extracted number: {number}")
            return number
        else:
            print("No number found in filename", file)
        
def get_most_recent_seg_n(folder_path):

    file = get_most_recent_seg(folder_path)
    if file:
        return extract_n(file)

def seek_by_percentage(container, percentage):
    
    stream = container.streams.video[0]  # or audio[0] for audio
    
    duration_sec = float(stream.duration * stream.time_base)
    target_sec = duration_sec * (percentage / 100.0)
    target_pts = int(target_sec / stream.time_base)
    print(f"target_sec {target_sec}  perc {percentage}  duration_sec {duration_sec}")

    # container.seek(int(target_sec))#, stream=stream)
    container.seek(target_pts, stream=stream)    

class HLSEncoder:
    def __init__(self, input_f, output_dir, args, ff_hls_time=10, ff_hls_list_size=10):
        self.input_file = input_f
        self.output_dir =os.path.join(os.path.expandvars("%APPDATA%"), output_dir)# os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=1)
        print("Outdir", self.output_dir)
        self.ff_hls_time = ff_hls_time
        self.ff_hls_list_size = ff_hls_list_size
        self.encode_queue = queue.Queue(maxsize=10)
        self.decode_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=100)
        self.running = False
        self.ctx = type('Context', (), {
            'is_paused': False,
            'queue_seek': None,
            'start_frames': 0,
            'seg_delta': None,
            'last_req_seg_n': 0
        })()
        self.args = args
        self.video_stream = None
        if args.device.type == "cuda":
            self.cuda_stream = torch.cuda.Stream(device=args.device)
        else:
            self.cuda_stream = None
        self.video_stream_ready = Future()
        self.c = Counter()
        
    def start(self):
        """Start the decode and encode threads"""
        if self.running:
            return
        
        self.running = True
        self.error = None
        
        self.decode_thread = threading.Thread(target=self.decode_vido, daemon=True)
        self.encode_thread = threading.Thread(target=self.encode_to_hls, daemon=True)

        self.decode_thread.start()
        self.encode_thread.start()

    def decode_vido(self):#,  settings):
        """Decode video frames in a separate thread"""
        try:
            input_container = av.open(self.input_file)
            self.video_stream = None
            audio_stream = None
            
            # Find video and audio streams
            for stream in input_container.streams:
                if stream.type == 'video' and self.video_stream is None:
                    self.video_stream = stream
                # elif stream.type == 'audio' and audio_stream is None:
                #     audio_stream = stream
            
            if self.video_stream is None:
                raise ValueError("No video stream found in input file")
            
            self.framerate = self.video_stream.base_rate  # or video_stream.base_rate
            self.video_stream_ready.set_result(True)
            
            
            # Seek if requested
            if self.ctx.queue_seek:                           
                seek_by_percentage(input_container, self.ctx.queue_seek)
                self.ctx.queue_seek = None
            
            # Decode and queue frames
            streams = [s for s in [self.video_stream, audio_stream] if s is not None]
            count = 0
            
            for packet in input_container.demux(streams):
                if not self.running:
                    break
                
                while self.ctx.is_paused and not self.ctx.queue_seek:
                    time.sleep(0.1)
                                # Add a safety check
                if packet is None or packet.size == 0:
                    continue

                if packet.stream.type == "video":


                    
                    if self.ctx.queue_seek:
                        break
                        
                    for frame_ in packet.decode():
                        count+= 1
                        if count % 100 == 0:
                            gc.collect()
                        # if frame.width != width or frame.height != height:
                        #     frame = frame.reformat(width=width, height=height)
                        
                        framergb = frame_.to_ndarray(format='rgb24')

                        # frame = np.ndarray((self.screen_height, self.screen_width, 3),
                        #                 dtype=np.uint8, buffer=self.process_frame_buffer.buf)
                        # deepcopy
                        frame_buffer = torch.from_numpy(framergb)
                        # if frame_buffer is None:
                        #     frame_buffer = frame.clone()
                        #     if torch.cuda.is_available():
                        #         frame_buffer = frame_buffer.pin_memory()
                        # else:
                        #     frame_buffer.copy_(frame)

                        if self.cuda_stream is not None:
                            with torch.cuda.stream(self.cuda_stream):
                                frame = frame_buffer.to(self.args.device)
                                frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
        
                                # if frame.shape[1:] != (self.frame_height, self.frame_width):
                                #     frame = TF.resize(frame, size=(self.frame_height, self.frame_width),
                                #                     interpolation=InterpolationMode.BILINEAR,
                                #                     antialias=True)
                                self.cuda_stream.synchronize()
                        else:
                            frame = frame_buffer.to(self.args.device)
                            frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0

                            # if frame.shape[1:] != (self.frame_height, self.frame_width):
                            #     frame = TF.resize(frame, size=(self.frame_height, self.frame_width),
                            #                     interpolation=InterpolationMode.BILINEAR,
                            #                     antialias=True)
                        
                        self.decode_queue.put(frame)#, timeout=1.0)

                elif packet.stream.type == "audio" and not self.audio_queue.full():
                    for frame in packet.decode():
                        self.audio_queue.put(frame)#, timeout=1.0)
        
            
        except Exception as e:
            print(f"Decoding error: {e}")
        finally:
            # Signal end of streams
            self.decode_queue.put(None)
            if audio_stream:
                self.audio_queue.put(None)
            input_container.close()
    
  

    def encode_to_hls(self, settings=(None, None, 3*1000*1000), serve_while_generating=True):
        """Encode to HLS format with video and audio using separate ffmpeg process"""
        self.video_stream_ready.result()
        print("Encode started")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Start HTTP server in a separate thread if requested
        if serve_while_generating:
            httpd = start_http_server(self.output_dir)
        
        # Check if input has audio
        input_container = av.open(self.input_file)
        has_audio = any(stream.type == 'audio' for stream in input_container.streams)
        input_container.close()

        if self.args.full_sbs:
            frame_width_scale = 2
        elif self.args.rgbd:
            frame_width_scale = 2
        elif self.args.half_rgbd:
            frame_width_scale = 1
        else:
            self.args.half_sbs = True
            frame_width_scale = 1
                
        width, height, bitrate = settings
        width = self.video_stream.width * frame_width_scale
        height = self.video_stream.height
        
        # Build ffmpeg command
        output_path = os.path.join(self.output_dir, 'master.m3u8')
        segment_pattern = os.path.join(self.output_dir, 'segment_%03d.ts')
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # Match your frame format
            '-s', f'{width}x{height}',
            '-r', str(self.framerate),
            '-i', '-',  # Read from stdin
        ]
        
        # Add audio input if available
        if has_audio:
            ffmpeg_cmd.extend([
                '-i', self.input_file,  # Use original file for audio
                '-map', '0:v',  # Video from first input (stdin)
                '-map', '1:a',  # Audio from second input
            ])
        else:
            ffmpeg_cmd.extend(['-an'])  # No audio
        
        # Add encoding options
        use_gpu_encoding = True
        if use_gpu_encoding:
            video_codec = 'h264_nvenc'
            video_options = [
                '-c:v', video_codec,
                '-preset', 'p1',
                '-cq', '23',
                '-rc', 'constqp',
            ]
        else:
            video_codec = 'libx264'
            video_options = [
                '-c:v', video_codec,
                '-preset', 'ultrafast',
                '-crf', '20',
                '-tune', 'zerolatency',
            ]
        
        ffmpeg_cmd.extend(video_options)
        
        # Add audio codec if available
        if has_audio:
            ffmpeg_cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        
        # Add HLS output options
        ffmpeg_cmd.extend([
            '-f', 'hls',
            '-hls_time', str(self.ff_hls_time),
            '-hls_list_size', str(self.ff_hls_list_size),
            '-hls_flags', 'independent_segments+append_list',
            '-hls_segment_type', 'mpegts',
            '-hls_segment_filename', segment_pattern,
            '-hls_playlist_type', 'event',
            output_path
        ])
        
        print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        
        # Start ffmpeg process
        try:
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Large buffer for video frames
            )
            
            # Function to handle ffmpeg output
            def monitor_ffmpeg():
                while self.running and ffmpeg_process.poll() is None:
                    try:
                        line = ffmpeg_process.stderr.readline().decode('utf-8')
                        if line:
                            print(f"ffmpeg: {line.strip()}")
                    except:
                        break
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_ffmpeg)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Encoding loop
            c = self.c
            video_finished = False
            
            while self.running and not video_finished:
                if self.encode_queue.qsize():
                    try:
                        # c.pt()

                        sbs = self.encode_queue.get()
                        sbs = (sbs * 255).to(torch.uint8)
                        
                        img_np = sbs.detach().cpu().numpy()
                        
                        # Permute dimensions from [C, H, W] to [H, W, C]
                        img_np = np.transpose(img_np, (1, 2, 0))
                        
                        # Convert to uint8 if needed
                        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                            img_np_bgr = (img_np * 255).astype(np.uint8)
                        else:
                            img_np_bgr = img_np
                        
                        img_np_bgr = np.ascontiguousarray(img_np_bgr)
                        # c.ct(1)

                        # Write frame to ffmpeg stdin
                        try:
                            ffmpeg_process.stdin.write(img_np_bgr)
                        except BrokenPipeError:
                            print("ffmpeg process terminated unexpectedly")
                            break
                        
                        
                    except queue.Empty:
                        pass
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        break
            
                time.sleep(0.001)
                
                # Handle seek requests
                if self.ctx.queue_seek:
                    print("Seeking requested, restarting encoding")
                    break
            
            # Close stdin to signal end of input
            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
        
            # Wait for ffmpeg to finish
            try:
                ffmpeg_process.wait(timeout=30)
                print("HLS encoding complete.")
            except subprocess.TimeoutExpired:
                print("ffmpeg process timed out, terminating...")
                ffmpeg_process.terminate()
                ffmpeg_process.wait()
            
        except Exception as e:
            print(f"Encoding error: {e}")
            # Clean up ffmpeg process if it exists
            if 'ffmpeg_process' in locals() and ffmpeg_process.poll() is None:
                ffmpeg_process.terminate()
        finally:
            self.running = False
            
            if serve_while_generating:
                # Keep server running for a bit after encoding finishes
                time.sleep(1)
                httpd.shutdown()
            
def start_http_server(output_dir):
    """Start a simple HTTP server to serve HLS files"""
    os.chdir(output_dir)
    server_address = ('0.0.0.0', 8123)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"Serving HLS files from {output_dir} on http://localhost:8000")
    return httpd



if __name__ == "__main__":
    def my_click_callback(x, y):
        perc = (x / notepad_hwnd.width)*100
        print(f"Custom callback: Clicked at ({x}, {y}) {perc}")
    
    handler = WindowClickHandler()
    
    notepad_hwnd :gw.Window = find_window_by_title("Notepad++")
    if not notepad_hwnd:
        print("Notepad++ window not found!")
        exit()
    
    handler.set_click_callback(notepad_hwnd, my_click_callback)
    
    try:
        print("Listening for clicks in Notepad++... Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handler.remove_click_callback()
        print("Listener stopped.")
        