import gc
import json
import os
import subprocess
import time
import threading
import av
import queue
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Literal, Tuple
import cv2
import numpy as np
import torch 
from concurrent.futures import  Future
from performanceTimer import Counter
import pygetwindow as gw, math

from torchvision.transforms import (
    functional as TF,
    InterpolationMode)

import re
from pynput import mouse
import win32gui



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

        
import win32file

def pause_unpause(op: Literal['toggle', 'pause', 'unpause'], pipe_path=None):
    if op== "toggle":
        command = {  "command": ["cycle", "pause" ]  } 
    elif op == "pause":
        command = {  "command": ["set_property", "pause", True]  } 
    elif op== "unpause":
        command = {  "command": ["set_property", "pause", False]  } 
    else:assert(0)
    res = send_cmd(proc_cmd(command), pipe_path)
    # print(res)
    return res

def create_pipe_read_write(pipe_path=None):
    return create_pipe(pipe_path, win32file.GENERIC_WRITE | win32file.GENERIC_READ)

def create_pipe(pipe_path=None, flags=win32file.GENERIC_WRITE):
    global mpv_ipc_current_pipe_index
    try:
        handle = win32file.CreateFile(pipe_path,flags,0,None,win32file.OPEN_EXISTING,0,None)
    except Exception as e:
        print(f"error creating first pipe({pipe_path})", e)

    return handle

def proc_cmd(cmd): return json.dumps(cmd) + '\n'

def seek_absolute_perc(perc, pipe_path=None):
    command = { "command": ["seek", perc, "absolute-percent", "exact"] }
    res = send_cmd(proc_cmd(command), pipe_path)
    if res:
        print(res)
    # print_playback_time(pipe_path)
    return res

def send_cmd(cmd, pipe_path=None, read_response=False):
    handle = create_pipe_read_write(pipe_path)
    response = None
    if handle:
        win32file.WriteFile(handle, cmd.encode())
        if read_response:
            result, data_recv = win32file.ReadFile(handle, 4096) 
            lines = data_recv.splitlines()
            if len(lines) > 1:
                data_ = lines[0]
            else: data_ = data_recv
            response = json.loads(data_.decode())
        else:
            response = None
        win32file.CloseHandle(handle)
    else: return None
    return response



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

class HLSEncoder:
    def __init__(self, input_f, output_dir, args, ff_hls_time=4, ff_hls_list_size=0):
        self.input_file = input_f

        # self.output_dir =os.path.join(os.path.expandvars("%APPDATA%"), output_dir)# os.path.abspath(output_dir)
        self.output_dir =  os.path.abspath(output_dir)
        self.init_dir()
        print("Outdir", self.output_dir)
        self.video_info = [0, 0, 0]
        self.ff_hls_time = ff_hls_time
        self.ff_hls_list_size = ff_hls_list_size
        self.encode_queue = queue.Queue(maxsize=10)
        self.decode_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=100)
        self.running = False

        self.args = args
        self.video_stream = None
        if args.device.type == "cuda":
            self.cuda_stream = torch.cuda.Stream(device=args.device)
        else:
            self.cuda_stream = None
        self.video_stream_ready = Future()
        self.c = Counter()
        self.mpv_ipc_control_pipe = r"\\.\pipe\mpv_iw3_hls_pipe"
        ##
        self.seg_delta = None
        self.is_paused = False
        self.seg_delta_pause_thres = 2

        thread = threading.Thread( target=self.check_segment_delta)
        thread.daemon = True  # Makes the thread exit when main program exits
        thread.start()
        
        self.last_req_seg_n = 0
        self.newest_seg_n = 0
        
        def my_click_callback(x, y):
            perc = (x / notepad_hwnd.width)*100
            # self.queue_seek =  perc
            seek_absolute_perc(perc, self.mpv_ipc_control_pipe)
        
        self.handler = WindowClickHandler()
        
        notepad_hwnd :gw.Window = find_window_by_title("Notepad++")
        if not notepad_hwnd:
            print("Notepad++ window not found!")
            exit()
        
        self.handler.set_click_callback(notepad_hwnd, my_click_callback)
        
    def seek(self):
        print("seek")
        self.queue_seek = 4
        
    def check_segment_delta(self):
        while 1:
            self.newest_seg_n = get_most_recent_seg_n(self.output_dir)

            if self.newest_seg_n is not None and self.last_req_seg_n is not None:
                self.seg_delta = self.newest_seg_n - self.last_req_seg_n
                
                if self.seg_delta >= self.seg_delta_pause_thres:
                    print("- Pausing", self.seg_delta, " -")
                    self.is_paused = True
                    pause_unpause('pause', self.mpv_ipc_control_pipe)
                else:
                    print("- Resuming", self.seg_delta, " -")
                    self.is_paused = False
                    pause_unpause('unpause', self.mpv_ipc_control_pipe)
            time.sleep(1)
            
    def deinit(self):
        self.handler.remove_click_callback()
        print("Listener stopped.")

    def init_dir(self):
        import shutil

        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                print("Error cleaning up:", e)
        while os.path.isdir(self.output_dir):#or len(os.listdir(self.output_dir)):
            print("Wait for delete")
            time.sleep(0.01)
        os.makedirs(self.output_dir, exist_ok=True)

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

    def decode_vido(self):
        binary =  os.path.expanduser("~") +  r"\rifef _\mpv-x86_64\mpv_.com"

        cap = cv2.VideoCapture(self.input_file)
        self.video_info = int(cap.get(3)), int(cap.get(4)), cap.get(5)
        w, h, fps = self.video_info
        cap.release()
        
        self.video_stream_ready.set_result([w, h, fps])

        # ./mpv_.com --vf=format=fmt=help
        mpv_command = [
            binary,
            self.input_file,
            "--ovc=rawvideo",  
            "--vf=format=fmt=rgb24",
            "--of=rawvideo",     # Set output format to raw video
            "-o", "-",
            f"--input-ipc-server={self.mpv_ipc_control_pipe}",
            # "--msg-level=all=warn"
        ]
        c = self.c
        try:
            process = subprocess.Popen(
                mpv_command,
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                # bufsize=0  # Unbuffered
            )
            print("MPV process spawned successfully.")

            frame_size_bytes = w * h * 3  # 3 bytes per pixel for RGB24
                        
            while True:

                # c.pt()

                data = b''
                while len(data) < frame_size_bytes:
                    chunk = process.stdout.read(frame_size_bytes - len(data))
                    if not chunk:
                        break
                    data += chunk
                assert len(data) == frame_size_bytes
                # c.ct(1)
                framergb = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                frame_buffer = torch.from_numpy(framergb)

                if self.cuda_stream is not None:
                    with torch.cuda.stream(self.cuda_stream):
                        frame = frame_buffer.to(self.args.device)
                        frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                        self.cuda_stream.synchronize()
                else:
                    frame = frame_buffer.to(self.args.device)
                    frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                    
                self.decode_queue.put(frame)#, timeout=1.0)

                # cv2.imshow("test",framergb)
                # cv2.waitKey(1)

                
        except FileNotFoundError:
            print("Error: MPV not found. Please ensure it's installed and in your system's PATH.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.decode_queue.put(None)
            if 'process' in locals() and process.poll() is None:
                process.terminate()
            print("MPV process terminated.")
    
  

    def encode_to_hls(self, bitrate= 3*1000*1000, serve_while_generating=True):
        """Encode to HLS format with video and audio using separate ffmpeg process"""
        width, height, framerate =  self.video_stream_ready.result()
        print("Encode started", width, height, framerate)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Start HTTP server in a separate thread if requested
        if serve_while_generating:
            httpd = start_http_server(self.output_dir, self)
        
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
                
        width = width * frame_width_scale
        
        # Build ffmpeg command
        output_path = os.path.join(self.output_dir, 'master.m3u8')
        segment_pattern = os.path.join(self.output_dir, 'segment_%03d.ts')
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # Match your frame format
            '-s', f'{width}x{height}',
            '-r', str(framerate),
            '-i', '-',  # Read from stdin
        ]
        
        if has_audio:
            ffmpeg_cmd.extend([
                '-i', self.input_file,  # Use original file for audio
                '-map', '0:v',  # Video from first input (stdin)
                '-map', '1:a',  # Audio from second input
            ])
        else:
            ffmpeg_cmd.extend(['-an'])  # No audio
        

        ffmpeg_cmd.extend([
            '-f', 'hls',
            '-hls_time', str(self.ff_hls_time),
            '-hls_list_size', str(self.ff_hls_list_size),
            '-hls_flags', 'independent_segments+append_list',
            '-hls_segment_type', 'mpegts',
            '-hls_segment_filename', segment_pattern,
            '-hls_playlist_type', 'event',
            # output_path
        ])
        
        
        if self.args.gpu_encoding:
            video_codec = 'h264_nvenc'
            video_options = [
                '-c:v', video_codec,
                '-preset', 'p1',
                '-cq', '23',
                '-rc', 'constqp',
                "-g", "144",
            ]
        else:
            video_codec = 'libx264'
            video_options = [
                '-c:v', video_codec,
                '-preset', 'ultrafast',
                '-crf', '20',
                '-tune', 'zerolatency',
            ]
                # Add HLS output options

        
        ffmpeg_cmd.extend(video_options)
        
        # Add audio codec if available
        if has_audio:
            ffmpeg_cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        
        ffmpeg_cmd.append(output_path)

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

            
            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
        
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
            
class LoggingHTTPRequestHandler(SimpleHTTPRequestHandler):
    # def __init__(self, request, client_address, server, *, directory = None):
    #     super().__init__(request, client_address, server, directory=directory, vp=None)
        
    def do_GET(self):
        print(f"Requested file: {self.path}")  # Log the requested path
        if self.path.endswith(".ts") and hasattr(self, "vp"):
            self.vp.last_req_seg_n = extract_n( self.path) or 0
            
        super().do_GET()

def start_http_server(output_dir, vp):
    # Change to the output directory so the server serves from there
    os.chdir(output_dir)
    
    # Start HTTP server in a separate thread
    server_address = ('0.0.0.0', vp.args.port)
    
    handler = LoggingHTTPRequestHandler
    handler.vp = vp  

    httpd = HTTPServer(server_address, handler)  # Use our custom handler
    
    
    def run_server():
        print(f"Serving HLS stream at http://localhost:{vp.args.port}/playlist.m3u8")
        httpd.serve_forever()
    
    thread = threading.Thread(target=run_server)
    thread.daemon = True
    thread.start()
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
        