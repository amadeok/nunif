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
import win32gui,win32pipe, win32file
import keyboard

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
        pass
        # print("No .ts files found in the directory.")
        
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



def format_file_size(bytes_size):
    """Format file size in human-readable format"""
    if bytes_size >= 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.2f} MB"
    elif bytes_size >= 1024:
        return f"{bytes_size / 1024:.2f} KB"
    else:
        return f"{bytes_size} bytes"
    
    import subprocess
import json

def get_video_pixel_format_ffprobe(video_path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-select_streams', 'v:0',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            pixel_format = stream.get('pix_fmt', 'Unknown')
            return pixel_format
        else:
            return 'No video stream found'
            
    except subprocess.CalledProcessError as e:
        return f'Error: {e}'
    except FileNotFoundError:
        return 'ffprobe not found. Please install FFmpeg.'

            
def get_pipe_bytes_available(pipe_handles):
    results = []
    
    for handle in pipe_handles:
        try:
            # PeekNamedPipe returns: (error_code, data, bytes_available, bytes_left_this_message)
            result = win32pipe.PeekNamedPipe(handle, 0)  # 0 means don't read any data
            bytes_available = result[1]  # Third element is bytes available
            results.append(bytes_available)
        except Exception as e:
            results.append(-1)
    
    return results


class HLSEncoder:
    def __init__(self, input_f, output_dir, args, ff_hls_time=4, ff_hls_list_size=0):
        self.input_file = input_f

        # self.output_dir =os.path.join(os.path.expandvars("%APPDATA%"), output_dir)# os.path.abspath(output_dir)
        self.output_dir =  os.path.abspath(output_dir)
        self.init_dir()
        print("Outdir", self.output_dir)
        
        cap = cv2.VideoCapture(self.input_file)
        self.video_info = int(cap.get(3)), int(cap.get(4)), cap.get(5)
        self.width, self.height, self.fps = self.video_info
        self.pixel_format = get_video_pixel_format_ffprobe(self.input_file)
        cap.release()
        
        self.ff_hls_time = ff_hls_time
        self.ff_hls_list_size = ff_hls_list_size
        self.encode_queue = queue.Queue(maxsize=5)
        self.decode_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=100)
        self.running = False
        self.audio_out_fut = Future()
        
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
        
        # self.handler.set_click_callback(notepad_hwnd, my_click_callback)
        
        
        self.video_pipe_name = r'\\.\pipe\ffmpeg_video_pipe__'
        self.audio_pipe_name = r'\\.\pipe\ffmpeg_audio_pipe__'
        self.out_audio_pipe_name = r'\\.\pipe\out_ffmpeg_audio_pipe__'
        self.out_video_pipe_name = r'\\.\pipe\out_ffmpeg_video_pipe__'

        self.pipe_size = 1024*1024 * 10
        self.video_pipe = None
        self.audio_pipe = None
        self.out_audio_pipe = None
        self.last_peeked_pipe_sizes ={}
        self.ffmpeg_process = None
        self.pixel_format = get_video_pixel_format_ffprobe(self.input_file)
        #audio
        self.audio_bytes_per_sample_and_channel = None
        self.audio_sample_rate=44100
        self.audio_channels=2
        self.audio_bits_per_sample=16
        bytes_per_sample = self.audio_bits_per_sample // 8
        self.bytes_per_second = self.audio_sample_rate * self.audio_channels * bytes_per_sample
        self.audio_bytes_per_sample_and_channel = self.audio_channels * bytes_per_sample
        ####        
        def on_trigger():
            print(f"{self.get_last_peek_sizes()} | {self.audio_queue.qsize()}  {self.decode_queue.qsize()}, {self.encode_queue.qsize()} ")

        keyboard.add_hotkey('ctrl+shift+a', on_trigger)

        keyboard.add_hotkey('ctrl+alt+1', lambda: print("Ctrl+Alt+1 pressed"))

        
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
    
    def get_last_peek_sizes(self):
        return {e: self.last_peeked_pipe_sizes[getattr(self, e)]/(1000*1000) for e in ["audio_pipe", "video_pipe", "out_audio_pipe", "out_video_pipe"]}
        
    def peek_pipes(self):
        return get_pipe_bytes_available( [self.audio_pipe, self.video_pipe, self.out_audio_pipe])
    def print_peeked_pipes(self):
        results = self.peek_pipes()
        print("results", results, "|",self.decode_queue.qsize(), self.encode_queue.qsize())
            # for handle, bytes_available, success in results:
            
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
        
        self.start_decoder()
        # self.decode_thread = threading.Thread(target=self.decode_vido, daemon=True)
        self.encode_thread = threading.Thread(target=self.encode_to_hls, daemon=True)

        # self.decode_thread.start()
        self.encode_thread.start()
        


    def create_pipes(self):
        """Create Windows named pipes for video and audio."""
        self.video_pipe = win32pipe.CreateNamedPipe(
            self.video_pipe_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            1, self.pipe_size, self.pipe_size, 0, None
        )
        self.audio_pipe = win32pipe.CreateNamedPipe(
            self.audio_pipe_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            2, self.pipe_size, self.pipe_size, 0, None
        )
        self.out_audio_pipe = win32pipe.CreateNamedPipe(
            self.out_audio_pipe_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            2, self.pipe_size, self.pipe_size, 0, None
        )
        
        self.out_video_pipe = win32pipe.CreateNamedPipe(
            self.out_video_pipe_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            2, self.pipe_size, self.pipe_size, 0, None
        )
        self.last_peeked_pipe_sizes = {self.out_audio_pipe:0,self.audio_pipe:0,self.video_pipe:0,self.out_video_pipe:0 }
        
        


    def spawn_ffmpeg(self):
        """Spawn FFmpeg process to write to named pipes."""
        use_mpv = 0
        binary =  "mpv_.com"

        ffmpeg_cmd = ["ffmpeg"]
        #['start', 'cmd', '/k',"ffmpeg"]
        if use_mpv:

            mpv_command = [
                # 'start', 'cmd', '/k',
                f"{binary}",
                # f'{self.input_file}',
                "D:/soft/media/[Judas] Boku no Hero Academia - S07E07.mkv",
                "--ovc=rawvideo",  
                "--oac=pcm_s16le", 
                # f"--vf=format=fmt={self.pixel_format}",#,fps={self.fps}",
                f"--vf=format=fmt=rgb24",#,fps={self.fps}",
                "--of=avi",     # Set output format to raw video
                "-o", "-",
                f"--input-ipc-server={self.mpv_ipc_control_pipe}",
                 "--msg-level=all=warn"
            ]
            # cmd = 'start cmd /k mpv_.com "D:/soft/media/[Judas] Boku no Hero Academia - S07E07.mkv" --ovc=rawvideo '
            # cmd+= '--oac=pcm_s16le --vf=format=fmt=rgb24 --of=avi -o - --input-ipc-server=\\\\.\\pipe\\mpv_iw3_hls_pipe'

            # cmd = " ".join(cmd)
            ffmpeg_cmd += [
            '-f', 'avi', 
            # '-pixel_format', 'rgb24', '-video_size', f'{self.width}x{self.height}',
            # '-framerate', str(self.fps),
            '-i', "-"]
                        
            source_process = subprocess.Popen(mpv_command, stdout=subprocess.PIPE)#, shell=True)
            # time.sleep(5)
        else:
            ffmpeg_cmd += ["-i", self.input_file]
            
        codec_map = {
            16: 'pcm_s16le',
            24: 'pcm_s24le', 
            32: 'pcm_s32le'
        }


        ffmpeg_cmd += [
            #  "-r", str(self.fps),
            # "test.mp4", "-y"
            '-f', 'rawvideo',
            "-bufsize", "10M",
            '-pix_fmt', 'rgb24', "-r", str(self.video_info[1]),
            self.video_pipe_name,
            "-y",
            "-bufsize", "10M",
            '-acodec', codec_map[self.audio_bits_per_sample],
            '-ac', str(self.audio_channels),
            '-ar', str(self.audio_sample_rate),
            '-f', 's16le',
            self.audio_pipe_name,
            "-y",
             "-loglevel",  "error"
        ]
        
        
        self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=source_process.stdout if use_mpv else None)#, shell=True)
                                            #    stdout=subprocess.PIPE, #    stderr=subprocess.PIPE


    def read_peek(self, handle, read_size=1024):
        result = win32pipe.PeekNamedPipe(handle, 0)  # 0 means don't read any data
        bytes_available = result[1]   
        self.last_peeked_pipe_sizes[handle]+=bytes_available     
        return win32file.ReadFile(handle, read_size )
    
    def write_peek(self, handle, data):
        ret =  win32file.WriteFile(handle, data)
        self.last_peeked_pipe_sizes[handle]+=ret[1]   
        return ret  

                                

    def read_video_pipe(self):
        """Read video data from pipe and push full frames to queue."""
        import copy
        w, h, fps = self.video_info
        frame_size_bytes = w*h*3
        try:
            while True:
                
                data = b''
                while len(data) < frame_size_bytes:
                    read_size = min(frame_size_bytes - len(data), self.pipe_size)
                    
                    result, chunk = self.read_peek(self.video_pipe,read_size )
                    if not chunk:
                        break
                    data += chunk

                # result, data = win32file.ReadFile(self.video_pipe, frame_size_bytes)
                le = len(data)
                if not ( le  == frame_size_bytes):raise Exception("le  != frame_size_bytes")
                


                framergb = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                
                # if not framergb.flags.writeable:
                #     framergb = copy.deepcopy( framergb)#.copy()  # Create a writable copy

                frame_buffer = torch.from_numpy(framergb)
                
                if self.cuda_stream is not None:
                    with torch.cuda.stream(self.cuda_stream):
                        frame = frame_buffer.to(self.args.device)
                        frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                        self.cuda_stream.synchronize()
                else:
                    frame = frame_buffer.to(self.args.device)
                    frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                    
                self.decode_queue.put(frame)
        except Exception as e:
            print(f"[Video Pipe] Read error: {e}")

    def read_audio_pipe(self):
        """Read audio data from pipe and push to queue."""
        win32pipe.ConnectNamedPipe(self.out_audio_pipe, None)
        print("out_named_pipe_connected")
        self.audio_out_fut.set_result(True)
        try:
            read_size_ = 256*256 #self.bytes_per_second#(256*256*32)
            tot = 0
            while True:
                data = b''
                while len(data) < read_size_:
                    read_size = read_size_ - len(data)
                    result, chunk = self.read_peek(self.audio_pipe, read_size )
                    if not chunk:
                        break
                    data += chunk
                    
                le_ = len(data)
                rem = le_ % self.audio_bytes_per_sample_and_channel
                if rem  != 0:
                    result, chunk = self.read_peek(self.audio_pipe, self.audio_bytes_per_sample_and_channel - rem )
                    data+=chunk
                
                le = len(data)
                if not ( le  % self.audio_bytes_per_sample_and_channel == 0 ):
                    raise Exception("le  % self.audio_bytes_per_sample_and_channel != 0")

                # result, chunk = win32file.ReadFile(self.video_pipe, read_size )
                
                # for x in range(1):
                #     self.write_peek(self.out_audio_pipe, data)
                bytes_written = 0
                while bytes_written < le:
                    remaining_data = data[bytes_written:]
                    result = self.write_peek(self.out_audio_pipe, remaining_data)
                    bytes_written += result[1]  # result[1] contains bytes written
                    
                    if bytes_written < le:
                        time.sleep(0.001)  # Small delay before retry
                assert(bytes_written == le)
                tot+=le
                print("le", tot)
                # self.audio_queue.put(data)
                # if self.audio_queue.qsize():
                    
        except Exception as e:
            print(f"[Audio Pipe] Read error: {e}")

    def start_decoder(self):
        """Start the FFmpeg process and pipe readers."""
        self.create_pipes()
        self.spawn_ffmpeg()

        print("Waiting for FFmpeg to connect to pipes...")
        win32pipe.ConnectNamedPipe(self.video_pipe, None)
        win32pipe.ConnectNamedPipe(self.audio_pipe, None)

        threading.Thread(target=self.read_video_pipe, daemon=True).start()
        threading.Thread(target=self.read_audio_pipe, daemon=True).start()

        self.video_stream_ready.set_result(self.video_info)

    def stop_decoder(self):
        """Stop and clean up resources."""
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
        if self.video_pipe:
            win32file.CloseHandle(self.video_pipe)
        if self.audio_pipe:
            win32file.CloseHandle(self.audio_pipe)
            
    
  

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
        has_audio =  any(stream.type == 'audio' for stream in input_container.streams)
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
        
        ffmpeg_cmd = ["ffmpeg"]

        
        if has_audio:
            ffmpeg_cmd.extend([
                "-probesize", "10KB",  "-rtbufsize", "100MB",
                "-f", "s16le", "-ar", "44100", "-ac", "2",
                '-i', self.out_audio_pipe_name, "-y",  # Use original file for audio
            ])
        else:
            ffmpeg_cmd.extend(['-an'])  # No audio
        
        ffmpeg_cmd.extend(
            [
            # '-y',  # Overwrite output files
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # Match your frame format
            '-s', f'{width}x{height}',
            '-r', str(framerate),
            '-i',  self.out_video_pipe_name, #'-',  # Read from stdin
            # '-y',  # Overwrite output files
        ])
        if has_audio:             
            ffmpeg_cmd.extend(['-map', '1:v',  '-map', '0:a'])

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

        
        ffmpeg_cmd.extend(video_options)
        
        ffmpeg_cmd.append(output_path)

        print(f"\nRunning ffmpeg command: {' '.join(ffmpeg_cmd)}\n")
        

        # time.sleep(1)100 000 000
        try:
            # ffmpeg_cmd = r"ffmpeg  -f s16le -ar 44100 -ac 2 -i \\.\pipe\out_ffmpeg_audio_pipe__  -f rawvideo -pix_fmt bgr24 -s 3840x1080 -r 23.976023976023978 -i -output.mp4".split(" ")

            
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,#shell=True,
                # stdin=subprocess.PIPE,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                bufsize=10*1024*1024  # Large buffer for video frames
            )
            # time.sleep(1)
            win32pipe.ConnectNamedPipe(self.out_video_pipe, None)

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
            # self.audio_out_fut.result()
            print("Encoder started")
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

                        bytes_written = 0
                        total_bytes = len(img_np_bgr)
                        
                        while bytes_written < total_bytes:
                            remaining_data = img_np_bgr[bytes_written:]
                            result = self.write_peek(self.out_video_pipe, remaining_data)
                            bytes_written += result[1] 
                            
                                
                        # continue
                        # try:
                        #     ffmpeg_process.stdin.write(img_np_bgr)
                        #     ffmpeg_process.stdin.flush()  # Ensure data is sent

                        # except BrokenPipeError:
                        #     print("ffmpeg process terminated unexpectedly")
                        #     break
                        
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
        