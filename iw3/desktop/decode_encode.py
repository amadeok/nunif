import copy
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
print("torch", torch.__version__)
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
        return handle

    except Exception as e:
        print(f"error creating first pipe({pipe_path})", e)


def proc_cmd(cmd): return json.dumps(cmd) + '\n'

def seek_absolute_perc(perc, pipe_path=None):
    command = { "command": ["seek", perc, "absolute-percent", "exact"] }
    res = send_cmd(proc_cmd(command), pipe_path)
    if res:
        print(res)
    # print_playback_time(pipe_path)
    return res

def send_cmd(cmd, pipe_path=None, read_response=False):
    if type(cmd) == dict: cmd = proc_cmd(cmd)
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
        # else:
        #     print("No number found in filename", file)
        
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
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = frame_count / self.fps

        # self.pixel_format = get_video_pixel_format_ffprobe(self.input_file)
        cap.release()
        self.keyframes = self.get_keyframes()
        self.ff_hls_time = ff_hls_time
        self.ff_hls_list_size = ff_hls_list_size
        self.encode_queue = queue.Queue(maxsize=10)
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
        # self.mpv_ipc_control_pipe = r"\\.\pipe\mpv_iw3_hls_pipe"
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

        self.pipe_size = 1024*1024 * 10

        self.ffmpeg_process = None
        # self.pixel_format = get_video_pixel_format_ffprobe(self.input_file)

        def on_trigger():
            print(f" --- {self.audio_queue.qsize()}  {self.decode_queue.qsize()}, {self.encode_queue.qsize()} ---")

        # keyboard.add_hotkey('ctrl+shift+a', on_trigger)

        # keyboard.add_hotkey('ctrl+alt+shift+q', lambda: self.stop())
        def seek():
            if not self.keyframes:print("No keyframes")
            print("------Input percentage")
            perc = input()
            print("perc", perc)
            try:
                perc = int(perc)
            except Exception as e:print("Nan", perc)
            self.seek_restart_perc(perc)
        # keyboard.add_hotkey('ctrl+alt+shift+f10', lambda: seek())
        from global_hotkeys import register_hotkeys, start_checking_hotkeys

        bindings = [["window + f10", None, seek, False],
                    ["window + f11", None, on_trigger, False],
                    ["window + f12", None, self.stop, False]
                    ]
        register_hotkeys(bindings)
        start_checking_hotkeys()

        
        self.decode_mpv_pipe_name = r'\\.\pipe\decode_mpv_pipe__'
        self.encode_mpv_pipe_name = r'\\.\pipe\encode_mpv_pipe__'

        self.cur_audio_file = "cur_audio.mp4"
        self.whole_audio_file = "audio.mp4"
        self.target_time = 0
        self.stop_flag = False
        self.decoder_thread = threading.Thread(target=lambda: 1)
        self.encoder_thread = threading.Thread(target=lambda: 1)
        self.encode_process = None
        self.decode_process = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        httpd = start_http_server(self.output_dir, self)
            

        
    def check_segment_delta(self):
        while 1:
            self.newest_seg_n = get_most_recent_seg_n(self.output_dir)

            if self.newest_seg_n is not None and self.last_req_seg_n is not None:
                self.seg_delta = self.newest_seg_n - self.last_req_seg_n
                
                if self.seg_delta >= self.seg_delta_pause_thres:
                    print("- Pausing", self.seg_delta, " -")
                    self.is_paused = True
                    pause_unpause('pause', self.encode_mpv_pipe_name)
                else:
                    print("- Resuming", self.seg_delta, " -")
                    self.is_paused = False
                    pause_unpause('unpause', self.encode_mpv_pipe_name)
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

    def restart(self):
        # self.stop_flag  = True
        self.stop()
        
        self.decoder_thread = threading.Thread(target=self.decoder)
        self.decoder_thread.start()
        self.encoder_thread = threading.Thread(target=self.encoder)
        self.encoder_thread.start()      


    def quit_decode_mpv(self):
        res = send_cmd({  "command": ["quit" ]  }, self.decode_mpv_pipe_name )
    def quit_encode_mpv(self):
        res = send_cmd({  "command": ["quit" ]  }, self.encode_mpv_pipe_name )
    
    def stop(self):
        def clear_queue(q: queue.Queue):
            while q.qsize():
                q.get_nowait()
                
        if self.encode_process:
            # self.encode_process.stdin.close()
            self.quit_encode_mpv()
        if self.decode_process:
            # self.decode_process.stdout.close()
            self.quit_decode_mpv()
            # self.video_queue.put(None)
        time.sleep(1)
        
        if self.decoder_thread.is_alive():
            clear_queue(self.decode_queue)
            # self.decode_process.stdout.close()
            self.decoder_thread.join()
        if self.encoder_thread.is_alive():
            self.encoder_thread.join()
        self.stop_flag  = False
        clear_queue(self.decode_queue)
        clear_queue(self.encode_queue)

            
    def extract_audio(self, start_time_seconds, out_file):
        cmd = ["ffmpeg", "-ss", str(start_time_seconds), "-i", self.input_file, "-vn", "-c:a", "copy", "-f", "mp4", out_file, "-y"]
        print("Getting audio")
        ret = subprocess.call(cmd)
        print(F"Got audio {out_file} at {start_time_seconds}")
    
    def get_all_audio(self):
        self.extract_audio(0, self.whole_audio_file)
        
    def get_audio_from_sec(self, start_second):
        self.extract_audio(start_second, self.cur_audio_file)
        
    def get_keyframes(self):
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'packet=pts_time,flags',
            '-of', 'csv=p=0',
            self.input_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        keyframes = []
        
        for line in result.stdout.strip().split('\n'):
            if line and 'K' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        timestamp = float(parts[0])
                        keyframes.append(timestamp)
                    except ValueError:
                        continue
        
        self.keyframes = sorted(set(keyframes))
        return self.keyframes
    
    # def seek_restart(self,time ):
    #     self.target_time = time
    #     self.restart()
        
    def seek_restart_perc(self,perc ):
        def get_element_at_percentage(percentage, lst):
            factor = len(lst) * percentage / 100
            index = round(factor)
            index = max(0, min(index, len(lst) - 1))
            return lst[index], index#, factor, index
        
        self.target_time, index = get_element_at_percentage(perc, self.keyframes)
        print("%", perc,  "target_time", self.target_time, "index", index,)

        self.restart()
        
    def seek_restart_keyframe_index(self, keyframe_index):
        if keyframe_index < 0 or keyframe_index >= len(self.keyframes):
            print(f"Invalid keyframe index: {keyframe_index}")
            return False
        self.target_time = self.keyframes[keyframe_index]
        self.restart()
        
    def seek_to_keyframe(self, keyframe_index):
        
        if keyframe_index < 0 or keyframe_index >= len(self.keyframes):
            print(f"Invalid keyframe index: {keyframe_index}")
            return False
        
        self.target_time = self.keyframes[keyframe_index]

        command = { "command": ["seek", self.target_time, "absolute"] }
        
        response = self.send_command(command)
        # self.get_audio_from_sec(target_time)

        if response and response.get("error") == "success":
            print(f"Seeked to keyframe {keyframe_index} at {self.target_time:.2f}s")
            return True
        return False
        
    def decoder(self):
        print("Decoder started")
        w, h, fps = self.video_info

        try:
            cmd = [
                'mpv_.com',
                # "--hr-seek=no",
                self.input_file,
                "--ovc=rawvideo",  
                f"--vf=format=fmt=rgb24",#,fps={self.fps}",
                "--of=rawvideo",     # Set output format to raw video
                '--input-ipc-server=' + self.decode_mpv_pipe_name,
                "-o", "-",
                f"--start={self.target_time}", #"--pause=yes",
                 "--msg-level=all=warn"
            ]
            bufsize = (w*h*3)
            self.decode_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                # stdin=subprocess.PIPE,
                # text=True,
                 bufsize=bufsize
            )

                
            data = b''
            frame_size_bytes = w*h*3
            pipe_size = bufsize
            while not self.stop_flag:
                data = b''
                while len(data) < frame_size_bytes and not self.stop_flag:
                    read_size = min(frame_size_bytes - len(data), pipe_size)
                    
                    chunk = self.decode_process.stdout.read(read_size)
                    if not chunk:
                        break
                    data += chunk
                assert(len(data) == frame_size_bytes)
                # self.decode_queue.put(data)
                # self.proc_decoded_frame_and_queue(w,h,data)
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
                
                time.sleep(0.001)

                # print("tick", self.decode_queue.qsize())
            print("Decoder ended")
        except Exception as e:
            print("Error at decoder", e)
        # threading.Thread(target=loop, daemon=True).start()
            

    def proc_decoded_frame_and_queue(self, w, h, data):
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
    
  
    def encoder(self):
        """Async function that encodes frames using FFmpeg subprocess"""
        print("Encoder started")
        
        width, height, framerate =  self.video_info#self.video_stream_ready.result()
        print("Encode started", width, height, framerate)        

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
        
        playlist_path = os.path.join(self.output_dir, 'master.m3u8')
        segment_pattern_path = os.path.join(self.output_dir, 'segment_%03d.ts')
        # segment_pattern_path = os.path.join(self.output_dir, 'segment_.ts')

        try:
   
            self.get_audio_from_sec(self.target_time)
            codec = self.args.video_codec
            
            output_hls = True
            
            ofopts = f'movflags=+frag_keyframe+empty_moov'
            if not "nvenc" in codec:
                ovcopts=f"preset={ self.args.preset},b={self.args.video_bitrate}"
                # ofopts = ofopts
            else:
                ovcopts= ",".join([
                    f"preset={ self.args.nvenc_preset}",
                    "rc=vbr",
                    "cq=18",           # Direct quality control (18 is very high quality)
                    "b=0",           # Required for -cq mode
                    #"maxrate=10M",     # Optional safety net: max bitrate
                    "bufsize=20M",     # Buffer size (2x maxrate)
                    "profile=high"
                ])
                ofopts += ","+ ",".join([
                        # '-f', 'hls',
                        f'hls_time={self.ff_hls_time}',
                        f'hls_list_size={self.ff_hls_list_size}',
                        f'hls_flags=independent_segments+append_list',
                        # f'hls_flags=single_file',
                        
                        f'hls_segment_type=mpegts',
                        f'hls_segment_filename={segment_pattern_path}',
                        f'hls_playlist_type=event',
                    ])
                
            
            # ofopts = segment_time=2,segment_format=mpegts,segment_list_size=25,segment_start_number=0,segment_list_flags=+live,segment_list=[F:\all\GitHub\vs-mlrt\scripts\server\express-hls-example\src\stream\out.m3u8] 
            cmd = [
                'mpv_.com',
                '-',  
                '--no-config', # Ignore user configurations for predictable behavior in scripts
                '--demuxer=rawvideo',
                f'--demuxer-rawvideo-w={width}',
                f'--demuxer-rawvideo-h={height}',
                '--demuxer-rawvideo-mp-format=bgr24',
                f'--demuxer-rawvideo-fps={framerate}',
                '--audio-file=' + self.cur_audio_file,
                '--o=' + playlist_path if output_hls else f"{self.output_dir}/out_{self.target_time}.mp4",
                f'--ovc={self.args.video_codec}', 
                f'--ovcopts={ovcopts}', # Options for the video codec
                f"--of={'hls' if output_hls else 'mp4'}",
                f"--ofopts={ofopts}",
                '--msg-level=all=error', # Optional: for cleaner logs
                '--input-ipc-server=' + self.encode_mpv_pipe_name,
            ]
                        

            self.encode_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                bufsize=width*height*3# 10**8
            )
            
            
            while not self.stop_flag:
                # time.sleep(0.001)
                # sbs = self.encode_queue.get()
                # continue
                # frame = self.decode_queue.get()
                # if frame is None: 
                #     break
                
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
                    written = self.encode_process.stdin.write(remaining_data)
                    bytes_written += written
                        
                # self.encode_process.stdin.write(frame_bytes)
                # time.sleep(0.001)  # Yield control

            # if self.encode_process:
            #     self.encode_process.stdin.close()
            #     self.encode_process.wait()
            #     print("FFmpeg encoder finished")
            print("Encoder Ended")

        except Exception as e:
            print(f"Encoder error: {e}")
            if self.encode_process:
                self.encode_process.kill()
                
    def encode_to_hls(self, bitrate= 3*1000*1000, serve_while_generating=True):
               
        
        ffmpeg_cmd = ["ffmpeg"]
                # "-probesize", "10KB",  "-rtbufsize", "100MB",

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
                bufsize=1*1024*1024  # Large buffer for video frames
            )
            

            print("Encoder started")
            while self.running :
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
            
            
    # def __init__(self, request, client_address, server, *, directory = None):
    #     super().__init__(request, client_address, server, directory=directory, vp=None)
        
        # if 'Range' in self.headers:
        #     range_header = self.headers['Range']
        #     print(f"Range requested: {range_header}")
            
        #     # # Parse the byte range (optional)
        #     # byte_range = self.parse_byte_range(range_header)
        #     # if byte_range:
        #     #     print(f"Parsed range: {byte_range}")
        
class LoggingHTTPRequestHandler(SimpleHTTPRequestHandler):
                
    def do_GET(self):
        print(f"Requested file: {self.path}")  # Log the requested path
        if self.path.endswith(".ts") and hasattr(self, "vp"):
            self.vp.last_req_seg_n = extract_n(self.path) or 0
            
        # Check if it's an m3u8 file
        if self.path.endswith('.m3u8'):
            self._handle_m3u8_request()
        else:
            super().do_GET()

    def _handle_m3u8_request(self):
        """Handle m3u8 requests by modifying content before sending"""
        try:
            # Get the full path to the file
            path = self.translate_path(self.path)
            
            if os.path.exists(path) and os.path.isfile(path):
                # Read the original content
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Modify the content
                modified_content = self._remove_lines_from_m3u8(content)
                assert not "DISCONTINUITY" in modified_content 
                assert not "EXT-X-ENDLIST" in modified_content

                print("\n\n\n", modified_content, "\n\n\n")
                self.send_response(200)
                self.send_header("Content-type", "application/vnd.apple.mpegurl")
                self.send_header("Content-Length", str(len(modified_content)))
                self.end_headers()
                
                # Send modified content
                self.wfile.write(modified_content.encode('utf-8'))
            else:
                # File not found, let parent handle it
                super().do_GET()
                
        except Exception as e:
            print(f"Error handling m3u8 request: {e}")
            super().do_GET()

    def _remove_lines_from_m3u8(self, content):
        """
        Remove specific lines from m3u8 content.
        Customize this method based on what lines you want to remove.
        """
        lines = content.split('\n')
        filtered_lines = []
        
        # Example patterns to remove - customize these
        patterns_to_remove = [
            '#EXT-X-DISCONTINUITY',
            '#EXT-X-ENDLIST',
        ]
        
        for line in lines:
            # Only keep lines that don't contain any of the patterns to remove
            if not any(pattern in line for pattern in patterns_to_remove):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

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
        