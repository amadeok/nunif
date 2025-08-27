import configparser
import os, win32file, threading, io , math, struct, subprocess, win32pipe, json
import shutil
from pynput import mouse
import  win32gui, time, pygetwindow as gw, re
import tempfile
from urllib.parse import urlparse
import yt_dlp

class DecimalAccumulator:
    def __init__(self, target=4.0):
        self.accumulator = 0.0
        self.target = target
        self.emissions = 0
    
    def add_number(self, num):
        # fractional = num - int(num)
        self.accumulator += num
        
        result = 0
        if self.accumulator >= self.target:
            result = self.target
            self.accumulator -= self.target
            self.emissions += 1
        
        return result, self.accumulator
    
def read_frame_of_size(stream, frame_size_bytes, bufsize):
    data = b''
    
    if isinstance(stream,  io.BufferedReader):
        read = lambda src, size: src.read(size) 
    else:
        read = lambda src, size: win32file.ReadFile(src, size)[1]
    while 1:
        le = len(data)
        if le >= frame_size_bytes:break
        read_size = min(frame_size_bytes - len(data), bufsize)
        
        chunk = read(stream, read_size) #stream.read(read_size)
        if not chunk:
            return None
        data += chunk
    assert(len(data) == frame_size_bytes)
    return data

def load_rife_config( config_file_path: str, interpolate_conf_map) -> bool:
    config_file_path = os.path.expandvars(config_file_path)
    config = configparser.ConfigParser()
    config.optionxform = str  # This preserves the original case

    try:
        config.read(config_file_path)
    except Exception as e:
        print(f"Error reading config file: {e}")
        return False
    
    if 'main' not in config:
        print("Error: 'main' section not found in config file")
        return False
    
    for key, value in config['main'].items():
        env_key = interpolate_conf_map.get(key, key.upper())
        os.environ[env_key] = value
        print(f"Set {env_key} = {value}")
        
    return True

class ThreadSafeByteFIFO:
    def __init__(self, maxsize=0):

        self._buffer = bytearray()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        # Condition to signal when the buffer is no longer full
        self._not_full = threading.Condition(self._lock)
        # Condition to signal when the buffer is no longer empty
        self._not_empty = threading.Condition(self._lock)

    def put(self, data):
        """Adds data to the buffer. Blocks if the buffer is full."""
        with self._not_full:
            # Wait as long as the buffer is full
            if self._maxsize > 0:
                while len(self._buffer) >= self._maxsize:
                    self._not_full.wait()
            
            self._buffer.extend(data)
            # Signal to a waiting 'get' thread that the buffer now has data
            self._not_empty.notify()

    def get(self, size):
        """Removes and returns 'size' bytes. Blocks if the buffer is empty."""
        with self._not_empty:
            # Wait as long as the buffer is empty
            while not self._buffer:
                self._not_empty.wait()

            data = self._buffer[:size]
            del self._buffer[:size]
            # Signal to a waiting 'put' thread that the buffer now has space
            self._not_full.notify()
            return data

    def peek(self, size):
        """Returns 'size' bytes without removing them from the buffer."""
        with self._lock:
            return self._buffer[:size]

    def __len__(self):
        """Returns the current size of the buffer."""
        with self._lock:
            return len(self._buffer)

def create_sine_wave_bytes(size_bytes, sample_rate=44100, frequency=440.0, amplitude=0.5):

    num_samples = size_bytes // 2
    max_value = 32767 * amplitude  # 16-bit signed integer max value
    angular_frequency = 2 * math.pi * frequency / sample_rate
    
    data = b''.join(
        struct.pack('<h', int(max_value * math.sin(angular_frequency * i)))
        for i in range(num_samples)
    )
    
    return data[:size_bytes]

def get_number():
    import tkinter as tk
    from tkinter import simpledialog    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    number = simpledialog.askinteger("Input", "Enter a number:",  parent=root,  minvalue=0,  maxvalue=100)
    root.destroy()
    return int(number) if number else None


from typing import Literal, TypeVar, Generic, Union, Optional
T = TypeVar('T')


class ThreadSafeValue(Generic[T]):
    def __init__(self, initial_value: T = None):
        self._value: T = initial_value
        self._lock = threading.Lock()
    
    @property
    def value(self) -> T:
        with self._lock:
            return self._value
    
    @value.setter
    def value(self, new_value: T) -> None:
        with self._lock:
            self._value = new_value
    
    def set(self, value: T) -> None:
        with self._lock:
            self._value = value
    
    def get(self) -> T:
        with self._lock:
            return self._value
    
    def increment(self, amount: Union[int, float] = 1) -> T:
        """Increment the value by the specified amount and return the new value."""
        with self._lock:
            if not isinstance(self._value, (int, float)):
                raise TypeError(f"Cannot increment non-numeric value of type {type(self._value)}")
            self._value += amount
            return self._value
    
    def decrement(self, amount: Union[int, float] = 1) -> T:
        """Decrement the value by the specified amount and return the new value."""
        with self._lock:
            if not isinstance(self._value, (int, float)):
                raise TypeError(f"Cannot decrement non-numeric value of type {type(self._value)}")
            self._value -= amount
            return self._value
    
    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._value)
    
    def __str__(self) -> str:
        with self._lock:
            return str(self._value)
    
    def __repr__(self) -> str:
        with self._lock:
            return f"ThreadSafeValue({repr(self._value)})"




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
    

def cycle_track(track_type :Literal['audio', 'sub'], pipe_path ):
    command = {  "command": ["cycle", track_type ]  } 
    res = send_cmd(proc_cmd(command), pipe_path)

def set_track_by_id(track_type :Literal['audio', 'sub'], id, pipe_path):
    command = {  "command": ["set", track_type, str(id) ]  } 
    res = send_cmd(proc_cmd(command), pipe_path)


def cycle_track(op: Literal['toggle', 'pause', 'unpause'], pipe_path=None):
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

def seek_absolute(time, pipe_path=None):
    command = { "command": ["seek", time, "absolute", "exact"] }
    res = send_cmd(proc_cmd(command), pipe_path)
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
    else: 
        response =  None
    if "handle" in locals():
        try:  win32file.CloseHandle(handle)
        except Exception  as e:print("Error closing handle")
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
    
def get_property(property):
    command = {  "command": ["get_property", property]  }
    return json.dumps(command) + '\n'

def get_property_partial(property, pipe_path=None, read_response=True):
        
    response = send_cmd(get_property(property), pipe_path, read_response=read_response)
    val  = None
    if response and 'data' in response:
        val = response['data']
        #print(f"Current playback speed: {current_speed}x")
    else:
        print(f"Failed to retrieve {property} property")

    return val

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


def generate_wav_header( sample_rate: int, bit_depth: int, channels: int, num_samples: int ) -> bytes:

    byte_rate = sample_rate * channels * bit_depth // 8
    block_align = channels * bit_depth // 8
    data_size = num_samples * channels * bit_depth // 8
    header_size = 44  # Standard WAV header size
    
    # RIFF chunk
    riff = b'RIFF'
    chunk_size = data_size + 36  # data_size + (header_size - 8)
    
    # WAVE chunk
    wave = b'WAVE'
    
    # fmt subchunk
    fmt_chunk = b'fmt '
    fmt_size = 16  # Size of fmt chunk
    audio_format = 1  # PCM
    fmt_chunk_data = struct.pack(
        '<LHHLLHH',
        fmt_size,
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bit_depth
    )
    
    # data subchunk
    data_chunk = b'data'
    data_chunk_size = data_size
    
    # Assemble header
    header = b''.join([
        riff,
        struct.pack('<L', chunk_size),
        wave,
        fmt_chunk,
        fmt_chunk_data,
        data_chunk,
        struct.pack('<L', data_chunk_size)
    ])
    
    return header

def get_keyframes(file):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'packet=pts_time,flags',
        '-of', 'csv=p=0',
        file
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

    keyframes = sorted(set(keyframes))
    return keyframes

if __name__ == "__main__":
    import sys
    path = "".join(sys.argv[1:])
    print("path", path)
    print(get_video_pixel_format_ffprobe(path))
    
    
def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
    

def download_url_to_temp(url, ytdlp_options, download=True, cleanup_on_failure=True):
    temp_dir = os.path.join( tempfile.gettempdir(), "iw3_downloads")# mkdtemp()
    
    try:
        ydl_opts = {
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'format': ytdlp_options,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=download)    
            downloaded_file = ydl.prepare_filename(info)
            return { 'success': True, 'file_path': downloaded_file, 'info': info }
            
    except Exception as e:
        if cleanup_on_failure:
            try:  shutil.rmtree(temp_dir)
            except:   pass
        return { 'success': False, 'file_path': None, 'error': str(e) }

def get_formats(url):
    with yt_dlp.YoutubeDL() as ydl:
        info = ydl.extract_info(url, download=False)
        return info

def select_best_formats(formats, max_w=1920):
    video_formats = []
    audio_formats = []
    
    for f in formats:
        if f.get('vcodec') != 'none' and f.get('acodec') == 'none':  # Video-only formats
            video_formats.append(f)
        elif f.get('acodec') != 'none' and f.get('vcodec') == 'none':  # Audio-only formats
            audio_formats.append(f)
        elif f.get('vcodec') != 'none' and f.get('acodec') != 'none':  # Combined formats
            # Treat as video format for selection, but note it has audio
            video_formats.append(f)
    
    # Select best video under 1080p
    best_video = None
    for v in video_formats:
        if v.get('width') is not None and v['width']  <= max_w:
            vb = v.get('bitrate', 0)
            bvb = best_video.get('bitrate', 0) if best_video else 0
            if best_video is None or vb >= bvb:
                best_video = v
    
    # Select best audio
    best_audio = None
    for a in audio_formats:
        if best_audio is None or a.get('bitrate', 0) >= best_audio.get('bitrate', 0):
            best_audio = a
    
    return best_video, best_audio

def get_yt_dlp_otions(url, max_w):
    info = get_formats(url)
    formats = info['formats']
    best_video, best_audio = select_best_formats(formats, max_w)
    mapping = {
        'info': info,
        'best_video_fmt': best_video,
        'best_audio_fmt': best_audio,
        'best_video_fmt_id': best_video['format_id'],
        'best_audio_fmt_id': best_audio['format_id']
    }
    return mapping
    info, best_video_fmt, best_audio_fmt, best_video_fmt_id, best_audio_fmt_id
    return info, best_video, best_audio, best_video['format_id'], best_audio['format_id']
    if best_video and best_audio:
        # If best_video is a combined format, we might not need separate audio
        if best_video.get('acodec') != 'none':
            # This format already includes audio, so we can use it alone
            format_code = str(best_video['format_id'])
        else:
            # Need to merge video and audio
            format_code = f"{best_video['format_id']}+{best_audio['format_id']}"
        print(f"Download command: yt-dlp -f '{format_code}' {url}")
    else:
        print("Could not find suitable formats.")
        return None, None
    return format_code

if __name__ == '__main__':
    url = input("Enter the video URL: ")
    main(url)