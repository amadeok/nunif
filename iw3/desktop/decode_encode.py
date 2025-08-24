import copy
import gc
import json
import os
import random
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

import time, shutil

from .decode_encode_utils import *
from global_hotkeys import register_hotkeys, start_checking_hotkeys



class HLSEncoder:
    def __init__(self, input_f, output_dir, args, ff_hls_time=4, ff_hls_list_size=0):
        self.input_file = input_f
        self.ff_hls_time = ff_hls_time
        self.ff_hls_list_size = ff_hls_list_size
        self.seg_delta_pause_thres = 2
        self.subtitle_id = 1
        self.restart_mpv_decode_on_seek = True
        self.mpv_bin = "mpv_.com"# os.path.join(os.path.expanduser("~"), r"rifef _\mpv-x86_64-v3-20250824-git-5faec4e\mpv.com") 
        self.use_ffmpeg_encoder = True
        ###
        
        self.b_print_debug = False
        self.mpv_log_levels = {"video_decode": "error", "audio_decode": "error", "encode":"v" }
        #fatal error warn info status v debug trace
        ###
        assert shutil.which(self.mpv_bin)

        self.decode_audio_mpv_ipc_pipe_name = r'\\.\pipe\iw3_decode_audio_mpv_ipc_pipe___'
        self.decode_video_mpv_ipc_pipe_name = r'\\.\pipe\iw3_decode_video_mpv_ipc_pipe___'
        self.encode_audio_mpv_ipc_pipe_name = r'\\.\pipe\iw3_encode_video_mpv_ipc_pipe___'

        self.encode_mpv_pipe_name = r'\\.\pipe\iw3_encode_mpv_pipe__'
        self.keyframes = []
        self.decode_audio_mpv_proc = None
        self.decode_video_mpv_proc = None

        cap = cv2.VideoCapture(self.input_file)
        self.video_info = int(cap.get(3)), int(cap.get(4)), cap.get(5)
        self.width, self.height, self.fps = self.video_info
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = frame_count / self.fps
        cap.release()
        self.sync_queue = queue.Queue()


        # audio
        self.audio_sample_rate=48000
        self.audio_channels=2
        self.audio_bits_per_sample=16
        self.audio_bytes_per_sample = self.audio_bits_per_sample // 8
        self.audio_bytes_per_second = self.audio_sample_rate * self.audio_channels * self.audio_bytes_per_sample
        self.audio_bytes_per_sample_and_channel = self.audio_channels * self.audio_bytes_per_sample
        self.dec_accumulator = DecimalAccumulator(target=self.audio_bytes_per_sample_and_channel)

        samples_per_frame = self.audio_sample_rate / self.fps
        self.audio_bytes_per_frame = samples_per_frame * self.audio_channels * self.audio_bytes_per_sample
        # self.audio_dec, self.audio_int = math.modf(self.audio_bytes_per_frame)

        self.audio_dec = self.audio_bytes_per_frame % self.audio_bytes_per_sample_and_channel
        self.audio_int = round(math.ceil(self.audio_bytes_per_frame) - self.audio_dec)
        ####
        
        round_fps = round(self.fps)
        # audio_buffer_frames_n = self.audio_sample_rate / round_fps
        self.decode_audio_queue = queue.Queue(maxsize=round_fps)
        self.decode_video_queue = queue.Queue(maxsize=round_fps)
        self.encode_video_queue = queue.Queue(maxsize=10)
        self.audio_buffer = ThreadSafeByteFIFO()

        self.video_frame_size = self.width*self.height*3
        self.audio_thread = threading.Thread(target=lambda: 1)
        self.video_thread = threading.Thread(target=lambda: 1)
        self.pipe = None

        self.seeking_flag = ThreadSafeValue[bool](False)
        self.queue_audio_drop_frame = ThreadSafeValue[int](0)
        self.__seek_start_time = 0

        
        self.decoded_audio_frames_n = ThreadSafeValue[int](0)
        self.decoded_video_frames_n = ThreadSafeValue[int](0)
        self.keyframes =  get_keyframes(self.input_file)


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


        self.running = False

        self.args = args
        if args.device.type == "cuda":
            self.cuda_stream = torch.cuda.Stream(device=args.device)
        else:
            self.cuda_stream = None

        self.c = Counter()

        self.seg_delta = None
        self.is_paused = False
        self.last_req_seg_n = 0
        self.newest_seg_n = 0
        
        def my_click_callback(x, y):
            perc = (x / notepad_hwnd.width)*100
            seek_absolute_perc(perc, self.mpv_ipc_control_pipe)

        self.handler = WindowClickHandler()

        notepad_hwnd :gw.Window = find_window_by_title("Notepad++")
        if not notepad_hwnd:
            print("Notepad++ window not found!")
            exit()

        # self.handler.set_click_callback(notepad_hwnd, my_click_callback)

        def toggle_print_debug():  self.b_print_debug = not self.b_print_debug

        bindings = [["window + f10", None, lambda:self.seek_perc_at_keyframe(get_number()), False],
                    ["window + f11", None, toggle_print_debug, False],
                    ["window + f12", None, self.quit_encode_mpv, False]
                    ]
        register_hotkeys(bindings)
        start_checking_hotkeys()
        
        def test():
            sid = 1
            while 1:
                time.sleep(10)
                perc = random.uniform(0, 80)
                print("----> seeking", perc, " ---")
                self.seek_perc_at_keyframe(perc)
                # set_track_by_id("sub", sid, self.decode_video_mpv_ipc_pipe_name)
                # sid+=1
        threading.Thread(target=test, daemon=True).start()
        # httpd = start_http_server(self.output_dir, self)

    def print_debug(self):
        self.c.tick(f"q-> da: {self.decode_audio_queue.qsize()} dv: {self.decode_video_queue.qsize()} ev: {self.encode_video_queue.qsize()}| v: {self.decoded_video_frames_n} a: {self.decoded_audio_frames_n} | ")


    def check_segment_delta(self):
        while 1:
            self.newest_seg_n = get_most_recent_seg_n(self.output_dir)

            if self.newest_seg_n is not None and self.last_req_seg_n is not None:
                self.seg_delta = self.newest_seg_n - self.last_req_seg_n

                if self.seg_delta >= self.seg_delta_pause_thres:
                    print("- Pausing", self.seg_delta, " -")
                    self.is_paused = True
                    # pause_unpause('pause', self.encode_mpv_pipe_name)
                else:
                    print("- Resuming", self.seg_delta, " -")
                    self.is_paused = False
                    # pause_unpause('unpause', self.encode_mpv_pipe_name)
            time.sleep(1)

    def get_last_peek_sizes(self):
        return {e: self.last_peeked_pipe_sizes[getattr(self, e)]/(1000*1000) for e in ["audio_pipe", "video_pipe", "out_audio_pipe", "out_video_pipe"]}

    def deinit(self):
        self.handler.remove_click_callback()
        print("Listener stopped.")

    def init_dir(self):
        def try_del():
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                print("Error cleaning up:", e)
                
        while os.path.isdir(self.output_dir):#or len(os.listdir(self.output_dir)):
            print("Wait for delete")
            try_del()
            time.sleep(0.01)
        os.makedirs(self.output_dir, exist_ok=True)



    def seek_perc_at_keyframe(self,perc ):
        if not perc:
            print("no percentage provided")
            return 
        def get_element_at_percentage(percentage, lst):
            factor = len(lst) * percentage / 100
            index = round(factor)
            index = max(0, min(index, len(lst) - 1))
            return lst[index], index#, factor, index

        target_time, index = get_element_at_percentage(perc, self.keyframes)
        print("%", perc,  "target_time", target_time, "index", index)
        self.__seek(target_time)



    def print_data(self):
        qs = self.queue_sizes()
        print(f"------> {qs} | v: {self.decoded_video_frames_n} a: {self.decoded_audio_frames_n} <-------")

    def queue_sizes(self):
        return  self.decode_video_queue.qsize(),self.decode_audio_queue.qsize()

    def empty_stdout(self, proc: subprocess.Popen):
        stdout_handle = win32file._get_osfhandle(proc.stdout.fileno())
        av_bytes = float("inf")
        while av_bytes:
            data = win32pipe.PeekNamedPipe(stdout_handle, 0)
            av_bytes = data[1]
            if av_bytes:
                print(f"dumping {av_bytes} from proc")
                chunk = read_frame_of_size(stdout_handle, av_bytes, av_bytes)



    def __seek(self, time_):
        sl= 0.1
        pipes = self.decode_video_mpv_ipc_pipe_name,#self.decode_audio_mpv_ipc_pipe_name,
        print("killing audio decode mpv..")
        self.quit_audio_decode_mpv()
        self.audio_thread.join()
        
        if self.restart_mpv_decode_on_seek:
            print("killing video decode mpv..")
            self.quit_video_decode_mpv()
            self.video_thread.join()
        else:
            print("Pausing decoders..")
            # self.seeking_flag.set(True)
            for p in pipes:
                pause_unpause("pause", p)
                
            time.sleep(sl)

            # print("Emptying queues..")

            # for q in self.decode_video_queue,self.decode_audio_queue:
            #     while q.qsize(): q.get()

            time.sleep(sl)
            print("Seeking..",)
            # seek_absolute(time_, self.decode_audio_mpv_ipc_pipe_name)
            seek_absolute(time_, self.decode_video_mpv_ipc_pipe_name)
            time.sleep(sl)
            print("Unpausing" )
            for p in pipes:
                pause_unpause("unpause", p)
        # self.seeking_flag.set(False)
        
        self.__seek_start_time = time_
        if self.restart_mpv_decode_on_seek:
            self.video_thread = threading.Thread(target=self.video_decoder, daemon=True)
            self.video_thread.start()
        self.audio_thread = threading.Thread(target=self.audio_decoder, daemon=True)
        self.audio_thread.start()
        

    def quit_audio_decode_mpv(self):
        res = send_cmd({  "command": ["quit" ]  }, self.decode_audio_mpv_ipc_pipe_name )
        try: self.decode_audio_mpv_proc.stdout.close()
        except Exception as e:print("Error closing decode audio mpv stdout handle", e)

    def quit_video_decode_mpv(self):
        res = send_cmd({  "command": ["quit" ]  }, self.decode_video_mpv_ipc_pipe_name )
        try: self.decode_video_mpv_proc.stdout.close()
        except Exception as e:print("Error closing decode video mpv stdout handle", e)
        
    def quit_encode_mpv(self):

        res = send_cmd({  "command": ["quit" ]  }, self.encode_audio_mpv_ipc_pipe_name )
        time.sleep(.1)
        def close_stdin():
            try:
                self.encode_process.stdin.close()
                time.sleep(.1)
            except Exception as e:print("Error closing encode stdin handle", e)
        def close_pipe():
            try: win32file.CloseHandle(self.pipe)
            except Exception as e:print("errror", e)
        
        threading.Timer(.1, close_stdin ).start()
        t = threading.Timer(.2, close_pipe )
        t.start()
        t.join()


    # def extract_audio(self, start_time_seconds, out_file):
    #     cmd = ["ffmpeg", "-ss", str(start_time_seconds), "-i", self.input_file, "-vn", "-c:a", "copy", "-f", "mp4", out_file, "-y"]
    #     print("Getting audio")
    #     ret = subprocess.call(cmd)
    #     print(F"Got audio {out_file} at {start_time_seconds}")

    # def get_all_audio(self):
    #     self.extract_audio(0, self.whole_audio_file)

    # def get_audio_from_sec(self, start_second):
    #     self.extract_audio(start_second, self.cur_audio_file)

    # def seek_restart(self,time ):
    #     self.target_time = time
    #     self.restart()


    def audio_decoder(self):
        """Spawn MPV instance for audio decoding."""

        audio_args =  [
            self.mpv_bin,
            self.input_file,
            '--no-config',
            "--hr-seek=no",
            f"--start={self.__seek_start_time}",
            # "--oacopts=ar=44100,ac=2",  # Sample rate: 44.1kHz, Channels: 2 (stereo)
            f"--af=format=srate={self.audio_sample_rate}",  # ,ac={self.audio_channels}",
            "--of=s16le",
            "--oac=pcm_s16le",
            "-o",
            "-",  # Output to stdout
            "--input-ipc-server=" + self.decode_audio_mpv_ipc_pipe_name,
            "--msg-level=all=warn",
        ]

        try:
            print("Starting audio MPV instance...")
            self.decode_audio_mpv_proc = subprocess.Popen(
                audio_args,
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,,
                bufsize=math.ceil(self.audio_bytes_per_frame)
            )
            print("Audio MPV instance started successfully!")

            while True:
                self.sync_queue.get()
                emission, current_total = self.dec_accumulator.add_number(self.audio_dec)
                audio_s = self.audio_int+(emission)
                assert audio_s % self.audio_bytes_per_sample_and_channel == 0

                chunk = read_frame_of_size(self.decode_audio_mpv_proc.stdout, audio_s, audio_s)
                if not chunk:
                    break  # EOF reached, process has exited
                le = len(chunk)
                assert le == audio_s


                self.decoded_audio_frames_n.increment()
                self.decode_audio_queue.put(chunk)
                time.sleep(0.001)

            # print(f"Audio: Received {len(self.audio_buffer)/1_000_000} MB {self.decode_audio_mpv_proc.poll()}")

        except FileNotFoundError:
            print("Error: MPV not found. Please install MPV and ensure it's in your PATH")
            return None
        except Exception as e:
            print(f"Error MPV audio loop: {e}")
            return None
        finally:
            print("Mpv audio decoder ended")

    def video_decoder(self):
        
        video_args = [ 
            self.mpv_bin, self.input_file, '--no-config',
            f"--start={self.__seek_start_time if self.restart_mpv_decode_on_seek else 0}",  
            *([f"--sid={self.subtitle_id}", "--vf=sub"] if self.subtitle_id != None else  []),
            "--hr-seek=no",
            "--ovc=rawvideo",
            f"--vf=format=fmt=rgb24",  # ,fps={self.fps}",
            "--of=rawvideo",  
            "--input-ipc-server=" + self.decode_video_mpv_ipc_pipe_name,
            "-o",  "-",
            f"--msg-level=all={self.mpv_log_levels['video_decode']}",
        ]

        try:
            print("Starting video MPV instance...")
            self.decode_video_mpv_proc = subprocess.Popen(
                video_args,
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                bufsize=self.video_frame_size
            )
            print("Video MPV instance started successfully!")

            while True:
                data = read_frame_of_size(self.decode_video_mpv_proc.stdout, self.video_frame_size, self.video_frame_size )

                framergb = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))

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


                self.decoded_video_frames_n.increment()
                self.decode_video_queue.put(frame)
                
                de =  self.decoded_video_frames_n.get()-self.decoded_audio_frames_n.get()
                for x in range(de if de > 2 else 1):
                    self.sync_queue.put(1)
                    # if x > 0:
                    #     print("sync queue", x," de:", de, "qs:",self.sync_queue.qsize(), " --")
                
        except Exception as e:
            print(f"Video read error: {e}")

    def get_encoder_cmd(self, out_width):


        playlist_path = os.path.join(self.output_dir, 'playlist.m3u8')
        segment_pattern_path = os.path.join(self.output_dir, 'segment_%03d.ts')
            
        codec = self.args.video_codec

        output_hls = True
        
        if self.use_ffmpeg_encoder:
            buf_settings = [
                 "-probesize", "1KB", #"-thread_queue_size", "500000",# "-rtbufsize", "20000000"
            ]
                        
            ofopts = ''
            if not "nvenc" in codec:
                video_opts = [
                    "-preset", self.args.preset,
                    "-b:v", self.args.video_bitrate
                ]
                ofopts = ["-movflags", "+frag_keyframe+empty_moov"]
            else:
                video_opts = [
                    "-preset", self.args.nvenc_preset,
                    "-rc", "vbr",
                    "-cq", "18",
                    "-b:v", "0",
                    "-bufsize", "20M",
                    "-profile", "high"
                ]

            if output_hls:
                hls_opts = [
                    "-f", "hls",
                    "-hls_time", str(self.ff_hls_time),
                    "-hls_list_size", str(self.ff_hls_list_size),
                    "-hls_flags", "independent_segments+append_list",
                    "-hls_segment_type", "mpegts",
                    "-hls_segment_filename", segment_pattern_path,
                    "-hls_playlist_type", "event"
                ]
            else:
                hls_opts = ["-f", "mp4"]

            cmd = [
                "ffmpeg",
                *buf_settings,
                "-i", self.encode_mpv_pipe_name,  # External audio file
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{out_width}x{self.height}",
                "-framerate", str(self.fps),
                # "-thread_queue_size", "50000",
                "-i", "-", 
                "-map", "0:a:0", 
                "-map", "1:v:0", 
                "-c:v", self.args.video_codec,
                "-pix_fmt", "yuv420p",  #important, otherwise will encode in unsupported pi
                *video_opts,
                "-r", str(self.fps),  # Output framerate
                "-c:a", "aac",  # Audio codec (adjust as needed)
                *hls_opts,
                *ofopts,
                "-y",  # Overwrite output file
                playlist_path if output_hls else f"{self.output_dir}/out_{self.target_time}.mp4"
            ]

            # Remove empty strings if any
            cmd = [arg for arg in cmd if arg]
        else:
            ofopts = f''
            if not "nvenc" in codec:
                ovcopts=f"preset={ self.args.preset},b={self.args.video_bitrate}"
                ofopts =  f'movflags=+frag_keyframe+empty_moov'
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
            ofopts = ",".join([
                        # '-f', 'hls',
                        # f"r={self.fps}",
                        f'hls_time={self.ff_hls_time}',
                        f'hls_list_size={self.ff_hls_list_size}',
                        f'hls_flags=independent_segments+append_list',
                        # f'hls_flags=single_file',
                        f'hls_segment_type=mpegts',
                        f'hls_segment_filename={segment_pattern_path}',
                        f'hls_playlist_type=event',
                    ])
                
            cmd = [
                    self.mpv_bin,
                    "-",
                    f"--cache-secs=2", "--cache=yes",
                    # r'd:\1.mkv',  
                    '--no-config', # Ignore user configurations for predictable behavior in scripts
                    # " --demuxer-lavf-o=thread_queue_size=50000,rtbufsize=20000000,probesize=1KB",
                    '--demuxer=rawvideo',
                    f'--demuxer-rawvideo-w={out_width}',
                    f'--demuxer-rawvideo-h={self.height}',
                    '--demuxer-rawvideo-mp-format=bgr24',
                    f'--demuxer-rawvideo-fps={self.fps}',
                    "--aid=2",
                    f'--audio-file={self.encode_mpv_pipe_name}', # Add external audio file
                    #  '--o=' + f"temp/stream.mpd", # Output file (implicitly overwrites)
                    #  "--of=dash",
                    '--o=' + playlist_path if output_hls else f"{self.output_dir}/out_{self.target_time}.mp4",
                    f'--ovc={self.args.video_codec}',
                    f'--ovcopts={ovcopts}', # Options for the video codec
                    f"--of={'hls' if output_hls else 'mp4'}",
                    f"--vf=format=fmt=yuv420p",  #important 
                    f"--vf=fps={self.fps}",
                    f'--ofopts={ofopts}', # Output format options
                    '--input-ipc-server=' + self.encode_audio_mpv_ipc_pipe_name,
                    f"--msg-level=all={self.mpv_log_levels['encode']}",
                ]
            cmd_ = " ".join(cmd)
            print("\ncmd", cmd_, "\n")
        return cmd

            
    def encoder(self):
        print("Encoder started")

        out_audio_pipe_size = self.audio_bytes_per_second*3 if self.use_ffmpeg_encoder else 1024*1024*1

        self.pipe = win32pipe.CreateNamedPipe(
            self.encode_mpv_pipe_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            2, out_audio_pipe_size, out_audio_pipe_size, 0, None
        )
        print(f"Output audio named pipe created: {self.encode_mpv_pipe_name} with size {out_audio_pipe_size} ")

        try:
            out_width = self.get_output_width(self.width)

            cmd = self.get_encoder_cmd(out_width)
            
            # time.sleep(2)
            self.encode_process = subprocess.Popen( cmd, stdin=subprocess.PIPE,
                                                #    bufsize= self.video_frame_size if self.use_ffmpeg_encoder else 1024*1024
                                                   )#1024*1024*1 )
            
            win32pipe.ConnectNamedPipe(self.pipe, None)
            print("Pipe connected")

            header = generate_wav_header( self.audio_sample_rate, self.audio_bits_per_sample,self.audio_channels, 0x1fffffff )
            # with open("example.wav", "wb") as f:
            #     f.write(header)
            
            win32file.WriteFile(self.pipe, header)

            tot_audio_bytes_wr = 0

            def write_audio(src:ThreadSafeByteFIFO):
                # nonlocal acc
                # acc+= self.audio_dec
                # acc, extra = math.modf(acc)
                # extra =int(extra)
                # audio_s = self.audio_int+(extra)
                emission, current_total = self.dec_accumulator.add_number(self.audio_dec)
                audio_s = self.audio_int+(emission)
                assert audio_s % self.audio_bytes_per_sample_and_channel == 0
                
                audio_data = src.get(audio_s)
                audio_written_bytes = win32file.WriteFile(self.pipe, audio_data)[1]
                return audio_written_bytes

            from collections import deque
            dummy_audio_data = ThreadSafeByteFIFO()
            dummy_audio_data.put(create_sine_wave_bytes(self.audio_bytes_per_second*3, self.audio_sample_rate))

            frames_written = 0
            ss = int(self.audio_bytes_per_second*1.5)
            while tot_audio_bytes_wr < ss:
                wr  = write_audio(dummy_audio_data)
                tot_audio_bytes_wr+=wr
                # print("dummy a", wr)
                frames_written+=1

            # remainder = tot_wr % self.audio_bytes_per_sample_and_channel
            # if remainder != 0:
            #     needed = self.audio_bytes_per_sample_and_channel - remainder
            #     f"{tot_wr} is not divisible by {self.audio_bytes_per_sample_and_channel}"
            #     tot_wr += win32file.WriteFile(self.pipe, bytearray(needed))[1]
            assert tot_audio_bytes_wr % 4 == 0

            dummy_image = bytearray([128, 128, 128] * out_width * self.height)

            for x in range(frames_written):
                ret = self.encode_process.stdin.write(dummy_image)
                # print(f"{x} dummy v", ret)
            print("Encoder started")

            # de = deque(maxlen=150)

            while 1:

                sbs = self.encode_video_queue.get()

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
                    

                audio_frame = self.decode_audio_queue.get()
                audio_bytes_written = win32file.WriteFile(self.pipe, audio_frame)[1]
                assert len(audio_frame) == audio_bytes_written
                # de.append(audio_bytes_written)
                # print("dea", np.mean(de), self.audio_bytes_per_frame)
                
                # time.sleep(0.001) 

            print("Encoder Ended")

        except Exception as e:
            print(f"Encoder error: {e}")
            if self.encode_process:
                self.encode_process.kill()


    def encoder_old(self):
        """Async function that encodes frames using FFmpeg subprocess"""
        print("Encoder started")

        width, height, framerate =  self.video_info#self.video_stream_ready.result()
        print("Encode started", width, height, framerate)

        width = self.get_output_width(width)


        # segment_pattern_path = os.path.join(self.output_dir, 'segment_.ts')

        try:

            playlist_path = os.path.join(self.output_dir, 'master.m3u8')
            segment_pattern_path = os.path.join(self.output_dir, 'segment_%03d.ts')
            
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
                self.mpv_bin,
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

    def get_output_width(self, width):
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
        return width

    def start(self):
        self.audio_thread = threading.Thread(target=self.audio_decoder)
        self.video_thread = threading.Thread(target=self.video_decoder)
        self.encode_thread = threading.Thread(target=self.encoder)
        thread = threading.Thread( target=self.check_segment_delta, daemon=True)

        self.audio_thread.start()
        self.encode_thread.start()
        self.video_thread.start()
        thread.start()

class LoggingHTTPRequestHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        print(f"Requested file: {self.path}")  # Log the requested path
        # if self.path.endswith(".ts") and hasattr(self, "vp"):
        #     self.vp.last_req_seg_n = extract_n(self.path) or 0

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
        print(f"Serving HLS stream at http://localhost:{vp.args.port}/master.m3u8")
        httpd.serve_forever()

    thread = threading.Thread(target=run_server)
    thread.daemon = True
    thread.start()
    return httpd

