import copy
import gc
import json
import os
import random
import subprocess
import time
import threading
import uuid
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
import pygetwindow as gw, math, pywintypes

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
    def __init__(self, input_f, output_dir, args, ff_hls_time=6, ff_hls_list_size=0):
        self.input_file = input_f
        self.ff_hls_time = ff_hls_time
        self.ff_hls_list_size = ff_hls_list_size
        self.seg_delta_pause_thres = 2
        self.subtitle_id = 1
        self.restart_mpv_decode_on_seek = True
        self.mpv_bin = "mpv_.com"# os.path.join(os.path.expanduser("~"), r"rifef _\mpv-x86_64-v3-20250824-git-5faec4e\mpv.com") 
        # self.mpv_bin = os.path.join(os.path.expanduser("~"), r"rifef _\mpv-x86_64-v3-20250824-git-5faec4e\mpv.com") 
        self.use_ffmpeg_encoder = True
        self.output_pixel_format = getattr(args, "output_pix_fmt", "yuv420p")
        self.args = args
        ###
        
        self.b_print_debug = False
        self.mpv_log_levels = {"video_decode": "error", "audio_decode": "error", "interpolate": "error", "encode":"info" }
        #mpv log level: fatal error warn info status v debug trace
        #ffmpeg log levels:  quiet panic fatal error warning info verbose debug trace 
        assert shutil.which(self.mpv_bin)

        pipe_id = str(uuid.uuid4())[:8]

        self.decode_audio_mpv_ipc_pipe_name = f'\\\\.\\pipe\\iw3_decode_audio_mpv_ipc_pipe___{pipe_id}'
        self.decode_video_mpv_ipc_pipe_name = f'\\\\.\\pipe\\iw3_decode_video_mpv_ipc_pipe___{pipe_id}'
        self.encode_mpv_ipc_pipe_name = f'\\\\.\\pipe\\iw3_encode_mpv_ipc_pipe___{pipe_id}'
        self.interpolate_ipc_control_pipe = f"\\\\.\\pipe\\iw3_rife_output_ipc_pipe___{pipe_id}"

        self.interpolate_output_pipe_name = f"\\\\.\\pipe\\iw3_rife_output_{pipe_id}"
        self.encode_mpv_pipe_name = f'\\\\.\\pipe\\iw3_encode_mpv_pipe__{pipe_id}'
        
        self.keyframes = []
        self.decode_audio_mpv_proc = None
        self.decode_video_mpv_proc = None
        self.interpolate_process = None

        cap = cv2.VideoCapture(self.input_file)
        self.video_info = int(cap.get(3)), int(cap.get(4)), cap.get(5)
        self.width, self.height, self.fps = self.video_info
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = frame_count / self.fps
        cap.release()
        self.sync_queue = queue.Queue()
        
        #####interpolation
        self.interpolate_conf_map = {
            "trtDll_path": "RIFE_PLAYER_TRT_DLL_PATH", 
            # "rootBin": "RIFE_PLAYER_ROOT_PATH",
            "mlrtScriptPath": "RIFE_PLAYER_MLRT_SCRIPT_PATH"
        }
        assert load_rife_config( getattr(args, "rife_config_path", r"C:\Users\%username%\source\repos vs\rifef_\rifef_\folders.ini"), self.interpolate_conf_map)
        self.interpolation_multiplier = getattr(args, "int_mult", 1)
        self.vsScriptPath = os.getenv("vsScriptPath")
        os.environ["RIFE_PLAYER_MULTIPLIER"] = str(self.interpolation_multiplier)
        os.environ["vs_output_pixel_format"] = self.output_pixel_format
        self.using_interpolator = args.output_mode == "hls_ffmpeg" and self.interpolation_multiplier > 1
        self.vapoursynth_buffer_frames = (1,4)
        self.interpolate_started = Future()
        #####
        
        # audio
        self.audio_sample_rate=48000
        self.audio_channels=2
        self.audio_bits_per_sample=16
        self.audio_bytes_per_sample = self.audio_bits_per_sample // 8
        self.audio_bytes_per_second = self.audio_sample_rate * self.audio_channels * self.audio_bytes_per_sample
        self.audio_bytes_per_sample_and_channel = self.audio_channels * self.audio_bytes_per_sample
        self.dec_accumulator = DecimalAccumulator(target=self.audio_bytes_per_sample_and_channel)

        out_fps = self.get_output_fps()# self.fps * self.interpolation_multiplier if self.interpolation_multiplier > 1 else self.fps
        samples_per_frame = self.audio_sample_rate / self.fps
        self.audio_bytes_per_frame = samples_per_frame * self.audio_channels * self.audio_bytes_per_sample
        # self.audio_dec, self.audio_int = math.modf(self.audio_bytes_per_frame)

        self.audio_dec = self.audio_bytes_per_frame % self.audio_bytes_per_sample_and_channel
        self.audio_int = round(math.ceil(self.audio_bytes_per_frame) - self.audio_dec)
        self.last_extra_audio_frame = 0#time.time()
        ####
        
        round_fps = round(out_fps)
        # audio_buffer_frames_n = self.audio_sample_rate / round_fps
        self.decode_audio_queue = queue.Queue(maxsize=round_fps)
        self.decode_video_queue = queue.Queue(maxsize=round_fps)
        self.encode_video_queue = queue.Queue(maxsize=10)
        # self.interpolate_output_queue = queue.Queue(maxsize=10)
        self.interpolate_input_queue =  queue.Queue(maxsize=10)
        
        self.audio_buffer = ThreadSafeByteFIFO()

        self.rgb_video_frame_size = self.width*self.height*3
        self.audio_thread = threading.Thread(target=lambda: 1)
        self.video_thread = threading.Thread(target=lambda: 1)
        self.interpolate_thread = threading.Thread(target=lambda: 1)
        self.pipe = None
        self.interpolate_output_pipe_handle = None

        # self.seeking_flag = ThreadSafeValue[bool](False)
        # self.queue_audio_drop_frame = ThreadSafeValue[int](0)
        self.__seek_start_time = 0
        self.__stop_all = False

        
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

        # self.handler = WindowClickHandler()

        # notepad_hwnd :gw.Window = find_window_by_title("Notepad++")
        # if not notepad_hwnd:
        #     print("Notepad++ window not found!")
        #     exit()

        # self.handler.set_click_callback(notepad_hwnd, my_click_callback)

        def toggle_print_debug(): 
            self.print_debug()
            self.b_print_debug = not self.b_print_debug

        bindings = [["window + f2", None, lambda:self.seek_perc_at_keyframe(get_number()), False],
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
        #threading.Thread(target=test, daemon=True).start()
        httpd = start_http_server(self.output_dir, self)
        
    def get_output_fps(self):
        if self.using_interpolator:
            return self.fps * self.interpolation_multiplier 
        else:
            return self.fps

    def print_debug(self):
        str_ =  f"q->  da: {self.decode_audio_queue.qsize():3d} dv: {self.decode_video_queue.qsize():3d} "
        str_ += f"ii: {self.interpolate_input_queue.qsize():3d} ev: {self.encode_video_queue.qsize():3d} "
        # str_ += f"ev: {self.encode_video_queue.qsize()} "
        str_ += f"| v: {self.decoded_video_frames_n} a: {self.decoded_audio_frames_n} | "
        self.c.tick(str_)

    def check_segment_delta(self):
        while not self.__stop_all:
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
            for x in range(10):
                time.sleep(0.1)
                if self.__stop_all: break

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

    def stop_all(self):
        self.__stop_all = True
        self.quit_audio_decode_mpv()
        self.quit_video_decode_mpv()
        self.quit_encode_mpv()
        self.stop_interpolator()
        print("Joining encode thread")
        self.encode_thread.join()
        print("Joining audio decode thread")
        self.audio_thread.join()
        print("Joining video decode thread")
        self.video_thread.join()
        print("Joining interpolate thread")
        self.interpolate_thread.join()
        print("Joining segment check thread")
        self.segment_thread.join()
        

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

        res = send_cmd({  "command": ["quit" ]  }, self.encode_mpv_ipc_pipe_name )
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


        
    def stop_interpolator(self):
        send_cmd({  "command": ["quit" ]  }, self.interpolate_ipc_control_pipe )

        if self.interpolate_process:
            try:
                try:  win32file.CloseHandle(self.interpolate_output_pipe_handle)
                except Exception as e :print("Error closing rife output handle")
                
                if self.interpolate_process.stdin:
                    self.interpolate_process.stdin.close()
                
                self.interpolate_process.terminate()
                for _ in range(10):  
                    if self.interpolate_process.poll() is not None:
                        break
                    time.sleep(0.1)
            
                # if self.process.poll() is None:
                #     self.process.kill()
            except:
                pass
        
        while not self.encode_video_queue.empty():
            try:  self.encode_video_queue.get_nowait()
            except queue.Empty:  break
            
        if self.interpolate_thread.is_alive():
            self.interpolate_thread.join()

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
                        print("No data from mpv audio decoder, end of file?")
                        break
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
                bufsize=self.rgb_video_frame_size
            )
            print("Video MPV instance started successfully!")
            
            while True:
                while self.is_paused:
                    time.sleep(0.1)
                data = read_frame_of_size(self.decode_video_mpv_proc.stdout, self.rgb_video_frame_size, self.rgb_video_frame_size )
                if not data:
                    print("No data from mpv video decoder, end of file?")
                    break
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
                
                self.sync_queue.put(1)
                for x in range(1):     
                    if de > 2 and self.sync_queue.qsize() < 3:
                        self.sync_queue.put(1)
                    #  if de > 2 and  time.time() - self.last_extra_audio_frame > 0.1:
                    #     self.last_extra_audio_frame = time.time()
                    #     for x in range(de+1):
                    #         self.sync_queue.put(1)
                        print("---- sync queue", self.sync_queue.qsize(), " ----")
                # amount = self.interpolation_multiplier if self.interpolation_multiplier > 1 else 1
                # if  1 or self.sync_queue.qsize() < 3:
                # for x in range(de if de > 2 else 1):
                #     self.sync_queue.put(1)
                #     if x > 0:
                #         print("sync queue", x," de:", de, "qs:",self.sync_queue.qsize(), " --")
                
        except Exception as e:
            print(f"Video read error: {e}")

    def get_encoder_cmd(self, out_width):

        out_fps = self.get_output_fps()# self.fps * self.interpolation_multiplier if self.interpolation_multiplier > 1 else self.fps
        playlist_path = os.path.join(self.output_dir, 'playlist.m3u8')
        segment_pattern_path = os.path.join(self.output_dir, 'segment_%03d.ts')
            
        codec = self.args.video_codec
        

        if self.args.output_mode == "local_mpv":
            buffer_frames = ":".join([str(e) for e in self.vapoursynth_buffer_frames])

            fmt_arg = f"format={self.output_pixel_format}"
            rife_arg = [f'--vf=vapoursynth=[{self.vsScriptPath}]:{buffer_frames},{fmt_arg}'] if self.interpolation_multiplier > 1 else [f"--vf={fmt_arg}"]
            cmd = [
                self.mpv_bin,
                "-",
                f"--cache-secs=2", "--cache=yes",
                # r'd:\1.mkv',  
                "--hwdec=auto-copy",
                "--hwdec-codecs=all",
                "--vo=gpu-next",
                "--profile=fast",
                # "--video-sync=desync",
                '--no-config', # Ignore user configurations for predictable behavior in scripts
                # " --demuxer-lavf-o=thread_queue_size=50000,rtbufsize=20000000,probesize=1KB",
                '--demuxer=rawvideo',
                f'--demuxer-rawvideo-w={out_width}',
                f'--demuxer-rawvideo-h={self.height}',
                '--demuxer-rawvideo-mp-format=bgr24',
                f'--demuxer-rawvideo-fps={out_fps}',
                "--aid=1",
                f'--audio-file={self.encode_mpv_pipe_name}', # Add external audio file
                *rife_arg,#,format={self.output_pixel_format}',
                #  '--o=' + f"temp/stream.mpd", # Output file (implicitly overwrites)
                #  "--of=mp4",
                #  "--o=test.mp4",
                # '--o=' + playlist_path if output_hls else f"{self.output_dir}/out_{self.target_time}.mp4",
                # f'--ovc={self.args.video_codec}',
                # f'--ovcopts={ovcopts}', # Options for the video codec
                # f"--of={'hls' if output_hls else 'mp4'}",
                # f"--vf=format=fmt=yuv420p",  #important 
                # f"--vf=fps={out_fps}",
                # f'--ofopts={ofopts}', # Output format options
                '--input-ipc-server=' + self.encode_mpv_ipc_pipe_name,
                f"--msg-level=all={self.mpv_log_levels['encode']}",
            ]
            return cmd
        else:
                

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
                    "-pixel_format", self.output_pixel_format if self.using_interpolator else "bgr24",
                    "-video_size", f"{out_width}x{self.height}",
                    "-framerate", str(out_fps),
                    # "-thread_queue_size", "50000",
                    "-i", "-", 
                    "-map", "0:a:0", 
                    "-map", "1:v:0", 
                    "-c:v", self.args.video_codec,
                    "-pix_fmt", "yuv420p",  #important, otherwise will encode in unsupported pi
                    *video_opts,
                    "-g", f"{out_fps*self.ff_hls_time}",
                    "-keyint_min", f"{out_fps*self.ff_hls_time}",
                    "-r", str(out_fps),  # Output framerate
                    "-c:a", "aac",  # Audio codec (adjust as needed)
                    *hls_opts,
                    *ofopts,
                    "-y",  # Overwrite output file
                    playlist_path if output_hls else f"{self.output_dir}/out_{self.target_time}.mp4",
                    "-loglevel",  self.mpv_log_levels['encode']
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
                            # f"r={out_fps}",
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
                        f'--demuxer-rawvideo-fps={out_fps}',
                        "--aid=2",
                        f'--audio-file={self.encode_mpv_pipe_name}', # Add external audio file
                        #  '--o=' + f"temp/stream.mpd", # Output file (implicitly overwrites)
                        #  "--of=dash",
                        '--o=' + playlist_path if output_hls else f"{self.output_dir}/out_{self.target_time}.mp4",
                        f'--ovc={self.args.video_codec}',
                        f'--ovcopts={ovcopts}', # Options for the video codec
                        f"--of={'hls' if output_hls else 'mp4'}",
                        f"--vf=format=fmt=yuv420p",  #important 
                        f"--vf=fps={out_fps}",
                        f'--ofopts={ofopts}', # Output format options
                        '--input-ipc-server=' + self.encode_mpv_ipc_pipe_name,
                        f"--msg-level=all={self.mpv_log_levels['encode']}",
                    ]
                cmd_ = " ".join(cmd)
                print("\ncmd", cmd_, "\n")
            return cmd

    def interpolator_feeder(self):
        while 1:
            sbs = self.interpolate_input_queue.get()
            bgr_data = self.tensor_to_bgr_data(sbs)
            self.interpolate_process.stdin.write(bgr_data)
            
    def interpolator(self):
        
        out_width = self.get_output_width(self.width)
        
        int_frame_size = int(out_width * self.height * 1.5) if self.output_pixel_format == "yuv420p" else out_width * self.height * 3

        try:
            pipe_size = int_frame_size# 1024*1024*1
            self.interpolate_output_pipe_handle = win32pipe.CreateNamedPipe(
                self.interpolate_output_pipe_name,
                win32pipe.PIPE_ACCESS_INBOUND,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
                1,   pipe_size,   pipe_size,  0,   None  # Security attributes
            )
            
            print(f"Created Windows named pipe for output: {self.interpolate_output_pipe_name}")
        except Exception as e:
            print(f"Error creating Windows named pipe: {e}")
        
        buffer_frames = ":".join([str(e) for e in self.vapoursynth_buffer_frames])
        cmd = [
            self.mpv_bin,
            '-',  # Read from stdin
            # "--hwdec=auto-copy",
            # "--hwdec-codecs=all",
            # "--vo=gpu-next",
            '--demuxer=rawvideo',
            f'--demuxer-rawvideo-w={out_width}',
            f'--demuxer-rawvideo-h={self.height}',
            '--demuxer-rawvideo-mp-format=bgr24',
            # f'--demuxer-rawvideo-fps={self.fps}',
            f'--vf=vapoursynth=[{self.vsScriptPath}]:{buffer_frames},format={self.output_pixel_format}',
            '--of=rawvideo',  # Output raw video
            '--ovc=rawvideo',  # Raw video codec
            f'--o={self.interpolate_output_pipe_name}',  # Write to named pipe
            # "--vo=no",
            # "--ao=no",
            f"--input-ipc-server={self.interpolate_ipc_control_pipe}",
            f"--msg-level=all={self.mpv_log_levels['interpolate']}"
        ]
        
        try:
            print("Starting MPV process with stdin input and named pipe output...")
            
            self.running = True
            self.interpolate_process = subprocess.Popen(    cmd,    stdin=subprocess.PIPE)#bufsize=self.frame_size*10   )


            self.interpolate_started.set_result(True)
            print("Pipe reader thread waiting for connection...")
            win32pipe.ConnectNamedPipe(self.interpolate_output_pipe_handle, None)
            print("Pipe reader thread connected to output pipe")
            while self.running:
                try:
                    # Read processed frame from pipe
                    processed_data = read_frame_of_size(self.interpolate_output_pipe_handle, int_frame_size, int_frame_size)
                    self.encode_video_queue.put(processed_data)
                    # result, processed_data = win32file.ReadFile(self.output_pipe_handle, self.frame_size)
                    continue
                    
                    if len(processed_data) == self.frame_size:
                        # Convert bytes back to numpy array
                        if self.pf == "rgb24":
                            processed_frame = np.frombuffer(processed_data, dtype=np.uint8)
                            processed_frame = processed_frame.reshape((self.height, self.width, 3))
                        elif self.pf == "yuv420p":
                            processed_frame = np.frombuffer(processed_data, dtype=np.uint8)
                            processed_frame = processed_frame.reshape((self.height * 3 // 2, self.width))
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_YUV420p2RGB)
                        
                        if self.output_callback:
                            self.output_callback(processed_frame)
                        else:
                            self.interpolate_output_queue.put(processed_frame)#, block=False)
                            
                    else:
                        print(f"Incomplete frame received: {len(processed_data)}/{self.frame_size} bytes")
                        
                except pywintypes.error as e:
                    if e.winerror == 232:  # Pipe is closing
                        print("Pipe reader: Pipe is closing")
                        break
                    else:
                        print(f"Pipe reader error: {e}")
                        break
                except Exception as e:
                    print(f"Unexpected error in pipe reader: {e}")
                    break
                    
        except Exception as e:
            print(f"Error in pipe reader thread: {e}")
        finally:
            try:
                win32file.FlushFileBuffers(self.interpolate_output_pipe_handle)
                win32file.CloseHandle(self.interpolate_output_pipe_handle)
            except:   pass
            self.running = False
                
                
    def encoder(self):
        print("Encoder started")
        out_width = self.get_output_width(self.width)
        out_fps = self.get_output_fps()
        out_audio_pipe_size = self.audio_bytes_per_second*3 if self.use_ffmpeg_encoder else 1024*1024*1
        if self.using_interpolator:
            out_frame_size = int(out_width * self.height * 1.5) if self.output_pixel_format == "yuv420p" else out_width * self.height * 3
        else:
            out_frame_size = out_width * self.height * 3
        print("Encoder out frame size" , out_frame_size)
        
        self.pipe = win32pipe.CreateNamedPipe(
            self.encode_mpv_pipe_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            2, out_audio_pipe_size, out_audio_pipe_size, 0, None
        )
        print(f"Output audio named pipe created: {self.encode_mpv_pipe_name} with size {out_audio_pipe_size} ")
        
        

        try:

            cmd = self.get_encoder_cmd(out_width)
            
            # time.sleep(2)
            bufsize = out_frame_size*3 if self.use_ffmpeg_encoder else 1024*1024 #1920*1080 half sbs 48 fps
            #maybe bufsize should be out_frame_size * framerate
            # bufsize = int(out_frame_size  * round(out_fps))
            self.encode_process = subprocess.Popen( cmd, stdin=subprocess.PIPE,
                                                    bufsize= bufsize
                                                   )#out_frame_size*3 for interpolation x2
            
            win32pipe.ConnectNamedPipe(self.pipe, None)
            print("Pipe connected")

            header = generate_wav_header( self.audio_sample_rate, self.audio_bits_per_sample,self.audio_channels, 0x1fffffff )
            # with open("example.wav", "wb") as f:
            #     f.write(header)
            
            win32file.WriteFile(self.pipe, header)

            tot_audio_bytes_wr = 0

            def write_audio(src:ThreadSafeByteFIFO):
                emission, current_total = self.dec_accumulator.add_number(self.audio_dec)
                audio_s = self.audio_int+(emission)
                assert audio_s % self.audio_bytes_per_sample_and_channel == 0
                
                audio_data = src.get(audio_s)
                audio_written_bytes = win32file.WriteFile(self.pipe, audio_data)[1]
                return audio_written_bytes

            dummy_audio_data = ThreadSafeByteFIFO()
            dummy_audio_data.put(create_sine_wave_bytes(self.audio_bytes_per_second*3, self.audio_sample_rate))

            frames_written = 0
            ss =  int(self.audio_bytes_per_second*(1.5/(self.interpolation_multiplier if self.using_interpolator else 1))) 
            # ss = int(self.audio_bytes_per_second*1.8 )
            while tot_audio_bytes_wr < ss:
                wr  = write_audio(dummy_audio_data)
                tot_audio_bytes_wr+=wr
                # print("dummy a", wr, frames_written)
                frames_written+=1

            # remainder = tot_wr % self.audio_bytes_per_sample_and_channel
            # if remainder != 0:
            #     needed = self.audio_bytes_per_sample_and_channel - remainder
            #     f"{tot_wr} is not divisible by {self.audio_bytes_per_sample_and_channel}"
            #     tot_wr += win32file.WriteFile(self.pipe, bytearray(needed))[1]
            assert tot_audio_bytes_wr % 4 == 0

            dummy_image = bytearray([128] * out_frame_size)

            dummy_frames_number  = frames_written*(self.interpolation_multiplier if self.using_interpolator else 1)
            for x in range(dummy_frames_number):
                ret = self.encode_process.stdin.write(dummy_image)
                # print(f"{x} dummy v", ret)
            print("Encoder started")

            # de = deque(maxlen=150)
            fn = 0
            while not self.__stop_all:  
                if self.using_interpolator:
                    bgr24_data = self.encode_video_queue.get()
                else:
                    sbs = self.encode_video_queue.get()
                    bgr24_data = self.tensor_to_bgr_data(sbs)
                    # c.ct(1)

                bytes_written = 0
                total_bytes = len(bgr24_data)

                while bytes_written < total_bytes:
                    remaining_data = bgr24_data[bytes_written:]
                    written = self.encode_process.stdin.write(remaining_data)
                    bytes_written += written
                    
                if not self.using_interpolator or fn % self.interpolation_multiplier == 0:
                    audio_frame = self.decode_audio_queue.get()
                    audio_bytes_written = win32file.WriteFile(self.pipe, audio_frame)[1]
                    assert len(audio_frame) == audio_bytes_written
                fn += 1

                # de.append(audio_bytes_written)
                # print("dea", np.mean(de), self.audio_bytes_per_frame)
                
                # time.sleep(0.001) 

            print("Encoder Ended")

        except Exception as e:
            print(f"Encoder error: {e}")
            if self.encode_process:
                self.encode_process.kill()

    def tensor_to_bgr_data(self, sbs):
        sbs = (sbs * 255).to(torch.uint8)

        img_np = sbs.detach().cpu().numpy()

                # Permute dimensions from [C, H, W] to [H, W, C]
        img_np = np.transpose(img_np, (1, 2, 0))

                # Convert to uint8 if needed
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            bgr_data = (img_np * 255).astype(np.uint8)
        else:
            bgr_data = img_np

        bgr_data = np.ascontiguousarray(bgr_data)
        return bgr_data



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
        self.audio_thread = threading.Thread(target=self.audio_decoder, daemon=True)
        self.video_thread = threading.Thread(target=self.video_decoder, daemon=True)
        if  self.using_interpolator:
            self.interpolator_feeder_thread = threading.Thread(target=self.interpolator_feeder, daemon=True)
            self.interpolate_thread = threading.Thread(target=self.interpolator, daemon=True)
        self.encode_thread = threading.Thread(target=self.encoder, daemon=True)
        self.segment_thread = threading.Thread( target=self.check_segment_delta, daemon=True)
        
        self.audio_thread.start()
        self.video_thread.start()
        if self.using_interpolator:
            self.interpolator_feeder_thread.start()
            self.interpolate_thread.start()
            self.interpolate_started.result()
        self.encode_thread.start()
        self.segment_thread.start()

import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading

class LoggingHTTPRequestHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        print(f"Requested file: {self.path}")  # Log the requested path
        
        # If root path is requested, serve index.html instead of master.m3u8
        if self.path == '/' or self.path == '/index.html':
            self.path = '/index.html'  # This will serve index.html from the output_dir
        
        # Handle .ts files
        if self.path.endswith(".ts") and hasattr(self, "vp"):
            self.vp.last_req_seg_n = extract_n(self.path) or 0

        super().do_GET()

    def end_headers(self):
        # Add CORS headers if needed
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def start_http_server(output_dir, vp):
    # Change to the output directory so the server serves from there
    import sys
    os.__file__
    try:
        
        shutil.copy(os.path.join( os.path.dirname(os.path.abspath(__file__)),"index.html"), output_dir)
    except Exception as e:
        print("Error copying index", e)
    os.chdir(output_dir)

    # Start HTTP server in a separate thread
    server_address = ('0.0.0.0', vp.args.port)

    handler = LoggingHTTPRequestHandler
    handler.vp = vp

    httpd = HTTPServer(server_address, handler)  # Use our custom handler
    
    def run_server():
        print(f"Serving HLS stream at http://localhost:{vp.args.port}/")
        print(f"Make sure you have an index.html file in {output_dir}")
        httpd.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return httpd
