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

from .realtime_player_seek_gui import SeekBarApp
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

from .realtime_player_utils import *
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
        self.use_single_ffmpeg_decoder = True
        self.safe_audio_mode = False
        self.b_print_debug = False
        # default_level = "info" if self.args.output_mode == "hls_ffmpeg" else "status"
        default_level = "warning" if self.args.output_mode == "hls_ffmpeg" else "info"
        self.mpv_log_levels = {"video_decode": "error", "audio_decode": "error", "interpolate": "error", "encode": default_level }
        #mpv log level: fatal error warn info status v debug trace
        #ffmpeg log levels:  quiet panic fatal error warning info verbose debug trace 
        assert shutil.which(self.mpv_bin)

        # ytdlp_options = 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best'
        # self.ytdlp_video_options = 'bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/bestvideo' 
        # self.ytdlp_audio_options = 'bestaudio[ext=m4a]/bestaudio'
        self.using_ytdlp = is_url(self.input_file)

        pipe_id = str(uuid.uuid4())[:8]

        self.decode_audio_mpv_ipc_pipe_name = None# f'\\\\.\\pipe\\iw3_decode_audio_mpv_ipc_pipe___{pipe_id}'
        self.decode_video_mpv_ipc_pipe_name = None# f'\\\\.\\pipe\\iw3_decode_video_mpv_ipc_pipe___{pipe_id}'
        self.encode_mpv_ipc_pipe_name = f'\\\\.\\pipe\\iw3_encode_mpv_ipc_pipe___{pipe_id}'
        self.interpolate_ipc_control_pipe = f"\\\\.\\pipe\\iw3_rife_output_ipc_pipe___{pipe_id}"

        self.interpolate_output_pipe_name = f"\\\\.\\pipe\\iw3_rife_output_{pipe_id}"
        self.encode_mpv_pipe_name = f'\\\\.\\pipe\\iw3_encode_mpv_pipe__{pipe_id}'
        
        self.new_frames_flag = None
        self.keyframes = []
        self.decode_audio_mpv_proc = None
        self.decode_video_mpv_proc = None
        self.interpolate_process = None
        
        # self.dont_use_ytdlp_url = False
        if not self.using_ytdlp:
            cap = cv2.VideoCapture(self.input_file)
            video_info = int(cap.get(3)), int(cap.get(4)), cap.get(5)
            self.width, self.height, self.fps = video_info
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = frame_count / self.fps
            cap.release()
        else:
            self.yt_dlp_info = get_yt_dlp_otions(self.input_file, 1080)
            video_fmt = self.yt_dlp_info["best_video_fmt"]
            audio_fmt = self.yt_dlp_info["best_audio_fmt"]
            # info  = self.yt_dlp_info["info"]
            info = video_fmt.get("width", None),video_fmt.get("height", None),video_fmt.get("fps", None), self.yt_dlp_info["info"].get("duration",None)
            
            if not all(info):
                print("yt-dlp format doesn't contain all info, trying ffprobe")
                info = get_video_info(video_fmt["url"])
            if not all(info):
                # info = get_video_info(self.input_file)
                # self.dont_use_ytdlp_url = True
                print("ffprobe failed to get info") 
                assert False
            self.width, self.height, self.fps,self.video_duration = info
            
            # self.video_duration = self.yt_dlp_info["info"]["duration"]

            # info = download_url_to_temp(self.input_file, f'{audio_fmt["format_id"]}', True, True )
        self.sync_queue = queue.Queue()
        
        #####interpolation
        self.interpolate_conf_map = {
            "trtDll_path": "RIFE_PLAYER_TRT_DLL_PATH", 
            # "rootBin": "RIFE_PLAYER_ROOT_PATH",
            "mlrtScriptPath": "RIFE_PLAYER_MLRT_SCRIPT_PATH"
        }
        assert load_rife_config( getattr(args, "rife_config_path", r"C:\Users\%username%\source\repos vs\rifef_\rifef_\folders.ini"), self.interpolate_conf_map)
        self.interpolation_multiplier = args.int_mult 
        if args.auto_settings:
            if self.fps > 30:
                print("Fps > than 30, overriding interpolation multipler and model")
                self.interpolation_multiplier = 1
                self.args.depth_model = "Any_V2_S"
                
        self.vsScriptPath = os.getenv("vsScriptPath")
        os.environ["RIFE_PLAYER_MULTIPLIER"] = str(self.interpolation_multiplier)
        os.environ["vs_output_pixel_format"] = self.output_pixel_format
        self.using_interpolator = args.output_mode == "hls_ffmpeg" and self.interpolation_multiplier > 1
        self.vapoursynth_buffer_frames = (4,4)
        self.interpolate_started = Future()
        #####
        
        # single ffmpeg decoder
        self.ffmpeg_decoder_audio_pipe = f'\\\\.\\pipe\\iw3_ffmpeg_audio_decode_pipe___{pipe_id}'
        self.ffmpeg_decoder_video_pipe = f'\\\\.\\pipe\\iw3_ffmpeg_video_decode_pipe___{pipe_id}'
        self.ffmpeg_decoder_process = None
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
        self.audio_int = round(self.audio_bytes_per_frame - self.audio_dec)
        # self.last_extra_audio_frame = 0#time.time()
        
        if self.safe_audio_mode:
            t = time.time()
            self.audio_buffer_cur_pos =  ThreadSafeValue[int](0)
            self.audio_buffer =  load_audio_to_pcm16(self.input_file, self.audio_sample_rate, self.audio_channels)
            print(f"Getting raw audio data took {time.time() -t:4.3f} secs, size: {len(self.audio_buffer)/(1000*1000):4.3f} MB")
        ####
        
        round_fps = round(out_fps)
        # audio_buffer_frames_n = self.audio_sample_rate / round_fps
        encode_queue_size = 1
        self.decode_audio_queue = queue.Queue(maxsize=round(round_fps*1))
        self.decode_video_queue = queue.Queue(maxsize=round_fps-encode_queue_size*0)
        self.encode_video_queue = queue.Queue(maxsize=encode_queue_size) #1 because otherwise interpolation will cause audio shift, bigger number causes desync with ytdlp also
        # self.interpolate_output_queue = queue.Queue(maxsize=10)
        self.interpolate_input_queue =  queue.Queue(maxsize=10)
        
        # self.audio_buffer = ThreadSafeByteFIFO()

        self.rgb_video_frame_size = self.width*self.height*3
        self.audio_thread = threading.Thread(target=lambda: 1)
        self.video_thread = threading.Thread(target=lambda: 1)
        self.interpolate_thread = threading.Thread(target=lambda: 1)
        self.pipe = None
        self.interpolate_output_pipe_handle = None

        self.seeking_flag = ThreadSafeValue[bool](False)
        # self.queue_audio_drop_frame = ThreadSafeValue[int](0)
        self.__seek_start_time = 0
        self.__stop_all = False
        self._last_playback_time = 0
        self.__decoder_frame_number = 0
        
        self.decoded_audio_frames_n = ThreadSafeValue[int](0)
        self.decoded_video_frames_n = ThreadSafeValue[int](0)
        
        #   yt-dlp -f [format_code] -g [URL]
        self.keyframes =  get_keyframes(self.input_file) if not self.using_ytdlp else []#self.input_file if not self.using_ytdlp else video_fmt["url"])


        # self.output_dir =os.path.join(os.path.expandvars("%APPDATA%"), output_dir)# os.path.abspath(output_dir)
        self.output_dir =  os.path.abspath(output_dir)
        self.init_dir()

        print("Outdir", self.output_dir)

        # # self.pixel_format = get_video_pixel_format_ffprobe(self.input_file)

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

        def toggle_print_debug(): 
            self.print_debug()
            self.b_print_debug = not self.b_print_debug

        bindings = [["window + f11", None, lambda:self.seek(get_number()), False],
                    ["window + f2", None, toggle_print_debug, False],
                    ["window + f3", None, self.stop_all, False]
                    ]
        register_hotkeys(bindings)
        start_checking_hotkeys()
        
        def test():
            sid = 1
            while 1:
                time.sleep(10)
                perc = random.uniform(0, 80)
                print("----> seeking", perc, " ---")
                self.seek(perc)
                # set_track_by_id("sub", sid, self.decode_video_mpv_ipc_pipe_name)
                # sid+=1
        #threading.Thread(target=test, daemon=True).start()
        httpd = start_http_server(self.output_dir, self)
    
        def start_seekbar():
            self.seek_bar_gui = SeekBarApp(port=self.args.port,
                                           get_info_fun=lambda: {"playback_time": self._last_playback_time,
                                                                 "duration": self.video_duration,
                                                                 "lambda": True},
                                           seek_fun=lambda perc: self.seek(perc),
                                           seek_relative_fun = lambda change: self.seek_relative(change)
                                           )
            
        threading.Thread(target=start_seekbar, daemon=True).start()
        
    def seek_relative(self, change_seconds):
        seek_time = self._last_playback_time  + change_seconds
        if len(self.keyframes):
            self.seek_to_closest_keyframe(seek_time)
            # if  self.using_ytdlp:
        else:
            self.__seek(seek_time)
            
    def seek(self, percentage):
        if len(self.keyframes): #means we have a file, either downloaded or not
            self.seek_perc_at_keyframe(percentage)
        else:
        # if self.using_ytdlp:  and not len(self.keyframes):
            seek_time = (percentage/100)*self.video_duration
            self.__seek(seek_time)
    
    def get_output_fps(self):
        if self.using_interpolator:
            return self.fps * self.interpolation_multiplier 
        else:
            return self.fps
    def has_stopped(self): return self.__stop_all
    
    def print_debug(self):
        str_ =  f"q->  da: {self.decode_audio_queue.qsize():3d} dv: {self.decode_video_queue.qsize():3d} "
        str_ += f"ii: {self.interpolate_input_queue.qsize():3d} ev: {self.encode_video_queue.qsize():3d} "
        # str_ += f"ev: {self.encode_video_queue.qsize()} "
        str_ += f"| v: {self.decoded_video_frames_n} a: {self.decoded_audio_frames_n} | "
        self.c.tick(str_)

    def check_segment_delta(self):
        while not self.__stop_all:
            if self.args.output_mode == "hls_ffmpeg":
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
                
    def check_playback_time(self):
        while not self.__stop_all:
            # if not self.seeking_flag.get() and (self.decode_video_mpv_ipc_pipe_name or self.use_single_ffmpeg_decoder):
            if self.decode_video_mpv_ipc_pipe_name or self.use_single_ffmpeg_decoder:
                self._last_playback_time = self.get_playback_time() or 1
            for x in range(10):
                time.sleep(0.1)
                if self.__stop_all: break

    def get_last_peek_sizes(self):
        return {e: self.last_peeked_pipe_sizes[getattr(self, e)]/(1000*1000) for e in ["audio_pipe", "video_pipe", "out_audio_pipe", "out_video_pipe"]}

    def init_dir(self):
        def try_del():
            try: shutil.rmtree(self.output_dir)
            except Exception as e:  print("Error cleaning up:", e)
                
        while os.path.isdir(self.output_dir):#or len(os.listdir(self.output_dir)):
            print("Wait for delete")
            try_del()
            time.sleep(0.01)
        os.makedirs(self.output_dir, exist_ok=True)

    def seek_to_closest_keyframe(self, target_time):
        if target_time is None:
            print("no time provided")
            return
    
        closest_keyframe = min(self.keyframes, key=lambda x: abs(x - target_time))
        closest_index = self.keyframes.index(closest_keyframe)
        
        print(f"Target time: {target_time}, Closest keyframe: {closest_keyframe}, Index: {closest_index}")
        self.__seek(closest_keyframe)

    def get_keyframe_at_perc(self, perc):
        def get_element_at_percentage(percentage, lst):
            factor = len(lst) * percentage / 100
            index = round(factor)
            index = max(0, min(index, len(lst) - 1))
            return lst[index], index#, factor, index

        target_time, index = get_element_at_percentage(perc, self.keyframes)
        return target_time, index
    
    def seek_perc_at_keyframe(self,perc ):
        if perc == None :
            print("no percentage provided")
            return 
        target_time, index = self.get_keyframe_at_perc(perc)
        print("%", perc,  "target_time", target_time, "index", index)
        self.__seek(target_time)

    def get_playback_time(self): 
        if self.use_single_ffmpeg_decoder:
            return (self.__seek_start_time or 0) + self.__decoder_frame_number / self.fps
        else:
            return get_property_partial("playback-time", self.decode_video_mpv_ipc_pipe_name)

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
        if self.use_single_ffmpeg_decoder:
            self.stop_ffmpeg_decoder()
        else:
            if not self.safe_audio_mode:
                self.quit_audio_decode_mpv()
            self.quit_video_decode_mpv()
            


        for qq in [self.decode_audio_queue, self.decode_video_queue, self.encode_video_queue, self.interpolate_input_queue]:
            while qq.qsize(): qq.get()
        
        self.quit_encode_mpv()
        if self.using_interpolator: self.stop_interpolator()
        
        for t_name in ["encode_thread", "audio_thread","video_thread", "interpolate_thread", "segment_thread"]:
            thread_ : threading.Thread = getattr(self, t_name, None)
            if thread_ and thread_.is_alive():
                print(f"Joining {t_name} ")
                thread_.join()
        print("---> Everthing stopped")

    def __seek(self, time_):
        sl= 0.1
        self.__seek_start_time = time_
        # self.seeking_flag.set(True)
        self.new_frames_flag = Future()
        if self.use_single_ffmpeg_decoder:
            pause_unpause("pause", self.encode_mpv_ipc_pipe_name)
            
            for q in self.decode_video_queue,self.decode_audio_queue: 
                while not q.full(): 
                    time.sleep(0.1)
                    print("waiting for queues to be full",self.decode_video_queue.qsize(),self.decode_audio_queue.qsize() )        
            print("queues are full") 
            self.stop_ffmpeg_decoder()
            # time.sleep(1)
            # print("getting 1")
            self.decode_audio_queue.get()
            self.decode_video_queue.get()
            # time.sleep(1)
            time.sleep(0.1)
            self.audio_thread.join()
            self.video_thread.join()
            self.ffmpeg_decoder_thread = threading.Thread(target=self.ffmpeg_decoder, daemon=True)
            self.ffmpeg_decoder_thread.start()   
            self.new_frames_flag.result()
            # self.new_frames_flag = None
            pause_unpause("unpause", self.encode_mpv_ipc_pipe_name)

        else:
            # for p in self.decode_video_mpv_ipc_pipe_name,self.decode_audio_mpv_ipc_pipe_name: pause_unpause("pause", p)
            # time.sleep(0.1)
            pipes = self.decode_video_mpv_ipc_pipe_name,#self.decode_audio_mpv_ipc_pipe_name,
            


            print("killing audio decode mpv..")
            if not self.safe_audio_mode:
                self.quit_audio_decode_mpv()
                self.audio_thread.join()
            else:
                position_in_bytes = self.__seek_start_time * self.audio_sample_rate * self.audio_channels * self.audio_bytes_per_sample
                position_in_bytes  =  (position_in_bytes // self.audio_bytes_per_sample_and_channel) * self.audio_bytes_per_sample_and_channel
                self.audio_buffer_cur_pos.set(round(position_in_bytes))
                
            
            if self.restart_mpv_decode_on_seek:
                print("killing video decode mpv..")
                self.quit_video_decode_mpv()
                self.video_thread.join()
            else:
                print("Pausing decoders..")
                # self.seeking_flag.set(True)
                for p in pipes:
                    pause_unpause("pause", p)
                    
                #time.sleep(sl)

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
            
            if self.restart_mpv_decode_on_seek:
                self.video_thread = threading.Thread(target=self.video_decoder, daemon=True)
                self.video_thread.start()
            if not self.safe_audio_mode:
                self.audio_thread = threading.Thread(target=self.audio_decoder, daemon=True)
                self.audio_thread.start()
            # threading.Timer(.1, lambda: self.seeking_flag.set(False)).start()
        
    def quit_audio_decode_mpv(self):
        res = send_cmd({  "command": ["quit" ]  }, self.decode_audio_mpv_ipc_pipe_name )
        try: self.decode_audio_mpv_proc.stdout.close()
        except Exception as e:print("Error closing decode audio mpv stdout handle", e)

    def quit_video_decode_mpv(self):
        res = send_cmd({  "command": ["quit" ]  }, self.decode_video_mpv_ipc_pipe_name )
        try: self.decode_video_mpv_proc.stdout.close()
        except Exception as e:print("Error closing decode video mpv stdout handle", e)
        
    def quit_encode_mpv(self):
        self.decode_audio_queue.put(None)
        self.decode_video_queue.put(None)
        res = send_cmd({  "command": ["quit" ]  }, self.encode_mpv_ipc_pipe_name )
        time.sleep(.1)
        def close_stdin():
            try:  self.encode_process.stdin.close()
            except Exception as e:print("Error closing encode stdin handle", e)
        def close_pipe():
            try: win32file.CloseHandle(self.pipe)
            except Exception as e:print("errror", e)
        
        threading.Timer(.1, close_stdin ).start()
        t = threading.Timer(.2, close_pipe )
        t.start()
        t.join()
        self.encode_process.wait()
        for q in [self.decode_audio_queue,self.decode_video_queue]: 
            while q.qsize(): q.get()

    def stop_interpolator(self):
        send_cmd({  "command": ["quit" ]  }, self.interpolate_ipc_control_pipe )
        self.interpolate_input_queue.put(None)
        if self.interpolate_process:
            try:
                def close_stdin():
                    try: self.interpolate_process.stdin.close()
                    except Exception as e:print("Error closing encode stdin handle", e)
                def close_pipe():
                    try: win32file.CloseHandle(self.interpolate_output_pipe_handle)
                    except Exception as e:print("errror", e)
                    
                threading.Timer(.1, close_stdin ).start()
                t = threading.Timer(.2, close_pipe )
                t.start()
                t.join()

                
                # self.interpolate_process.terminate()
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

    def safe_audio_decoder(self):

        try:
            print("safe_audio_decoder started ")

            while not self.__stop_all:
                # self.sync_queue.get()
                emission, current_total = self.dec_accumulator.add_number(self.audio_dec)
                audio_s = self.audio_int+(emission)
                assert audio_s % self.audio_bytes_per_sample_and_channel == 0

                cur_pos = self.audio_buffer_cur_pos.get()
                chunk = self.audio_buffer[cur_pos: cur_pos+audio_s]
                cur_pos+= audio_s # read_frame_of_size(self.decode_audio_mpv_proc.stdout, audio_s, audio_s)
                self.audio_buffer_cur_pos.set(cur_pos)
                if not chunk:
                        print("No data from mpv audio decoder, end of file?")
                        break
                le = len(chunk)
                assert le == audio_s
                self.decoded_audio_frames_n.increment()
                self.decode_audio_queue.put(chunk)
                time.sleep(0.001)

            # print(f"Audio: Received {len(self.audio_buffer)/1_000_000} MB {self.decode_audio_mpv_proc.poll()}")

        except Exception as e:
            print(f"Error safe_audio_decoder loop: {e}")
            return None
        finally:
            print("Mpv audio decoder ended")

    def audio_decoder(self):
        """Spawn MPV instance for audio decoding."""

        pipe_id = str(uuid.uuid4())[:8]
        self.decode_audio_mpv_ipc_pipe_name = f'\\\\.\\pipe\\iw3_decode_audio_mpv_ipc_pipe___{pipe_id}'
        input_ = self.yt_dlp_info["best_audio_fmt"]["url"] if self.using_ytdlp and 0 else self.input_file
        
        audio_args =  [
            self.mpv_bin,
            input_,
            '--no-config',
             f"--hr-seek={'yes' if self.using_ytdlp or 1 else 'no'}",
            # f"--start={self.__seek_start_time}",
            "--pause=yes",
            # "--oacopts=ar=44100,ac=2",  # Sample rate: 44.1kHz, Channels: 2 (stereo)
            f"--af=format=srate={self.audio_sample_rate}",  # ,ac={self.audio_channels}",
            "--of=s16le",
            "--oac=pcm_s16le",
            "-o",
            "-",  # Output to stdout
            "--input-ipc-server=" + self.decode_audio_mpv_ipc_pipe_name,
            f"--msg-level=all={self.mpv_log_levels['audio_decode']}",
            *([f'--ytdl-format={self.yt_dlp_info["best_audio_fmt"]["format_id"]}', ] if self.using_ytdlp else [])

        ]
        try:
            print("Starting audio MPV instance...")
            self.decode_audio_mpv_proc = subprocess.Popen(
                audio_args,
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,,
                bufsize=math.ceil(self.audio_bytes_per_frame)
            )
            self.__ipc_mpv_seek(self.decode_audio_mpv_ipc_pipe_name)

            print("Audio MPV instance started successfully!")
        except FileNotFoundError:
            print("Error: MPV not found. Please install MPV and ensure it's in your PATH")
            return None
        
        self.audio_decoder_loop()
        
    def audio_decoder_loop(self, src):
        src = src or self.decode_audio_mpv_proc.stdout

        try:
            while not self.__stop_all:
                # self.sync_queue.get()
                emission, current_total = self.dec_accumulator.add_number(self.audio_dec)
                audio_s = self.audio_int+(emission)
                assert audio_s % self.audio_bytes_per_sample_and_channel == 0

                chunk = read_frame_of_size(src, audio_s, audio_s)
                if not chunk:
                        print("No data from mpv audio decoder, end of file?")
                        break
                le = len(chunk)
                assert le == audio_s

                self.decoded_audio_frames_n.increment()
                self.decode_audio_queue.put(chunk)
                # print("audio")
                time.sleep(0.001)

            # print(f"Audio: Received {len(self.audio_buffer)/1_000_000} MB {self.decode_audio_mpv_proc.poll()}")

        except Exception as e:
            print(f"Error MPV audio loop: {e}")
            return None
        finally:
            print("Mpv audio decoder ended")
    
    def video_decoder(self):
        
        pipe_id = str(uuid.uuid4())[:8]

        self.decode_video_mpv_ipc_pipe_name =  f'\\\\.\\pipe\\iw3_decode_video_mpv_ipc_pipe___{pipe_id}'
        input_ = self.yt_dlp_info["best_video_fmt"]["url"] if self.using_ytdlp and 0 else self.input_file

        video_args = [ 
            self.mpv_bin, input_, '--no-config',
            # f"--start={self.__seek_start_time if self.restart_mpv_decode_on_seek else 0}",  
            "--pause=yes",
            *([f"--sid={self.subtitle_id}", "--vf=sub"] if self.subtitle_id != None else  []),
             f"--hr-seek={'yes' if self.using_ytdlp else 'no'}",
            "--ovc=rawvideo",
            f"--vf=format=fmt=rgb24",  # ,fps={self.fps}",
            "--of=rawvideo",  
            "--input-ipc-server=" + self.decode_video_mpv_ipc_pipe_name,
            "-o", "-",
            f"--msg-level=all={self.mpv_log_levels['video_decode']}",
            *([f'--ytdl-format={self.yt_dlp_info["best_video_fmt"]["format_id"]}'] if self.using_ytdlp else [])
        ]
        
        # video_args = [
        #     self.ffmpeg_bin,
        #     '-i', input_,
        #     '-ss', str(self.__seek_start_time) if not self.restart_mpv_decode_on_seek else '0',
        #     '-an',  # no audio
        #     '-sn',  # no subtitles (we'll handle them separately)
        #     '-vcodec', 'rawvideo',
        #     '-pix_fmt', 'rgb24',
        #     '-f', 'rawvideo',
        #     '-',
        #     '-loglevel', self.ffmpeg_log_levels['video_decode'],
        # ]
        # # Add subtitle handling if needed
        # if self.subtitle_id != None:
        #     video_args.extend([
        #         '-vf', f'subtitles={input_}:si={self.subtitle_id}'
        #     ])

        # # Add ytdlp format selection if using ytdlp
        # if self.using_ytdlp:
        #     video_args.extend([
        #         '-format_id', self.yt_dlp_info["best_video_fmt"]["format_id"]
        #     ])
        
        print("Starting video MPV instance...")
        self.decode_video_mpv_proc = subprocess.Popen(
            video_args,
            stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            bufsize=self.rgb_video_frame_size
        )
        self.__ipc_mpv_seek(self.decode_video_mpv_ipc_pipe_name, 0.1) #mpv will seek backwards to the closest keyframe so add +0.1 to the time
        
        print("Video MPV instance started successfully!")
        
        self.video_decoder_loop()

    def video_decoder_loop(self, src=None):
        src = src or self.decode_video_mpv_proc.stdout
        try:

            while not self.__stop_all:
                while self.is_paused:
                    time.sleep(0.1)
                self.signal_audio_thread()
                if self.new_frames_flag != None and not self.new_frames_flag.done():
                    # for x in range(3):
                    #     time.sleep(1)
                    #     print("Setting self.new_frames_flag result", self.decode_audio_queue.qsize(),self.decode_video_queue.qsize())
                    self.new_frames_flag.set_result(1)

                data = read_frame_of_size(src, self.rgb_video_frame_size, self.rgb_video_frame_size )
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

                self.__decoder_frame_number +=1
                self.decoded_video_frames_n.increment()
                self.decode_video_queue.put(frame)
                
                # if not self.safe_audio_mode:

        except Exception as e:
            print(f"Video read error: {e}")

    def ffmpeg_decoder(self):
        size = self.rgb_video_frame_size
        self.audio_pipe = win32pipe.CreateNamedPipe(
            self.ffmpeg_decoder_audio_pipe,
            win32pipe.PIPE_ACCESS_DUPLEX, #| win32file.FILE_FLAG_OVERLAPPED,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
            2, size, size, 0, None
        )
        
        self.video_pipe = win32pipe.CreateNamedPipe(
            self.ffmpeg_decoder_video_pipe,
            win32pipe.PIPE_ACCESS_DUPLEX, #| win32file.FILE_FLAG_OVERLAPPED,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
            2, size, size, 0, None
        )
        
        video_input = self.yt_dlp_info["best_video_fmt"]["url"] if self.using_ytdlp else self.input_file
        audio_input = self.yt_dlp_info["best_audio_fmt"]["url"]  if self.using_ytdlp and self.yt_dlp_info["best_audio_fmt"] else None#self.input_file

        codec_map = { 16: 'pcm_s16le', 24: 'pcm_s24le',  32: 'pcm_s32le' }
        seek_time = self.__seek_start_time+.1
        
        vf_video_args = []
        
        subtitle_file = None
        if self.args.subtitle != None:
            if os.path.isfile(self.args.subtitle):
                subtitle_file = self.args.subtitle
            else:
                subtitle_file = extract_subtitles(self.input_file, stream_index=self.args.subtitle )
                
            if os.path.isfile(subtitle_file):
                shutil.copy(subtitle_file,  os.path.join(os.getcwd(), "subtitle.srt" ))
                vf_video_args.append(f'subtitles=subtitle.srt')
            else:
                print("Failed to get subtitle with id from file", self.args.subtitle)
               
        v_seek_args = ["-ss",  str(seek_time) ] if self.__seek_start_time != 0 else []
        v_seek_args_2 = []
        if subtitle_file and self.__seek_start_time != 0:
            v_seek_args_2 = v_seek_args[:]
            v_seek_args.insert(2, '-copyts')
        
        audio_input_args = ( (["-ss",  self.__seek_start_time ] if self.__seek_start_time != 0 else []) + ['-i', audio_input]) if audio_input else []
        
        if len(vf_video_args): vf_video_args.insert(0, "-vf")
        
        ffmpeg_cmd = [
            'ffmpeg',
            # "-t", "10",
            *audio_input_args,
            *v_seek_args,
            '-i', video_input,
            *v_seek_args_2,
            
            '-map', f'{1 if audio_input else 0}:v',  # Process video only
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-f', 'rawvideo', '-y', 
            *vf_video_args,
            self.ffmpeg_decoder_video_pipe,
            
            '-map', '0:a',  # Process audio only
            '-acodec', codec_map[self.audio_bits_per_sample],
            '-ac', str(self.audio_channels),
            '-ar', str(self.audio_sample_rate),
            '-f', 's16le', '-y',
            self.ffmpeg_decoder_audio_pipe,
            
            "-loglevel",  self.mpv_log_levels["video_decode"]
        ]
        cmd_str = " ".join(ffmpeg_cmd)
        print("Ffmpeg decoder cmd\n",cmd_str, "\n")
        
        print("Starting ffmpeg decoder instance...")
        self.ffmpeg_decoder_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            # bufsize=self.rgb_video_frame_size
        )
        win32pipe.ConnectNamedPipe(self.video_pipe, None)
        win32pipe.ConnectNamedPipe(self.audio_pipe, None)
        self.__decoder_frame_number = 0
        
        self.video_thread = threading.Thread(target=self.video_decoder_loop, args=(self.video_pipe,), daemon=True)
        self.audio_thread = threading.Thread(target=self.audio_decoder_loop, args=(self.audio_pipe,), daemon=True)
        self.video_thread.start()
        self.audio_thread.start()
        # self.video_thread.join()
        # self.audio_thread.join()
              
    def stop_ffmpeg_decoder(self):
        print("Stopping ffmpeg decoder")
        for pp in self.video_pipe, self.audio_pipe:
            def close_ffmpeg_decode_task(pipe = pp):
                win32file.CloseHandle(pipe)
            threading.Thread(target=close_ffmpeg_decode_task, daemon=True).start()
        self.ffmpeg_decoder_process.stdin.write(b"q\n")
        self.ffmpeg_decoder_process.stdin.flush()
        self.ffmpeg_decoder_thread.join()

    def __ipc_mpv_seek(self, pipe, offset =0):
        while 1:
            try:
                time.sleep(0.1)
                seek_absolute(self.__seek_start_time+offset, pipe)
                time.sleep(0.1)
                pause_unpause("unpause", pipe)
                break
            except  Exception as e:
                print("Error seeking to start position:", e)

    def signal_audio_thread(self):
        de =  self.decoded_video_frames_n.get()-self.decoded_audio_frames_n.get()
        self.sync_queue.put(1)
        if de > 2 and self.sync_queue.qsize() < 3:
            self.sync_queue.put(1)
                #  if de > 2 and  time.time() - self.last_extra_audio_frame > 0.1:
                #     self.last_extra_audio_frame = time.time()
                #     for x in range(de+1):
                #         self.sync_queue.put(1)
            print("\r---- sync queue", self.sync_queue.qsize(), " ----", end="")
        # amount = self.interpolation_multiplier if self.interpolation_multiplier > 1 else 1
        # if  1 or self.sync_queue.qsize() < 3:
        # for x in range(de if de > 2 else 1):
        #     self.sync_queue.put(1)
        #     if x > 0:
        #         print("sync queue", x," de:", de, "qs:",self.sync_queue.qsize(), " --")

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
                #"--hwdec=auto-copy",
                #"--hwdec-codecs=all",
                #"--vo=gpu-next",
                "--profile=fast",
                "--video-sync=display-resample",
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
        while not self.__stop_all:
            sbs = self.interpolate_input_queue.get()
            if type(sbs) != torch.Tensor :break
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
            
            bufsize = out_frame_size*3 if self.use_ffmpeg_encoder else 1024*1024 #1920*1080 half sbs 48 fps
            #maybe bufsize should be out_frame_size * framerate
            # bufsize = int(out_frame_size  * round(out_fps))
            if self.args.output_mode == "local_mpv":
                my_env = os.environ.copy()
                target_env = os.getenv('pythonPath'.upper())
                my_env["PATH"] = f"{target_env};{my_env['PATH']}"
            else: my_env = None
            
            self.encode_process = subprocess.Popen( cmd, stdin=subprocess.PIPE,
                                                    bufsize= bufsize, env=my_env  )#out_frame_size*3 for interpolation x2
            
            win32pipe.ConnectNamedPipe(self.pipe, None)
            print("Pipe connected")

            header = generate_wav_header( self.audio_sample_rate, self.audio_bits_per_sample,self.audio_channels, 
                                         0x1fffffff)#   # 0x1fffffff otherwise it stops
            # with open("example.wav", "wb") as f: f.write(header)
            
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
                # print("v")
                    
                if not self.using_interpolator or fn % self.interpolation_multiplier == 0:
                    # self.signal_audio_thread()
                    # if not self.safe_audio_mode:
                    audio_frame = self.decode_audio_queue.get()
                    # else:
                    #     emission, current_total = self.dec_accumulator.add_number(self.audio_dec)
                    #     audio_s = self.audio_int+(emission)
                    #     assert audio_s % self.audio_bytes_per_sample_and_channel == 0
                    #     audio_frame = self.audio_buffer[self.audio_buffer_cur_pos:self.audio_buffer_cur_pos+audio_s]
                    #     self.audio_buffer_cur_pos += audio_s
                    
                    audio_bytes_written = win32file.WriteFile(self.pipe, audio_frame)[1]
                    assert len(audio_frame) == audio_bytes_written
                    # print("a", audio_bytes_written)
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
        if self.use_single_ffmpeg_decoder:
            self.ffmpeg_decoder_thread = threading.Thread(target=self.ffmpeg_decoder, daemon=True)
            self.ffmpeg_decoder_thread.start()
        else:
            self.audio_thread = threading.Thread(target=self.audio_decoder if not self.safe_audio_mode else self.safe_audio_decoder, daemon=True)
            self.video_thread = threading.Thread(target=self.video_decoder, daemon=True)
            self.audio_thread.start()
            self.video_thread.start()
            
        threading.Thread( target=self.check_playback_time, daemon=True).start()

        if self.using_interpolator:
            self.interpolator_feeder_thread = threading.Thread(target=self.interpolator_feeder, daemon=True)
            self.interpolate_thread = threading.Thread(target=self.interpolator, daemon=True)
            self.interpolator_feeder_thread.start()
            self.interpolate_thread.start()
            self.interpolate_started.result()
            
        self.encode_thread = threading.Thread(target=self.encoder, daemon=True)
        self.encode_thread.start()
            
        if self.args.output_mode == "hls_ffmpeg":
            self.segment_thread = threading.Thread( target=self.check_segment_delta, daemon=True)
            self.segment_thread.start()


from http.server import SimpleHTTPRequestHandler, HTTPServer


class LoggingHTTPRequestHandler(SimpleHTTPRequestHandler):

    def do_GET(self):

        if self.path == '/api/info':
            self.handle_info_api()
            return
        
        print(f"Requested file: {self.path}")  # Log the requested path
        
        if self.path == '/' or self.path == '/index.html':
            self.path = '/index.html'  # This will serve index.html from the output_dir
        
        if self.path.endswith(".ts") and hasattr(self, "vp"):
            self.vp.last_req_seg_n = extract_n(self.path) or 0

        super().do_GET()

    def do_POST(self):
        if self.path == '/api/seek':
            self.handle_seek_api()
            return
        
        self.send_error(404, "POST endpoint not found")

    def handle_info_api(self):
        try:
            response_data = {
                "playback_time" : self.vp._last_playback_time,
                "duration": self.vp.video_duration
            }
            self.send_json_response(response_data)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status_code=500)

    def handle_seek_api(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            
            post_data = self.rfile.read(content_length)
            
            data = json.loads(post_data.decode('utf-8'))
            
            seek_position = float(data.get('position', 0))
            
            if not 0 <= seek_position <= 100:
                self.send_json_response(
                    {"error": "Seek position must be between 0 and 100"}, 
                    status_code=400
                )
                return
                        
            self.vp.seek(round(seek_position))
            response_data = {
                "success": True, 
                "message": f"Seeked to {seek_position}%",
                "position": seek_position
            }
            self.send_json_response(response_data, status_code=501)

            
        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON data"}, status_code=400)
        except ValueError:
            self.send_json_response({"error": "Invalid position value"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status_code=500)

    def send_json_response(self, data, status_code=200):
        """Helper method to send JSON responses"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Convert data to JSON and encode as bytes
        json_data = json.dumps(data, indent=2)
        self.wfile.write(json_data.encode('utf-8'))

    def end_headers(self):
        # Add CORS headers if needed
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_http_server(output_dir, vp):
    try:
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), output_dir)
    except Exception as e:
        print("Error copying index", e)
    os.chdir(output_dir)

    server_address = ('0.0.0.0', vp.args.port)

    handler = LoggingHTTPRequestHandler
    handler.vp = vp

    httpd = HTTPServer(server_address, handler)  # Use our custom handler
    
    def run_server():
        print(f"Serving HLS stream at http://localhost:{vp.args.port}/")
        print(f"API endpoints: http://localhost:{vp.args.port}/api/status")
        print(f"Make sure you have an index.html file in {output_dir}")
        httpd.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return httpd