
import os
import shutil
import sys
import threading
import time
from .utils import (
    init_win32,
    set_state_args
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
from .realtime_player_process import HLSEncoder
from performanceTimer import Counter

TORCH_VERSION = Version(torch.__version__)
ENABLE_GPU_JPEG = (TORCH_VERSION.major, TORCH_VERSION.minor) >= (2, 7)
TORCH_NUM_THREADS = torch.get_num_threads()

from .utils import init_num_threads, get_local_address, is_private_address


import tempfile
from urllib.parse import urlparse
import yt_dlp

def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def download_url_to_temp(url, cleanup_on_failure=True):
    temp_dir = os.path.join( tempfile.gettempdir(), "iw3_downloads")# mkdtemp()
    
    try:
        ydl_opts = {
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': False,
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            downloaded_file = ydl.prepare_filename(info)
            
            return {
                'success': True,
                'file_path': downloaded_file,
                'temp_dir': temp_dir,
                'info': info
            }
            
    except Exception as e:
        if cleanup_on_failure:
            try:  shutil.rmtree(temp_dir)
            except:   pass
        
        return {
            'success': False,
            'file_path': None,
            'temp_dir': temp_dir if not cleanup_on_failure else None,
            'error': str(e)
        }

        
def iw3_desktop_main_hls(args):
    
    if is_url(args.input):
        file_path = download_url_to_temp(args.input, False)
        if file_path:
            args.input = file_path["file_path"]
            print(f"Downloaded file: {file_path}")
        else:
            print("Download failed")
    else:
        print("Not a valid URL")
        if not args.input or  not path.isfile(args.input):
            print("File not found:", args.input)
            exit()
        
    init_num_threads(args.gpu[0])
    c = Counter()


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

    vp = HLSEncoder(args.input, args.segment_folder, args=args)
    vp.start()
    
    def loop():
        print("Waiting for input...")
        line = sys.stdin.readline()
        print(f"Received: {line.strip()}")
        if line == "q" or line == "q\n" or line == "q\n\r":
            print("------> Stopping all <--------")
            vp.stop_all()
    
    threading.Thread(target=loop, daemon=True).start()
    
    try:
        if args.compile:
            depth_model.compile()
            if side_model is not None and not isinstance(side_model, DeviceSwitchInference):
                side_model = compile_model(side_model)
        count = 0

        output_queue = vp.interpolate_input_queue if  vp.using_interpolator else vp.encode_video_queue
        
        while not vp.has_stopped():
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

def create_parser():

    parser = IW3U.create_parser(required_true=False)
    parser.add_argument("--port", type=int, default=8123,
                        help="HTTP listen port")

    parser.add_argument("--full-sbs", action="store_true", help="Use Full SBS for Pico4")
    # parser.add_argument("--input_file", type=str, help="input_file")
    parser.add_argument("--segment_folder", type=str, help="output for the video segment files ", default="hls_out")
    parser.add_argument("--nvenc-preset", type=str, help="nvenc preset", default="p1")
    parser.add_argument("--cli-mode", type=int, help="cli mode", default=False)
    parser.add_argument("--int-mult", type=int, help="RIFE interpolation multiplier, 2 means twice the framerate", default=1)
    parser.add_argument("--output-mode", type=str,
                        help="local_mpv plays the output with mpv in the pc's screen, this is for use with Virtual Desktop, hls_ffmpeg streams the output in hls format with ffmpeg",
                        default="local_mpv", choices=['local_mpv', 'hls_ffmpeg'])
    parser.add_argument("--output-pix-fmt", type=str, help="output pixel format", default="yuv420p")
    # parser.add_argument('--input_res_scale', 
    #                type=float,
    #                choices=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #                help='Choose from 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0')
    ######## to avoid mpv dropping the gpu frequency set mpv and python binaries to prefer maximum performance in nvidia control panel


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
    # if  "cli-mode" in sys.argv or 1:        
    parser = create_parser()
    args = parser.parse_args()
    set_state_args(args)
    iw3_desktop_main_hls(args)    

if __name__ == "__main__":
    cli_main()
