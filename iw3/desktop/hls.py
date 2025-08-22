
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
    vp.restart()
    
    try:
        if args.compile:
            depth_model.compile()
            if side_model is not None and not isinstance(side_model, DeviceSwitchInference):
                side_model = compile_model(side_model)
        count = 0

        while True:
            with args.state["args_lock"]:
                

                frame =  vp.decode_queue.get()

                if type(frame) != torch.Tensor and not frame:
                    print("Decode terminated")
                    break
                # c.pt()

                sbs = IW3U.process_image(frame, args, depth_model, side_model)
                # c.tick(f" {vp.audio_queue.qsize()}  {vp.decode_queue.qsize()}, {vp.encode_queue.qsize()} | ")

                vp.encode_queue.put(sbs)

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
    parser.add_argument("--input_file", type=str, help="input_file")
    parser.add_argument("--segment_folder", type=str, help="output for the video segment files ", default="hls_out")
    parser.add_argument("--nvenc-preset", type=str, help="nvenc preset", default="p1")
    

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
    init_win32()

    parser = create_parser()
    args = parser.parse_args()
    set_state_args(args)
    iw3_desktop_main_hls(args)    

if __name__ == "__main__":
    cli_main()
