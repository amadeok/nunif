import torch
from .utils import create_parser, set_state_args, iw3_main
from . import models # noqa
from nunif.logger import logger
from nunif.device import device_is_cuda

pipe_name = r'\\.\pipe\test_video'

import win32pipe, subprocess, threading, time
import win32file

def create_pipe():
    # Named pipe path
    
    # Create the named pipe
    size = 256*256*256*100
    pipe = win32pipe.CreateNamedPipe(
        pipe_name,
        win32pipe.PIPE_ACCESS_DUPLEX,
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
        2, size, size, 0, None
    )
    
    print(f"Named pipe created: {pipe_name}")
    return pipe

def connect_pipe(pipe):
    try:
        win32pipe.ConnectNamedPipe(pipe, None)
        print("Client connected to pipe")
        # while 1:
        #     ret = win32file.ReadFile(pipe, 256*256)
        #     print(ret[0], len(ret[1]))
        return True
    except Exception as e:
        print(f"Error connecting pipe: {e}")
        return False


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # args.pipe = create_pipe()
    
    #         # Spawn MPV with stdin input
    # def mpv():
    #     process = subprocess.Popen(
    #         [r'C:\Users\amade\rifef _\mpv-x86_64\mpv_.com', '-', f"<  {pipe_name}"],  # '-' indicates stdin
    #         # stdin=subprocess.PIPE,
    #         # stdout=subprocess.DEVNULL,
    #         # stderr=subprocess.DEVNULL
    #     )
    # # threading.Timer(1, mpv).start()    
    # win32pipe.ConnectNamedPipe(args.pipe, None)
    
    # time.sleep(1)
    
    # threading.Timer(0, lambda: connect_pipe(args.pipe)).start()

    # threading.Timer(3, lambda: subprocess.Popen([r"C:\Users\amade\rifef _\mpv-x86_64\mpv_.com", pipe_name ])).start()
    
    # win32file.WriteFile(args.pipe, b'23122312dh812331')
    # time.sleep(5)
    
    set_state_args(args)
    iw3_main(args)
    

    
    if device_is_cuda(args.state["device"]):
        max_vram_mb = int(torch.cuda.max_memory_allocated(args.state["device"]) / (1024 * 1024))
        logger.debug(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    main()
