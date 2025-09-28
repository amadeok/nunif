
for x in range(1000):
    print("---------------------> x", x)
    
import sys, os

RIFE_PLAYER_MLRT_SCRIPT_PATH =  os.getenv("RIFE_PLAYER_MLRT_SCRIPT_PATH")
sys.path+=[RIFE_PLAYER_MLRT_SCRIPT_PATH]
# print = lambda *args: 1 if os.getenv("vs_silent") else print
print("os.getenv('vs_silent')", os.getenv("vs_silent"), "--------------------------------------------")

# print("-----> version" ,sys.version, "----bin", sys.executable, "RIFE_PLAYER_MLRT_SCRIPT_PATH", RIFE_PLAYER_MLRT_SCRIPT_PATH)

import time, vapoursynth as vs, fractions, configparser, threading
from vsmlrt import RIFE, RIFEModel, Backend
def mult32(number):  return  ((number + 31) // 32) * 32
tw = 1280; th =720; autoscale = False

def getW(): return  mult32(tw) if autoscale  else mult32(video_in_dw)
def getH(): return mult32(th) if autoscale else mult32(video_in_dh)

def pad_to_multiple_of_32(clip):
    width = clip.width
    height = clip.height
    pad_left = (32 - (width % 32)) % 32
    pad_right = (32 - ((width + pad_left) % 32)) % 32
    pad_top = (32 - (height % 32)) % 32
    pad_bottom = (32 - ((height + pad_top) % 32)) % 32
    print(f"Adding borders: pad_left: {pad_left}, pad_right: {pad_right}, pad_top: {pad_top}, pad_bottom: {pad_bottom})")
    return [clip.std.AddBorders(pad_left, pad_right, pad_top, pad_bottom), [pad_left, pad_right, pad_top, pad_bottom]]




appdata_dir = os.getenv('APPDATA')
class context():
    def __init__(self) -> None:
        self.exit = 0

ctx = context()

file_path = os.path.join(appdata_dir, 'Rife Player//RIFE_PLAYER_SCRIPT_STATUS.txt')

def funvar(ctx_):
    while 1:
        with open(file_path, "w+") as ff:
            ff.write("")
        time.sleep(1)
        if ctx_.exit: 
            print("writer fun ended")
            break
        
threading.Thread(target=funvar, args=(ctx,)).start()
clip = video_in
print("Vapoursynth input format: ", clip.format)
print("Python Path:")
print(sys.executable)
print("Python Version:")
print(sys.version)
try:
    cMatrix = '709'
    cRange = 'limited'
    core = vs.core
    multiplier = 2
    envRPmult = os.getenv("RIFE_PLAYER_MULTIPLIER")
    if envRPmult:
        print("Using env variable RIFE_PLAYER_MULTIPLIER: ", envRPmult ) 
        multiplier = int(envRPmult)
    else:
        print("End variable RIFE_PLAYER_MULTIPLIER not found, using default: ", multiplier ) 
    disableInt = envRPmult == 1 #os.getenv("RIFE_PLAYER_DISABLE_INT")
    print("Disable interpolation: ", disableInt, ",  RIFE_PLAYER_MULTIPLIER: ", envRPmult)
    if disableInt  and disableInt == "yes":
        clip.set_output()
    else:

        try:core.std.LoadPlugin("/content/drive/MyDrive/rifef/libmiscfilters.so") #/content/vs-miscfilters-obsolete/build
        except Exception as e: 
            print(e)

        #print("-----> clip bps1 ", clip[0].format.bits_per_sample )
        # if cFamily == 'ColorFamily.RGB':
        #   clip = core.resize.Bicubic(clip, format=vs.RGBS, range_in_s=cRange, filter_param_a=1, filter_param_b=0, width=mult32(getW()), height=mult32(getH()))
        # else:
        #   clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=cMatrix, range_in_s=cRange, filter_param_a=1, filter_param_b=0, width=mult32(getW()), height=mult32(getH()))
        # new_width = (video_in_dw // 32) * 32
        # new_height = (video_in_dh // 32) * 32
        # left = (video_in_dw - new_width)
        # right = left
        # top = (video_in_dh - new_height)
        # bottom = top
        #print( left, right, top, bottom," before cropRel")
        #clip = vs.core.std.CropRel(clip, left, right, top, bottom) #this fails sometimes
        #clip = vs.core.std.CropAbs(clip, left=0, top=0, width=new_width, height=new_height)
        #clip = clip.misc.SCDetect(threshold=0.1)
        #clip = core.std.Expr(clip, "x 1 -")
        
        oriw = clip.width
        orih = clip.height
        padding = None
        needsPadding = orih % 32 != 0 or oriw % 32 != 0
        print("oriw orih ", oriw, orih)
        if needsPadding:
            clip, padding = pad_to_multiple_of_32(clip)
        else:
            print("No padding required ", oriw, "x", orih)
        #print("test3 before bicubic")
        clip = clip.resize.Bicubic(format=vs.RGBH,matrix_in_s="709") #RGBH for 16 bit per sample output
        #print("-----> clip bps2 ", clip[0].format.bits_per_sample )
        #print("test3 before backend")
        trt_backend = Backend.TRT(fp16=True,device_id=0,num_streams=2,output_format=1,use_cuda_graph=True,workspace=None)
        #trt_backend = Backend.TRT(fp16=True,device_id=0,num_streams=2,output_format=1,use_cuda_graph=True,workspace=None,static_shape=False,min_shapes=[64,64],max_shapes=[2560,1440])
        # trt_backend = Backend.TRT(fp16=True,device_id=0,num_streams=2,output_format=1,use_cuda_graph=True,workspace=None,static_shape=False,min_shapes=[7,7],max_shapes=[2560,1440])
        #print("test3 before rife")

        clip = RIFE(clip,model=RIFEModel.v4_6, backend=trt_backend, multi=multiplier)# v4_6
        if needsPadding:
            clip =  clip.std.Crop(padding[0], padding[1],padding[2],padding[3])

        #print("output")

    # print("test3 after rife")
        fps = 24000/1001
        try:
            fps =  float(container_fps)*multiplier
        except:
            pass
        #YUV420P8
        pixel_format = vs.YUV444P8 if os.getenv("vs_output_pixel_format") == "yuv444p" else vs.YUV420P8
        clip = core.resize.Bicubic(clip,  format=pixel_format, matrix_s=cMatrix, range_s=cRange, filter_param_a=1, filter_param_b=0) #width=getW(), height=getH(),
        # clip = core.resize.Bicubic(clip,  matrix_s=cMatrix, range_s=cRange, filter_param_a=1, filter_param_b=0) #width=getW(), height=getH(),
        
        #clip = vs.core.std.AddBorders(clip,  right=left, bottom=top )  
        fps_fraction = fractions.Fraction(container_fps * multiplier).limit_denominator()
        output_num, output_den = fps_fraction.numerator, fps_fraction.denominator

        print("NEW FRAMERATE", container_fps, " -> ",  fps, " : ", output_num,  "/", output_den, " / ", clip.format)
        
        # target_fps_num = 60
        # target_fps_den = 1
        # import math, functools
        # clip = core.std.AssumeFPS(clip, fpsnum=23999, fpsden=1000)

        # def frame_adjuster(n, clip, target_fps_num, target_fps_den):
        #     real_n = math.floor(n / (target_fps_num / target_fps_den * clip.fps_den / clip.fps_num))
        #     one_frame_clip = clip[real_n] * (len(clip) + 100)
        #     return one_frame_clip

        # attribute_clip = core.std.BlankClip(clip, length=math.floor(len(clip) * target_fps_num / target_fps_den * clip.fps_den / clip.fps_num), fpsnum=target_fps_num, fpsden=target_fps_den)
        # adjusted_clip = core.std.FrameEval(attribute_clip, functools.partial(frame_adjuster, clip=clip, target_fps_num=target_fps_num, target_fps_den=target_fps_den))
        # adjusted_clip.set_output()

        #clip = core.std.AssumeFPS(clip, fpsnum=output_num, fpsden=output_den)
        #clip = core.std.AssumeFPS(clip, fpsnum=output_num, fpsden=output_den) GIVES WRONG FPS ON COLAB

        # output to mpv
        print("set_output")

        clip.set_output()
except Exception as e:
    import traceback

    def draw_text(clip, text, align=4, scale=2):
        blank = vs.core.std.BlankClip(clip, width=2, height=clip.height, format=clip.format)#clip.width*0
        merged = vs.core.std.StackHorizontal([clip, blank])
        text_clip = vs.core.text.Text(clip=merged, text=text,alignment=align, scale=scale )#, x=x, y=y, size=font_size, color=color)
        return text_clip

    
    exception_str = traceback.format_exc()
    errormsg = f"Interpolation script failed, error: \n----------------\n{e}\n{exception_str}----------------\n"
    try:
        clip = draw_text(video_in, errormsg )
    except:
        clip = clip.resize.Bicubic(format=vs.RGB24,matrix_in_s=cMatrix) #RGBH for 16 bit per sample output
        clip = draw_text(clip, errormsg )
        clip = core.resize.Bicubic(clip,  format=vs.YUV420P8, matrix_s=cMatrix, range_s=cRange, filter_param_a=1, filter_param_b=0) #width=getW(), height=getH(),
    # pyautogui
    # try:
    #     pyautogui.alert(text=errormsg, title='', button='OK')
    # except Exception as e:
    #     print(e)
    print( errormsg)
    clip.set_output()



appdata_dir = os.getenv('APPDATA')

file_path = os.path.join(appdata_dir, 'Rife Player//RIFE_PLAYER_SCRIPT_STATUS.txt')
if os.path.exists(file_path):
    # Delete the file
    os.remove(file_path)
    print(f"{file_path} has been deleted.")
else:
    print(f"{file_path} does not exist.")

print( "RIFE_PLAYER_SCRIPT_STATUS deleted")
ctx.exit = 1




