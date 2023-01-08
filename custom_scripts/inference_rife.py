import sys
sys.path.append('ECCV2022-RIFE')

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

#assumes videogen is sorted, returns a subarray with all the frames whose index is >= starting_frame
def find_frames(videogen, starting_frame, frames_count):
    start = 0
    while int(videogen[start][:-4]) < starting_frame:
        start+=1
    return videogen[start : start + frames_count]
    

def motion_interpolation(frames_dir, output_dir, fps, frames_count, exp=1, scale=1.0, starting_frame=0,model_dir='ECCV2022-RIFE/train_log', fp16=False, ext='mp4', codec='vp09'):

    assert (not frames_dir is None)
    assert scale in [0.25, 0.5, 1.0, 2.0, 4.0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if(fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(model_dir, -1)
        print("Loaded ArXiv-RIFE model")
    model.eval()
    model.device()


    videogen = []
    for f in os.listdir(frames_dir):
        if 'png' in f:
            videogen.append(f)
    tot_images = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    videogen = find_frames(videogen, starting_frame, frames_count)
    tot_frame = frames_count
    lastframe = cv2.imread(os.path.join(frames_dir, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]
    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None

    if output_dir is not None:
        vid_out_name = output_dir
    else:
        #vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** exp), int(np.round(fps)), ext)
        vid_out_name = '{}_{}X_{}fps.{}'.format('hein', (2 ** exp), int(np.round(fps)), ext)
    fourcc = cv2.VideoWriter_fourcc(codec[0], codec[1], codec[2], codec[3])
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, fps, (w, h))

    def clear_write_buffer(write_buffer):
        cnt = 0
        while True:
            item = write_buffer.get()
            if item is None:
                break
            vid_out.write(item[:, :, ::-1])

    def build_read_buffer(read_buffer, videogen):
        try:
            for frame in videogen:
                 if not frames_dir is None:
                      frame = cv2.imread(os.path.join(frames_dir, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
                 read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)

    def make_inference(I0, I1, n):
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def pad_image(img):
        if(fp16):
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    tmp = max(32, int(32 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)
    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (write_buffer,))

    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    temp = None # save lastframe when processing static frame

    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:
            frame = read_buffer.get() # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, scale)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

        if ssim < 0.2:
            output = []
            for i in range((2 ** exp) - 1):
                output.append(I0)
            '''
            output = []
            step = 1 / (2 ** args.exp)
            alpha = 0
            for i in range((2 ** args.exp) - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            '''
        else:
            output = make_inference(I0, I1, 2**exp-1) if exp else []

        
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break

    write_buffer.put(lastframe)
    import time
    while(not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()
    if not vid_out is None:
        vid_out.release()