import sys
sys.path.append('ECCV2022-RIFE')

from inference_rife import motion_interpolation

motion_interpolation('outputs/images', 'outputs/videos/child_dream_short3.mp4', 30, frames_count=12, exp=1, scale=0.5, codec='vp09')

