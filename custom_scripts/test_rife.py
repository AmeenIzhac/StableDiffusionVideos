import sys
sys.path.append('ECCV2022-RIFE')

from inference_rife import motion_interpolation

motion_interpolation('outputs/images', 'outputs/videos/child_dream_short.mp4', 30, 2, exp=2, scale=1.0, codec='vp09')

