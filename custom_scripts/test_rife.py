import sys
sys.path.append('ECCV2022-RIFE')

from inference_rife import motion_interpolation

motion_interpolation('outputs/images', 'outputs/videos/quoient.mp4', fps=30, frames_count=24, starting_frame=0, exp=2, scale=1.0, codec='avc1')

