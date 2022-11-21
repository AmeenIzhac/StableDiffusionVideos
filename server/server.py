import os
import sys
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess

sys.path.append('../custom_scripts/')
from txt2video import load_model, generate_video, ImageArgs, VideoArgs, PathArgs

# Load model
cfg_path = '../stable-diffusion-2/configs/stable-diffusion/v1-inference.yaml'
optimized_cfg_path = '../stable-diffusion-2/optimizedSD/v1-inference.yaml'
ckpt_path = '../stable-diffusion-2/models/ldm/stable-diffusion-v1/model.ckpt'
model = load_model(optimized_cfg_path, ckpt_path, optimized=True)


app = Flask(__name__)
cors = CORS(app)


@app.route('/')
def index():
    return jsonify({"The server": "is running!"})

@app.route('/test')
def test():
    return send_file('./outputs/videos/Lego_man_committing_seppuku.mp4', mimetype='video/mp4')


@app.route('/api')
def api():
    # Extract options
    args = request.args
    prompts = args.get('prompts')
    video_name = str(prompt).replace(" ", "_")
    
    # Prepare options
    image_args = ImageArgs()
    image_args.steps = 50
    image_args.W = int(args.get('width')) if ("width" in args) else 512
    image_args.H = int(args.get('height')) if ("height" in args) else 512

    video_args = VideoArgs()
    video_args.video_name = video_name
    video_args.prompts = prompts
    video_args.fps = 20
    video_args.strength = 0.375
    video_args.zoom = 1.005
    video_args.x = -5
    video_args.frames = int(args.get('frames')) if ("frames" in args) else 60
    video_args.upscale = True

    path_args = PathArgs()
    path_args.image_path = os.path.abspath('./outputs/frames/')
    path_args.video_path = os.path.abspath(f'./outputs/videos/{video_name}.mp4')

    # Generate video
    generate_video(image_args, video_args, path_args, model)

    # Respond with the video contents
    return send_file(path_args.video_path, mimetype='video/mp4')


if __name__ == '__main__':
    # ssl_context = ('./certicates/server.crt', './certicates/server.key')
    app.run(
        host='192.168.1.125', 
        port=8080)
