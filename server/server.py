import os
import sys
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess

sys.path.append('custom_scripts/')
from txt2video import load_model, generate_video, generate_walk_video, ImageArgs, VideoArgs, PathArgs, FloatWrapper

# Load model
path_args = PathArgs()
path_args.image_path = './outputs/frames'
path_args.video_path = './outputs/videos'
model = load_model(path_args, optimized=True)

app = Flask(__name__)
cors = CORS(app)

progress = FloatWrapper()

@app.route('/')
def index():
    return jsonify({"The server": "is running!"})

@app.route('/test')
def test():
    return send_file('./outputs/videos/Lego_man_committing_seppuku.mp4', mimetype='video/mp4')

@app.route('/getProgress')
def getProgress():
    return jsonify({"progress": progress.x})

@app.route('/getVideo')
def getVideo():
    args = request.args
    for arg in args:
        print(arg, args[arg])
    file_name = args.get('fileName').replace(' ', '_')
    file_path = os.path.abspath(f'{os.path.join(path_args.video_path, file_name)}.mp4')
    return send_file(file_path, mimetype='video/mp4')


@app.route('/api')
def api():
    # reset progress
    progress.x = 0
    
    # Extract options
    args = request.args
    
    prompts = args.get('prompts').split(';')
    video_name = str(prompts[0]).replace(" ", "_")
    
    # Prepare options
    image_args = ImageArgs()
    image_args.steps = 50
    image_args.W = int(args.get('width')) 
    image_args.H = int(args.get('height')) 

    video_args = VideoArgs()
    video_args.video_name = video_name

    video_args.prompts = prompts

    video_args.x = float(args.get('xShift'))
    video_args.y = float(args.get('yShift'))
    video_args.zoom = float(args.get('zoom'))
    video_args.angle = float(args.get('angle'))

    video_args.frames = int(args.get('frames'))
    video_args.fps = int(args.get('fps'))

    fMult = int(args.get('fMult'))
    print(f'\n\n\n===================\nfMult is {fMult}\n===================\n\n\n')
    video_args.interp_exp = fMult

    video_args.strength = float(args.get('strength'))

    video_args.upscale = bool(args.get('upscale'))

    def empty_dir(dir):
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

    #delete the previous one
    empty_dir(path_args.image_path)
    empty_dir(path_args.video_path)

    # Generate video
    if args.get('isImg2Img') == 'true':
        generate_video(image_args, video_args, path_args, model, progress)
    else:
        generate_walk_video(image_args, video_args, path_args, model, int(args.get('noNoises')), progress)

    # Respond with the video contents
    return send_file(path_args.video_path, mimetype='video/mp4')

if __name__ == '__main__':
    # ssl_context = ('./certicates/server.crt', './certicates/server.key')
    app.run(
        host='192.168.1.108', 
        port=8080)
