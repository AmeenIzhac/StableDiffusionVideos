import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def index():
    return jsonify({"This is": "a test"})

@app.route('/api')
def api():
    args = request.args
    prompt = args.get('prompt')
    video_name = str(prompt).replace(" ", "_")
    # os.system("conda activate ldm")
    os.system(f'python ../stable-diffusion-2/optimizedSD/optimized_txt2video.py --prompt "{prompt}" --video_name "{video_name}" --H 512 --W 512 --scale 10 --ddim_steps 40 --sampler ddim --strength 0.38 --frames 30 --fps 15 --zoom 1.02 --turbo --ckpt \'../stable-diffusion-2/models/ldm/stable-diffusion-v1/model.ckpt\' --config \'../stable-diffusion-2/optimizedSD/v1-inference.yaml\' ')
    # os.system("conda deactivate")
    return send_file(
         f'./outputs/txt2vid-samples/videos/{video_name}.mp4',
        #'./test.mp4',
        mimetype='video/mp4')



if __name__ == '__main__':
    # ssl_context = ('./certicates/server.crt', './certicates/server.key')
    app.run(debug=True, host='192.168.1.126', port=8080)