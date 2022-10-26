from os import sendfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def index():
    return jsonify({"This is": "a test"})

@app.route('/api')
def api():
    # get video from filesystem
    video = open('test.mp4', 'rb').read()
    return send_file('./test.mp4', mimetype='video/mp4')
    # data = {'prompt': 'my prompt', 'video': video}
    # return jsonify(data, 200)


if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.125', port=8080)