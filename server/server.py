from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def index():
    return jsonify({"This is": "a test"})

@app.route('/api')
def api():
    # get video from filesystem
    # video = open('test.mp4', 'rb').read()
    data = {'prompt': 'my prompt'}
    return jsonify(data, 200)


if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.125', port=8080)