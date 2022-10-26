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
    args = request.args
    prompt = args.get('prompt')
    print(prompt)
    return send_file('./test.mp4', mimetype='video/mp4')



if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.125', port=8080)