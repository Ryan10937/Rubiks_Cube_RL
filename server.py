from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'history/general_history'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/test', methods=['POST'])
def test():
  print(request.form)
@app.route('/upload', methods=['POST'])
def upload_files():
  if 'files' not in request.files:
    return jsonify({'error': 'No files part in the request'}), 400
  files = request.files.getlist('files')
  saved_files = []
  for file in files:
    if file.filename == '':
      continue
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    saved_files.append(filename)
  return jsonify({'saved_files': saved_files}), 200

@app.route('/files', methods=['GET'])
def get_files():
  try:
    n = int(request.args.get('n', 1))
  except ValueError:
    return jsonify({'error': 'Invalid value for n'}), 400
  files = os.listdir(app.config['UPLOAD_FOLDER'])
  files = sorted(files)[:n]
  return jsonify({'files': files})

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
  app.run(debug=True,port=6740)


  # Example Python request for uploading files (using requests library):
  #
  # import requests
  #
  # url = 'http://localhost:5000/upload'
  # files = [
  #     ('files', open('file1.txt', 'rb')),
  #     ('files', open('file2.txt', 'rb'))
  # ]
  # response = requests.post(url, files=files)
  # print(response.json())
  #
  # Example response:
  # {
  #   "saved_files": ["file1.txt", "file2.txt"]
  # }
  #
  # Example Python request for downloading a file:
  #
  # import requests
  #
  # url = 'http://localhost:5000/download/file1.txt'
  # response = requests.get(url)
  # with open('file1.txt', 'wb') as f:
  #     f.write(response.content)