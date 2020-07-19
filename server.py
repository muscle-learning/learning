#!/usr/bin/python3
# -*- coding: utf-8 -*-
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS

import os
import werkzeug
from datetime import datetime

from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import json
import boto3


# flask
app = Flask(__name__)
CORS(app)

# ★ポイント1
# limit upload file size : 1MB
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# ★ポイント2
# ex) set UPLOAD_DIR_PATH=C:/tmp/flaskUploadDir
UPLOAD_DIR = "upload"


# rest api : request.files with multipart/form-data
# <form action="/data/upload" method="post" enctype="multipart/form-data">
#   <input type="file" name="uploadFile"/>
#   <input type="submit" value="submit"/>
# </form>
@app.route('/api/upload', methods=['POST'])
def upload_multipart():
    # ★ポイント3
    if 'uploadFile' not in request.files:
        make_response(jsonify({'result': 'uploadFile is required.'}))

    file = request.files['uploadFile']
    fileName = file.filename
    if '' == fileName:
        make_response(jsonify({'result': 'filename must not empty.'}))

    # ★ポイント4
    saveFileName = datetime.now().strftime("%Y%m%d_%H%M%S_") \
                   + werkzeug.utils.secure_filename(fileName)
    file.save(os.path.join(UPLOAD_DIR, saveFileName))

    print(saveFileName)

    # 保存したモデルの読み込み
    model = model_from_json(open('model/stomach.json').read())
    # 保存した重みの読み込み
    model.load_weights('model/stomach.hdf5')

    categories = ["broken", "fat"]

    # 画像を読み込む
    img_path = "upload/" + saveFileName
    img = image.load_img(img_path, target_size=(250, 250, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 予測
    features = model.predict(x)

    # 事後処理（ファイル削除とS3へのアップロード）
    bucket_name = "muscle-uploads"
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).upload_file(img_path, saveFileName)
    os.remove(img_path)

    return make_response(jsonify({'result': {"broken": str(features[0, 0]), "notBroken": str(features[0, 1])}}))

@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    print("werkzeug.exceptions.RequestEntityTooLarge")
    return 'result : file size is overed.'


if __name__ == "__main__":
    from waitress import serve

    serve(app, host='0.0.0.0', port=3000)
