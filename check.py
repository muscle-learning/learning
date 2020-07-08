#綾鷹を選ばせるプログラム

from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

#保存したモデルの読み込み
model = model_from_json(open('model/stomach.json').read())
#保存した重みの読み込み
model.load_weights('model/stomach.hdf5')

categories = ["broken","fat"]


#画像を読み込む
img_path = str(input())
img = image.load_img(img_path,target_size=(250, 250, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#予測
features = model.predict(x)

print(features)

#予測結果によって処理を分ける
if features[0,0] == 1:
    print ("broken")

elif features[0,1] == 1:
    print ("fat")

