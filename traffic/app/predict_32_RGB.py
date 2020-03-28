# coding:utf-8

import keras
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import pickle
import matplotlib.pyplot as plt
import cv2

folder = [
    "animal",
    "caution",
    "cross_sec",
     "n_cross",
    "n_ent",
    "n_over",
     "oneway",
    "rail",
    "rottary",
    "safe",
    "slow",
    "stop",
    "uneven"
    ]


image_size = 32



def unpickle(file):
    # 保存されたpickleファイルを読み込み
    # 'rb'は｢読み込み専用(r)｣かつ｢バイト列(b)｣を意味する
    with open(file, "rb") as f:
        return pickle.load(f, encoding = "bytes")

# データの読み込みを実行
X_train_f = unpickle("pic_32_color_train_all_gaucian.pickle")
Y_train_f = unpickle("label_32_color_train_all_gaucian.pickle")
X_test_f = unpickle("pic_32_color_test_all_gaucian.pickle")
Y_test_f = unpickle("label_32_color_test_all_gaucian.pickle")

X_train_f = np.array(X_train_f)
X_test_f = np.array(X_test_f)

#one-hotエンコーディング
Y_train = np.identity(len(folder))[Y_train_f].astype('i')
Y_test = np.identity(len(folder))[Y_test_f].astype('i')

def process_image(image):
    
    # サイズをVGG16指定のものに変換する
    #image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # RGBからそれぞれvgg指定の値を引く(mean-subtractionに相当)
    image[:, :, 0] -= 100
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 123.68
    
    # 0-1正規化
    image /= image.max()
    
    
    return image

X_train = X_train_f.astype('float32')
X_test = X_test_f.astype('float32')


# X_trainにaugmetation処理
X_train_list = []
for img in X_train:
    X_train_list.append(process_image(img))
X_train_aug = np.array(X_train_list) # 扱いやすいようlistをndarrayに変換

# X_testにaugmetation処理
X_test_list = []
for img in X_test:
    X_test_list.append(process_image(img))
X_test_aug = np.array(X_test_list) # 扱いやすいようlistをndarrayに変換

from tensorflow.keras.layers import Input, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16

# 入力画像のサイズを指定
input_tensor = Input(shape=(image_size, image_size, 3))

# 学習済みモデルの読み込み
# ダウンロードに数十分かかります
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# 必要なパラメータの追加
input_height = image_size
input_width = image_size
n_class = len(folder)

# 学習済みモデルに加える全結合層部分を定義
# 最終層はノード数がクラスラベルの数に一致していないのでスライシングで取り除く
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256))
top_model.add(Activation('relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(n_class))
top_model.add(Activation('softmax'))

# base_modelとtop_modelを接続
from tensorflow.keras.models import Model
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))


# 畳み込み層の重みを固定（学習させない）
for layer in model.layers[:15]:
        layer.trainable = False

# モデルのコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.0001),
              metrics=['accuracy'])

batch_size = 100
n_epoch = 20 # 簡単に動作確認をするため､epochを1に設定


# 同じように学習完了後、histから結果を確認できます。
hist = model.fit(X_train_aug,
                 Y_train,
                 epochs=n_epoch,
                 validation_data=(X_test_aug, Y_test),
                 verbose=1,
                 batch_size=batch_size)

#評価 & 評価結果出力
print(model.evaluate(X_test, Y_test))


# モデルの保存
open('AI_standard_32_all_RGB.json',"w").write(model.to_json())

# 学習済みの重みを保存
model.save_weights('AI_standard_32_all_RGB_weight.hdf5')


import pickle
with open("AI_standard_32_all_RGB_history.pickle", mode='wb') as f:
    pickle.dump(hist.history, f)