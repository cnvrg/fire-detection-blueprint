# Copyright (c) 2022 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as np
import base64
import pathlib
import os
import sys
scripts_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scripts_dir))
from prerun import download_model_files
download_model_files()

if os.path.exists('/input/train'):
    model_path = '/input/train/model.h5'
else:
    scripts_dir = pathlib.Path(__file__).parent.resolve()
    model_path = os.path.join(scripts_dir, 'model.h5')
model = tf.keras.models.load_model(model_path)

def predict(data):
    predictions = []
    for i in data['img']:
        decoded = base64.b64decode(i)
        nparr = np.fromstring(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # img = cv2.imread(data)
        img = cv2.resize(img, (224,224))
        img = image.img_to_array(img)
        img_array = np.expand_dims(img, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        # prediction = [{'label': 'fire', 'score': float(probabilities[0])},
        #               {'label': 'natural', 'score': float(probabilities[1])}]
        pred = np.argmax(probabilities)
        prediction = {'label': str(int(pred)),
                      'score': float(probabilities[pred])}
        predictions.append(prediction)
    return prediction
    