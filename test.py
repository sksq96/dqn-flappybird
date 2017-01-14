# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2016-05-19 13:03:53
# @Last Modified by:   shubham
# @Last Modified time: 2016-05-25 16:13:09

import cv2
import numpy as np

from game.FlappyBird import FlappyBird
from random import choice

def image_reshape(image_data):
	image_data = cv2.cvtColor(cv2.resize(image_data, (80, 80)), cv2.COLOR_BGR2GRAY)
	_, image_data = cv2.threshold(image_data,1,255,cv2.THRESH_BINARY)
	return np.stack((image_data, image_data, image_data, image_data), axis=2)


bird = FlappyBird()
while True:
	actions = [0, 1] if choice([True]+[False]*10) else [1, 0]
	image_data, reward, terminal, score = bird.flapOnce(actions)
	image_data = image_reshape(image_data)
	# print(image_data.tolist())
	# break


