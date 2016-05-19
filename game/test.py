# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2016-05-19 13:03:53
# @Last Modified by:   shubham
# @Last Modified time: 2016-05-19 16:02:31

from FlappyBird import FlappyBird
from random import choice

def main():

	while True:
		bird = FlappyBird()
		while True:
			crashInfo = bird.flapOnce(choice([True]+[False]*10))
			if crashInfo: break

if __name__ == '__main__':
	main()

