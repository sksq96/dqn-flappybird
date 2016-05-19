# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2016-05-19 17:03:33
# @Last Modified by:   shubham
# @Last Modified time: 2016-05-19 19:36:58

from random import randint, choice
from pprint import pprint

gamma = 0.9
num_states, num_actions = 6, 6

Q = [[0 for _ in range(num_actions)] for _ in range(num_states)]

min_reward, max_reward = 0, 100
graph = {
	0: [(4,min_reward)],
	1: [(3,min_reward), (5,max_reward)],
	2: [(3,min_reward)],
	3: [(1,min_reward),(2,min_reward),(4,min_reward)],
	4: [(0,min_reward),(3,min_reward),(5,max_reward)],
	5: [(1,min_reward),(4,min_reward),(5,max_reward)],
}

# Reinforcement training
goal = 5
for episode in range(100):
	state = randint(0, 5)
	while state is not goal:
		actions = graph[state]
		next_state, reward = choice(actions)

		Q[state][next_state] = reward + gamma * max(Q[next_state])
		state = next_state
pprint(Q)

# Reinforcement testing
for state in range(5):
	print(state, end=' ')
	while state is not goal:
		state, reward = max(enumerate(Q[state]), key=lambda x:x[1])
		print(state, end=' ')
	print()

