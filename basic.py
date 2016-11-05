import gym
from time import sleep
env = gym.make('DoomDeathmatch-v0')
env.reset()
env.mode='normal'
num_episode = 5
num_step = 50
for i in range(num_episode):
	print 'episode num:' + str(i+1)
	env.reset()
	for j in range(num_step):
		env.render()
		ob, reward, done, _ = env.step(env.action_space.sample())
        if done:
            break
        # sleep(1)
	sleep(2.0)

env.render(close=True)