import gym
import time

env = gym.make('DoomBasic-v0')
# env.reset()
num_epis = 2

episode_count = 100
max_steps = 10
reward = 0
done = False

actions_num = env.action_space.shape

for i in range(episode_count):
    print 'new episode num:' + str(i+1)
    ob = env.reset()

    for j in range(max_steps):
        print 'step num:' + str(j+1)
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        
        env.render()
        print 'action taken'
        time.sleep(0.5)
        # print action, reward

        action = [0]*actions_num
        curr_state, _, _, _ = env.step(action)
        env.render()
        if done:
            break
env.render(close=True)
