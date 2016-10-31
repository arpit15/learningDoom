import gym
env = gym.make('DoomBasic-v0')
env.reset()
num_epis = 2

episode_count = 100
max_steps = 10
reward = 0
done = False

for i in range(episode_count):
    ob = env.reset()

    for j in range(max_steps):
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        
        env.render()
        print action, reward
        if done:
            break
env.render(close=True)
