import gym

env = gym.make('LunarLander-v2')
for i_episode in range(200):
    state = env.reset()
    print(state)



    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()