import gym
from learner import PixelDataDoubleDeepQLearner

env = gym.make('Atlantis-v0')
max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

learner = None

def main():
    global learner
    TIMESTEP_MAX = max_steps
    K_SKIP = 4
    identity = lambda r, t: r

    learner = PixelDataDoubleDeepQLearner(env.observation_space.shape, (32, 64, env.action_space.n), 128, 4096, identity, rgb_array = True)
    total = 0
    for i_episode in range(10000):
        s1 = env.reset()
        ep_reward = 0
        action_reward = 0
        for t in range(max_steps):
            if t % 4 == 0:
                action = learner.act(s1)
                action_reward = 0
                s0 = s1

            s1, reward, done, info = env.step(action)
            action_reward += reward
            ep_reward += reward

            if t % 4 == 0:
                learner.learn(s0, s1, action_reward, action, done, t+1)
            
            # if i_episode % 500 == 0: env.render()
            if done: break
        total += ep_reward
        print("Episode {0:8d}: {1:4d} timesteps, {2:4f} average".format(i_episode, t+1, total/(i_episode+1)))
    env.close()
main()
