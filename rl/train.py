import numpy as np
import pickle
import gymnasium as gym
import torch
from ddpg_actor import DDPGAgent

class Train:
  def __init__(self, opts, env):
    ############## Hyperparameters ##############
    self.env_name = opts.env_name 
    self.eps = opts.eps                       # noise of DDPG policy
    self.train_iter = opts.train              # update networks for given batched after every episode
    self.lr  = opts.lr                        # learning rate of DDPG policy
    self.random_seed = opts.seed
    self.update_every = opts.update_every
    self.max_episodes = opts.max_episodes     # max training episodes
    self.max_timesteps = 2000                 # max timesteps in one episode
    ##############################################

    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    
    self.ddpg = DDPGAgent(self.observation_space, self.action_space, eps = self.eps, learning_rate_actor = self.lr,
                      update_target_every = self.update_every)
    
    # logging variables
    self.rewards = []
    self.lengths = []
    self.losses = []
    self.timestep = 0
    self.log_interval = 20    # print avg reward in the interval

  def train_loop(self):
    

    # training loop
    for episode in range(1, self.max_episodes+1):
        ob, _ = self.env.reset()
        self.ddpg.reset()
        total_reward=0


        for t in range(self.max_timesteps):
            self.timestep += 1
            a = self.ddpg.act(ob)
            (ob_new, reward, done, trunc, _) = self.env.step(a)
            total_reward+= reward
            self.ddpg.store_transition((ob, a, reward, ob_new, done))
            ob=ob_new
            
            if done or trunc: 
              break

        self.losses.extend(self.ddpg.train(self.train_iter))

        self.rewards.append(total_reward)
        self.lengths.append(t)

        # save every 500 episodes
        if episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(self.ddpg.state(), f'./results/DDPG_{self.env_name}_{episode}-eps{self.eps}-t{self.train_iter}-l{self.lr}-s{self.random_seed}.pth')
            self.save_statistics()

        # logging
        if episode % self.log_interval == 0: 
            avg_reward = np.mean(self.rewards[-self.log_interval:])
            avg_length = int(np.mean(self.lengths[-self.log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))

    return self.rewards, self.losses


  def save_statistics(self):
    with open(f"./results/DDPG_{self.env_name}-eps{self.eps}-t{self.train_iter}-l{self.lr}-s{self.random_seed}-stat.pkl", 'wb') as f:
      pickle.dump({"rewards" : self.rewards, "lengths": self.lengths, "eps": self.eps, "train": self.train_iter,
                  "lr": self.lr, "update_every": self.update_every, "losses": self.losses}, f)
      

  def render_env(self):
        eval_env = gym.make(self.env_name, render_mode="human")
        obs, _ = eval_env.reset()
        self.ddpg.reset()

        for _ in range(200):
            action = self.ddpg.act(obs, eps=0.0)  
            obs, reward, done, trunc, _ = eval_env.step(action)
            if done or trunc:
                break

        eval_env.close()
