import numpy as np
import pickle
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from ddpg_actor import DDPGAgent

class Train:
  def __init__(self, opts, env):
    ############## Hyperparameters ##############
    self.env_name = opts.env_name 
    self.eps = opts.eps                       # noise of DDPG policy
    self.train_iter = opts.train              # update networks for given batched after every episode
    self.lr  = opts.lr                        # learning rate of DDPG policy
    self.random_seed = opts.seed
    self.max_episodes = opts.max_episodes     # max training episodes
    self.max_timesteps = 2000                 # max timesteps in one episode
    ##############################################

    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    
    self.ddpg = DDPGAgent(self.observation_space, self.action_space, eps = self.eps, learning_rate_actor = self.lr)
    
    # logging variables
    self.rewards = []
    self.lengths = []
    self.losses = []
    self.wins = []
    self.winrate = []
    self.timestep = 0
    self.log_interval = 20    # print avg reward in the interval

  def train_loop(self):
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
        
        if episode > 10:
          self.losses.extend(self.ddpg.train(self.train_iter))


        

        self.rewards.append(total_reward)
        self.lengths.append(t)

        eps_now = self.ddpg.ac.eps

        # save every 500 episodes
        if episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(self.ddpg.state(), f'./results/DDPG_{self.env_name}_{episode}-eps{eps_now}-t{self.train_iter}-l{self.lr}-s{self.random_seed}.pth')
            self.save_statistics()

        if episode % 100 == 0:
          winrate_eval = self.evaluate(episodes=20)
          self.winrate.append((episode, winrate_eval))
          print(f"Episode {episode} | Winrate (eps=0): {winrate_eval:.3f}")

        # logging
        if episode % self.log_interval == 0: 
            avg_reward = np.mean(self.rewards[-self.log_interval:])
            avg_length = int(np.mean(self.lengths[-self.log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))

        # exploration scheduling
        self.ddpg.ac.eps = max(0.05, self.ddpg.ac.eps * 0.999)

    return self.rewards, self.losses


  def save_statistics(self):
    with open(f"./results/DDPG_{self.env_name}-eps{self.eps}-t{self.train_iter}-l{self.lr}-s{self.random_seed}-stat.pkl", 'wb') as f:
      pickle.dump({"rewards" : self.rewards, "lengths": self.lengths, "eps": self.eps, "train": self.train_iter,
                  "lr": self.lr, "losses": self.losses, "wins": self.wins, "winrate": self.winrate}, f)
      

  def evaluate(self, episodes=20):
    eval_env = gym.make(self.env_name)
    wins = []

    for episode in range(episodes):
        obs, _ = eval_env.reset(seed=episode)
        done = trunc = False

        while not (done or trunc):
            action = self.ddpg.act(obs, eps=0.0)  
            obs, _, done, trunc, info = eval_env.step(action)

        winner = info.get("winner", 0)
        wins.append(1 if winner == 1 else 0)

    eval_env.close()
    return np.mean(wins)


  def render_env(self):
    eval_env = gym.make(self.env_name)
    obs, _ = eval_env.reset()
    eval_env.render()          
    self.ddpg.reset()

    for _ in range(10000):
        action = self.ddpg.act(obs, eps=0.0)
        obs, _, done, trunc, _ = eval_env.step(action)
        eval_env.render()      
        if done or trunc:
            break

    eval_env.close()



  def load_checkpoint(self, path):
    state = torch.load(path, map_location="cpu")
    self.ddpg.restore_state(state)


  def plot_rewards(self, window=20):
    rewards = np.array(self.rewards)

    plt.figure()
    plt.plot(rewards, alpha=0.4, label="episode reward")

    if len(rewards) >= window:
        moving_avg = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            range(window - 1, len(rewards)),
            moving_avg,
            label=f"{window}-episode avg"
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"DDPG on {self.env_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


  def plot_winrate(self):
    episodes, winrates = zip(*self.winrate)

    plt.figure()
    plt.plot(episodes, winrates, label="Winrate (eval, eps=0)")
    plt.xlabel("Episode")
    plt.ylabel("Winrate")
    plt.ylim(0, 1)
    plt.title(f"Winrate on {self.env_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

