import numpy as np

class TD3Trainer:
    def __init__(self, agent, env,
                 max_episodes=2000,
                 max_steps=2000,
                 train_iters=32,
                 eval_interval=100):
        self.agent = agent
        self.env = env
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.train_iters = train_iters
        self.eval_interval = eval_interval

        self.rewards = []
        self.winrate = []

    def train(self):
        for ep in range(1, self.max_episodes + 1):
            obs, _ = self.env.reset()
            self.agent.reset()
            ep_reward = 0

            for t in range(self.max_steps):
                action = self.agent.get_action(obs, noise=True)
                next_obs, reward, done, trunc, info = self.env.step(action)

                self.agent.replay_buffer.push(
                    obs, action, reward, next_obs, done or trunc
                )

                ep_reward += reward
                obs = next_obs
                if done or trunc:
                    break

            self.rewards.append(ep_reward)

            # === Training ===
            if self.agent.total_steps > self.agent.batch_size:
                for _ in range(self.train_iters):
                    self.agent.update_step()

            # === Evaluation ===
            if ep % self.eval_interval == 0:
                wr = self.evaluate(20)
                self.winrate.append((ep, wr))
                print(f"Episode {ep} | Reward {ep_reward:.1f} | Winrate {wr:.2f}")

    def evaluate(self, episodes=20):
        wins = []
        for i in range(episodes):
            obs, _ = self.env.reset(seed=i)
            done = False

            while not done:
                action = self.agent.get_action(obs, noise=False)
                obs, _, done, trunc, info = self.env.step(action)
                done = done or trunc

            wins.append(1 if info.get("winner", 0) == 1 else 0)

        return np.mean(wins)
