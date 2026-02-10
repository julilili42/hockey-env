import numpy as np
import matplotlib.pyplot as plt

def moving_avg(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode="valid")

data = np.load("logs/td3_weak.npz", allow_pickle=True)

rewards = data["rewards"]
winrate = data["winrate"]

# unpack winrate
eps, wrs = zip(*winrate)

plt.figure(figsize=(6,4))
plt.plot(eps, wrs, label="Winrate vs weak")
plt.axhline(0.55, color="red", linestyle="--", label="Pass threshold")
plt.xlabel("Episode")
plt.ylabel("Winrate")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(moving_avg(rewards, 100))
plt.xlabel("Episode")
plt.ylabel("Avg reward (100 MA)")
plt.tight_layout()
plt.show()
