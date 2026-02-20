# Presentation Script — TD3 for Competitive Air Hockey
# Target: 3 minutes (~430 words)
# Tip: Practice with a timer. Speak slowly and clearly.

---

## Slide 1: Title (5 sec)

Hi, I'm Julian Jurcevic from team alphabet-td3. I'll present my work on training a TD3 agent for competitive air hockey.

---

## Slide 2: Air Hockey Environment (25 sec)

The environment is a two-player air hockey game built on Gymnasium and Box2D. Each player controls a paddle with four continuous actions: translation in x and y, rotation, and shooting. The state is 18-dimensional and encodes positions, velocities, and angles of both players and the puck. The main reward is plus or minus ten for scoring or conceding, with additional shaped rewards.

The key challenges are non-stationarity — since the opponent changes over time — fast dynamics with sparse rewards, and the risk of overfitting to a single opponent.

---

## Slide 3: TD3 — Three Key Ideas (25 sec)

I use TD3, which addresses overestimation bias through three mechanisms. First, clipped double Q-learning: two critics are maintained and the Bellman target uses the minimum of both, reducing overestimation. Second, target policy smoothing adds clipped noise to the target action, which regularizes the critic. Third, the actor is updated less frequently than the critics — every second step — and target networks are updated via Polyak averaging. The architecture uses two hidden layers of 256 units with tanh activations.

---

## Slide 4: Exploration & Replay (20 sec)

Exploration noise is linearly annealed from an initial scale down to a minimum, allowing broad exploration early and stable exploitation later. I compared four noise types and found Ornstein-Uhlenbeck noise to work best, likely due to its temporal correlation producing smoother action sequences. I use a uniform replay buffer with 300k transitions. Prioritized replay was tested but hurt performance, as I'll show later.

---

## Slide 5: Three-Stage Curriculum (25 sec)

Training follows a three-stage curriculum. Stage one trains purely against the weak opponent to learn basic puck control. Stage two introduces a mix of weak and strong opponents with some self-play. Stage three increases self-play up to 30% for generalization. Transitions between stages are triggered when the weak win rate plateaus above 85%. The self-play pool stores snapshots every 150 episodes and samples opponents proportional to a difficulty score that favors opponents the agent struggles against.

---

## Slide 6: Noise Ablation (20 sec)

In the noise ablation, OU noise clearly outperformed all alternatives — 89% win rate against strong versus 81% for Gaussian, a gain of 8 percentage points. Pink noise came second, which is consistent with the temporal correlation hypothesis. Uniform noise showed the highest variance across seeds.

---

## Slide 7: Self-Play & PER Ablation (20 sec)

Prioritized replay consistently reduced performance by 15 to 20 percent, likely because non-stationary opponents amplify the variance of priority-based sampling. Self-play slightly reduced benchmark scores since the agent adapts to past versions of itself. However, it is retained in the final agent to improve robustness against unseen tournament opponents.

---

## Slide 8: Curriculum Progression (15 sec)

Looking at the training progression: Stage one reaches 95% against weak but only 30% against strong — clear overfitting. The curriculum progressively improves strong performance to around 85% while maintaining weak win rates. Model selection uses the minimum of both win rates to enforce robustness.

---

## Slide 9: Conclusion (20 sec)

To summarize: curriculum learning was the most impactful design choice. OU noise gave a clear gain of 8% against the strong opponent. Self-play is retained for tournament generalization despite lower benchmark scores. PER did not work in this non-stationary setting. The final agent achieves approximately 95% against weak and 85% against strong. Limitations include single-seed training curves and manual curriculum tuning.

---

## Slide 10: Thank You (5 sec)

Thank you for your attention. I'm happy to take questions.

---

# Total: ~435 words ≈ 2:55 at normal pace
# Leave ~5 sec buffer for slide transitions
