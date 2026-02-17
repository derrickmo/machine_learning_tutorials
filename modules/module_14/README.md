# Module 14 — Reinforcement Learning

## Introduction

Module 14 provides a comprehensive treatment of reinforcement learning, progressing from foundational MDP theory and Bellman equations through tabular methods (Monte Carlo, TD learning, Q-learning) to deep RL algorithms (DQN, REINFORCE, PPO) and advanced topics including model-based RL, offline RL, and imitation learning. This module is essential in the curriculum because RL represents a fundamentally different learning paradigm from supervised and unsupervised learning -- agents must learn from sparse, delayed rewards through trial-and-error interaction with an environment, making exploration, credit assignment, and stability the central challenges. After completing this module, students will be able to implement every major RL algorithm from scratch, from value iteration on a custom GridWorld through PPO on gymnasium environments, and understand when to apply model-free vs model-based, on-policy vs off-policy, and value-based vs policy-based approaches. Module 14 builds on the neural network foundations from Module 5 and the RLHF-specific PPO introduced in Module 13, while providing the RL theory that deepens understanding of alignment methods and connects to planning and decision-making problems across the curriculum.

**Folder:** `module_14_reinforcement_learning/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 14-01 | RL Fundamentals — MDPs & Bellman Equations | States, actions, rewards, transitions; Markov Decision Processes; Bellman expectation and optimality equations; value iteration and policy iteration from scratch; discount factor intuition | GridWorld (custom) | ~3 min |
| 14-02 | Policy Evaluation — Monte Carlo & TD Learning | First-visit and every-visit MC evaluation from scratch; TD(0) update rule; TD(λ) and eligibility traces; bias-variance tradeoff between MC and TD; n-step returns; this is the foundational bridge that makes Q-learning (14-03) derivation make sense | GridWorld (custom) | ~3 min |
| 14-03 | Q-Learning & SARSA | Temporal difference control; Q-learning update rule from scratch (off-policy); SARSA (on-policy); epsilon-greedy exploration; Q-table on GridWorld; convergence properties; now builds directly on TD foundations from 14-02 | GridWorld (custom) | ~5 min |
| 14-04 | Exploration vs. Exploitation — Bandits, UCB & Thompson Sampling | Multi-armed bandit formulation; epsilon-greedy; Upper Confidence Bound (UCB1) from scratch; Thompson Sampling (posterior sampling); optimism in the face of uncertainty; PAC-MDP bounds concepts; the defining challenge that separates RL from supervised learning | Synthetic (bandit rewards) | ~3 min |
| 14-05 | Deep Q-Networks (DQN) | Neural network as Q-function approximator; experience replay buffer; target network and soft updates; Double DQN — overestimation bias correction; training on CartPole/LunarLander | CartPole (gymnasium) | ~8 min |
| 14-06 | Policy Gradient — REINFORCE | Policy gradient theorem derivation; REINFORCE algorithm from scratch; variance reduction with baselines; advantage estimation (A = Q - V); comparison with value-based methods | CartPole (gymnasium) | ~8 min |
| 14-07 | Actor-Critic — A2C & PPO | Actor-critic architecture; A2C — advantage actor-critic with bootstrap; PPO — clipped surrogate objective (full general RL formulation); GAE (generalized advantage estimation); direct connection to Module 13 RLHF | LunarLander (gymnasium) | ~10 min |
| 14-08 | Model-Based RL, Planning & MCTS | Learned dynamics models; Dyna architecture (model-free + model-based hybrid); MBPO concepts; Monte Carlo Tree Search — UCT algorithm from scratch, AlphaGo/AlphaZero connection; when model-based beats model-free; meta-RL concepts (RL², learning to learn) | GridWorld, CartPole | ~8 min |
| 14-09 | Offline RL — CQL, IQL & Conservative Methods | Learning from fixed datasets without environment interaction; distribution shift in offline RL; Conservative Q-Learning (CQL) — pessimistic value estimation; Implicit Q-Learning (IQL); practical relevance: healthcare, robotics, recommendations where exploration is dangerous or impossible | Synthetic offline (CartPole trajectories) | ~8 min |
| 14-10 | Imitation Learning & Inverse RL | Behavioral cloning from scratch; distribution shift (compounding error); DAgger (Dataset Aggregation) — interactive imitation learning; inverse RL (reward recovery from demonstrations); connection to SFT in Module 13 (SFT is behavioral cloning for LLMs) | CartPole demonstrations | ~8 min |

---

## Topic Details

### 14-01: RL Fundamentals -- MDPs & Bellman Equations
Students will formalize the reinforcement learning problem through the Markov Decision Process framework, defining states, actions, rewards, transition dynamics, and the discount factor that controls the tradeoff between immediate and future rewards. The Bellman expectation equations for state-value V(s) and action-value Q(s,a) functions are derived, followed by the Bellman optimality equations that characterize the optimal policy. Students implement both value iteration and policy iteration from scratch as dynamic programming algorithms on a custom GridWorld environment, observing convergence and understanding the relationship between iterative policy evaluation and policy improvement. This topic establishes the mathematical language and core concepts -- value functions, policies, optimality -- that every subsequent RL topic in the module builds upon.

### 14-02: Policy Evaluation -- Monte Carlo & TD Learning
This topic introduces the two foundational approaches for estimating value functions when the environment dynamics are unknown: Monte Carlo methods that average complete episode returns, and Temporal Difference learning that bootstraps from partial estimates. Students implement first-visit and every-visit MC evaluation from scratch, then derive and implement the TD(0) one-step update rule, followed by TD(lambda) with eligibility traces that interpolates between MC and TD. The bias-variance tradeoff is analyzed rigorously -- MC has zero bias but high variance from full-episode returns, while TD has lower variance but introduces bias through bootstrapping -- with n-step returns providing the configurable middle ground. This topic is intentionally placed before Q-learning (14-03) because understanding policy evaluation makes the transition to control algorithms a natural extension rather than a conceptual leap.

### 14-03: Q-Learning & SARSA
Building directly on the TD foundations from 14-02, students implement the two fundamental temporal difference control algorithms: Q-learning (off-policy) and SARSA (on-policy). Q-learning is derived from the Bellman optimality equation with the key insight that the max operator over next-state actions enables learning the optimal policy while following an exploratory behavior policy. SARSA is contrasted as the on-policy alternative that updates toward the value of the action actually taken, making it more conservative in stochastic environments. Both algorithms are implemented with epsilon-greedy exploration on the custom GridWorld, with Q-table visualization showing how the value landscape evolves during training. Convergence properties and the differences between on-policy and off-policy learning are made concrete through experiments, preparing students for the function approximation extensions in DQN (14-05).

### 14-04: Exploration vs. Exploitation -- Bandits, UCB & Thompson Sampling
Students tackle the exploration-exploitation dilemma -- the defining challenge that separates RL from supervised learning -- through the multi-armed bandit formulation where no sequential state transitions exist and the problem reduces to pure action selection under uncertainty. Epsilon-greedy exploration is implemented as the simplest baseline, followed by Upper Confidence Bound (UCB1) which selects actions based on both estimated reward and uncertainty, formalizing the "optimism in the face of uncertainty" principle. Thompson Sampling is implemented as a Bayesian approach that samples from posterior distributions over arm rewards and acts according to the probability that each arm is optimal. PAC-MDP bounds concepts connect the theoretical foundations to sample complexity guarantees. Though placed after Q-learning in the sequence, this topic addresses the fundamental exploration problem that pervades all RL algorithms.

### 14-05: Deep Q-Networks (DQN)
This topic bridges tabular RL and deep RL by replacing the Q-table with a neural network function approximator, enabling RL in high-dimensional state spaces where tabular methods are intractable. Students implement the full DQN architecture including the experience replay buffer that breaks temporal correlations in training data and the target network with soft updates that stabilizes learning by providing a slowly-moving regression target. Double DQN is implemented to address the overestimation bias inherent in standard DQN's max operator, using the online network for action selection and the target network for evaluation. Training on CartPole through the gymnasium interface demonstrates the complete deep RL training loop, establishing the pattern that scales to more complex environments and connects forward to policy gradient methods (14-06) and actor-critic architectures (14-07).

### 14-06: Policy Gradient -- REINFORCE
Students derive the policy gradient theorem from first principles, showing that the gradient of the expected return with respect to policy parameters can be estimated from sampled trajectories without differentiating through the environment dynamics. The REINFORCE algorithm is implemented from scratch as the simplest instantiation of this theorem, using episode returns as the gradient weight. Variance reduction with baselines is added -- subtracting a learned state-dependent baseline from returns reduces gradient variance without introducing bias -- leading naturally to the advantage function A(s,a) = Q(s,a) - V(s). A comparison with value-based methods (DQN from 14-05) highlights the fundamental tradeoff: policy gradient methods can handle continuous action spaces and stochastic policies but suffer from high variance, while value-based methods are more sample-efficient but limited to discrete actions.

### 14-07: Actor-Critic -- A2C & PPO
This topic combines value-based and policy-based approaches in the actor-critic framework, where an actor network selects actions and a critic network estimates values to reduce gradient variance. Students implement A2C (Advantage Actor-Critic) with bootstrapped advantage estimation, then build PPO (Proximal Policy Optimization) with the clipped surrogate objective that constrains policy updates to a trust region, preventing the catastrophically large updates that plague vanilla policy gradients. Generalized Advantage Estimation (GAE) is implemented to provide the tuneable bias-variance tradeoff in advantage computation via the lambda parameter. The direct connection to Module 13's RLHF is made explicit: 13-08 introduced PPO in the language model alignment context, while this notebook derives PPO in its full generality with environment rollouts, episodic returns, and value network training on LunarLander.

### 14-08: Model-Based RL, Planning & MCTS
Students explore model-based reinforcement learning, where a learned dynamics model enables planning ahead rather than relying solely on trial-and-error interaction. The Dyna architecture is implemented from scratch, combining real environment interaction with simulated experience from the learned model to dramatically improve sample efficiency. MBPO (Model-Based Policy Optimization) concepts extend this to the deep RL setting with uncertainty-aware model usage. Monte Carlo Tree Search (MCTS) with the UCT algorithm is implemented from scratch, demonstrating how search-based planning with a simulator achieves superhuman performance in the AlphaGo/AlphaZero paradigm. The notebook analyzes when model-based methods outperform model-free approaches -- environments with expensive interaction, clear structure, and sufficient model accuracy -- and introduces meta-RL concepts (RL2) where the agent learns the learning algorithm itself.

### 14-09: Offline RL -- CQL, IQL & Conservative Methods
This topic addresses learning optimal policies from fixed datasets without any online environment interaction, a critically important setting for domains where exploration is dangerous, expensive, or ethically problematic -- healthcare treatment decisions, autonomous driving, and recommendation systems. Students implement Conservative Q-Learning (CQL), which adds a regularization term that penalizes Q-values for out-of-distribution actions to combat the distribution shift that causes standard off-policy methods to fail catastrophically on static datasets. Implicit Q-Learning (IQL) is implemented as an alternative that avoids querying out-of-distribution actions entirely by using expectile regression. The fundamental challenge of offline RL -- extrapolation error when the policy visits states or actions not covered by the dataset -- is demonstrated empirically on synthetic CartPole trajectories, connecting the theoretical concerns to measurable failure modes.

### 14-10: Imitation Learning & Inverse RL
Students implement imitation learning algorithms that learn policies from expert demonstrations rather than reward signals, starting with behavioral cloning -- supervised learning on state-action pairs from an expert policy -- and analyzing the compounding error problem caused by distribution shift when the learned policy encounters states the expert never visited. DAgger (Dataset Aggregation) is implemented as the solution: iteratively collecting labels from the expert on states visited by the current policy to close the distribution gap. Inverse RL -- recovering the latent reward function that the expert is implicitly optimizing -- is introduced as a more principled but computationally expensive alternative. The explicit connection to SFT in Module 13 is drawn: supervised fine-tuning of language models is behavioral cloning applied to LLMs, with the same distribution shift problems and the same motivation for online feedback (RLHF) over pure imitation.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 14-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 14-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 14-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 14-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 14-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 14-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 14-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 14-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 14-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 14-10 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |

---

## Module-Specific Packages

- `gymnasium` — RL environments (14-03 through 14-07)

---

## Datasets

- GridWorld custom (14-01, 14-02, 14-03, 14-04, 14-08)
- Synthetic bandit rewards (14-04)
- CartPole gymnasium (14-05, 14-06, 14-08)
- LunarLander gymnasium (14-07)
- Synthetic offline CartPole trajectories (14-09)
- CartPole demonstrations (14-10)

---

## Prerequisites Chain

- **14-01:** Requires 1-07, 1-09
- **14-02:** Requires 14-01
- **14-03:** Requires 14-02
- **14-04:** Requires 1-07, 14-01
- **14-05:** Requires 14-03, 5-07
- **14-06:** Requires 14-01, 5-06
- **14-07:** Requires 14-06
- **14-08:** Requires 14-03, 14-07
- **14-09:** Requires 14-03, 14-07
- **14-10:** Requires 14-06, 13-06

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 14 — Reinforcement Learning
| Concept | Owner |
|---------|-------|
| MDPs, Bellman equations, value/policy iteration | 14-01 |
| MC evaluation, TD(0), TD(λ), eligibility traces | 14-02 |
| Q-learning, SARSA, TD control | 14-03 |
| Multi-armed bandits, UCB, Thompson sampling, exploration | 14-04 |
| DQN, experience replay, target networks | 14-05 |
| Policy gradient, REINFORCE | 14-06 |
| Actor-critic, A2C, PPO (general RL) | 14-07 |
| Model-based RL, Dyna, MCTS, meta-RL | 14-08 |
| Offline RL (CQL, IQL, conservative methods) | 14-09 |
| Imitation learning, behavioral cloning, DAgger, inverse RL | 14-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ PPO (14-07) teaches the full general PPO algorithm for RL environments with advantage estimation, value networks, and environment rollouts. The clipped surrogate objective was introduced in 13-08 in the RLHF context — reference back to 13-08 but re-derive PPO for the general RL setting with GAE and episodic returns. Both notebooks must be self-contained.
- ⚠️ TD learning (14-02) is foundational for Q-learning (14-03). The pedagogical reordering is intentional.
- ⚠️ Imitation learning (14-10) connects to SFT in Module 13-06 — SFT is behavioral cloning for LLMs. Reference the connection but do not re-teach SFT mechanics.

---

## Special Notes

- Expanded from 5 → 10 topics. Reordered for pedagogical flow: foundations → evaluation → control → exploration → deep → policy → actor-critic → advanced.
- All RL environments should be simple enough to train on CPU in <5 minutes.
- GridWorld should be implemented from scratch as a class with render() method.
