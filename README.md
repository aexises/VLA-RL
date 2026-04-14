# Proposal: Dense Reward Shaping for GRPO-based Training of VLA Models

## 1. Motivation

The VLA training framework in https://arxiv.org/pdf/2509.09674 relies on **binary (sparse) rewards**, which introduces:

- High variance in policy gradient estimates
- Poor credit assignment over long horizons (vision-language-action trajectories)
- Sample inefficiency

Meanwhile, https://arxiv.org/pdf/2505.07395 and related works suggest that **dense reward signals can be constructed without altering the optimal policy**, particularly via **potential-based reward shaping (PBRS)** and reward redistribution.

Key theoretical result:

> Reward shaping of the form  
> \( r'(s,a,s') = r(s,a,s') + \gamma \Phi(s') - \Phi(s) \)  
> preserves optimal policies (policy invariance)  [oai_citation:0‡Next.gr](https://www.next.gr/ai/reinforcement-learning/reward-shaping-in-reinforcement-learning?utm_source=chatgpt.com)

Additionally, dense reward redistribution across tokens or steps can be **equivalent to PBRS** in autoregressive settings  [oai_citation:1‡arXiv](https://arxiv.org/abs/2402.00782?utm_source=chatgpt.com).

---

## 2. Hypothesis

**Replacing binary rewards in GRPO-based VLA training with a structured dense reward derived via potential-based shaping or reward redistribution will:**

1. Improve **sample efficiency**
2. Reduce **variance of gradient estimates**
3. Enable **better temporal credit assignment**
4. Preserve the **optimal policy (no bias)**

---

## 3. Problem Formulation

### 3.1 Original Setting (from VLA + GRPO)

We model VLA training as an MDP:

- State: \( s_t = (o_t, x_{1:t}) \)  
  (visual observation + generated tokens/actions)
- Action: \( a_t \sim \pi_\theta(\cdot | s_t) \)
- Trajectory: \( \tau = (s_0, a_0, ..., s_T) \)

Sparse reward:
\[
R(\tau) \in \{0,1\}
\]

GRPO objective:
\[
\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \right]
\]

where advantage is computed using trajectory-level reward.

---

### 3.2 Proposed Dense Reward Formulation

We define a **potential function**:
\[
\Phi: \mathcal{S} \rightarrow \mathbb{R}
\]

Construct shaped reward:
\[
r'_t = r_t + \gamma \Phi(s_{t+1}) - \Phi(s_t)
\]

Since original reward is terminal:
\[
r_t =
\begin{cases}
R(\tau), & t = T \\
0, & \text{otherwise}
\end{cases}
\]

Thus:
\[
r'_t = \gamma \Phi(s_{t+1}) - \Phi(s_t), \quad t < T
\]
\[
r'_T = R(\tau) - \Phi(s_T)
\]

This produces **dense intermediate rewards**.

---

## 4. Constructing the Potential Function

We propose three **provably grounded** constructions:

### 4.1 Value Function-Based (Bootstrapped PBRS)

\[
\Phi(s_t) = V_\psi(s_t)
\]

- Learned critic approximates expected return
- Equivalent to **bootstrapped reward shaping**  [oai_citation:2‡arXiv](https://arxiv.org/abs/2501.00989?utm_source=chatgpt.com)

---

### 4.2 Reward Model Decomposition (Token-level)

Inspired by dense reward redistribution:

\[
R(\tau) = \sum_{t} w_t
\]

where:
- \( w_t \propto \text{importance}(a_t) \)
- Derived from internal reward model (e.g., attention weights)

This is equivalent to PBRS in autoregressive models  [oai_citation:3‡arXiv](https://arxiv.org/abs/2402.00782?utm_source=chatgpt.com)

---

### 4.3 Goal-distance / Progress Potential

\[
\Phi(s_t) = -d(s_t, s_{goal})
\]

- For VLA: distance in **latent semantic/action space**
- Common dense reward:
\[
r_t = -\|s_t - s_{goal}\|
\]  [oai_citation:4‡Next.gr](https://www.next.gr/ai/reinforcement-learning/reward-shaping-in-reinforcement-learning?utm_source=chatgpt.com)

---

## 5. Modified GRPO Objective

We redefine advantage using dense rewards:

\[
\hat{A}_t = \sum_{k=t}^{T} \gamma^{k-t} r'_k - V_\psi(s_t)
\]

GRPO becomes:

\[
\mathcal{L}_{Dense-GRPO}(\theta) =
\mathbb{E}_{\tau} \left[
\sum_{t} \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t
\right]
\]

---

## 6. Algorithm

### Dense-GRPO for VLA Training
Initialize policy πθ, value function Vψ

for each iteration:
Sample trajectories τ ~ πθ
for each trajectory:
    Compute terminal reward R(τ)

    for t in [0, T]:
        Compute potential Φ(s_t)

    for t in [0, T-1]:
        r'_t = γ Φ(s_{t+1}) - Φ(s_t)

    r'_T = R(τ) - Φ(s_T)

    Compute returns:
        G_t = Σ γ^{k-t} r'_k

    Compute advantages:
        A_t = G_t - Vψ(s_t)

Update θ using GRPO objective
Update ψ via regression to G_t
---

## 7. Theoretical Properties

### 7.1 Policy Invariance

PBRS guarantees:
\[
\pi^*_{original} = \pi^*_{shaped}
\]

Thus:
- No reward hacking (under correct Φ)
- Same optimal behavior  [oai_citation:5‡Emergent Mind](https://www.emergentmind.com/topics/potential-based-reward-shaping?utm_source=chatgpt.com)

---

### 7.2 Variance Reduction

Dense rewards:
- Reduce trajectory-level variance
- Improve gradient signal per timestep

---

### 7.3 Credit Assignment

Dense shaping provides:
- Local gradients for each token/action
- Better alignment in long-horizon VLA tasks

---

## 8. Alternatives from Referenced Literature

### 8.1 Bootstrapped Reward Shaping
- Uses learned value function as Φ
- No manual design required  [oai_citation:6‡arXiv](https://arxiv.org/abs/2501.00989?utm_source=chatgpt.com)

---

### 8.2 Potential Landscape Learning (SLOPE)
- Learns optimistic potential surfaces
- Addresses sparse-reward flatness  [oai_citation:7‡arXiv](https://arxiv.org/abs/2602.03201?utm_source=chatgpt.com)

---

### 8.3 State-space Segmentation
- Decomposes task into subgoals
- Provides structured potentials  [oai_citation:8‡ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167739X24001262?utm_source=chatgpt.com)

---

### 8.4 Hierarchical / Reachability-based shaping
- Uses subgoal reachability
- Useful for long-horizon tasks  [oai_citation:9‡Springer](https://link.springer.com/article/10.1007/s11063-024-11632-x?utm_source=chatgpt.com)

---

## 9. Expected Outcomes

| Property | Binary Reward | Dense PBRS |
|----------|-------------|-----------|
| Sample efficiency | Low | High |
| Variance | High | Lower |
| Credit assignment | Poor | Fine-grained |
| Policy correctness | ✓ | ✓ (guaranteed) |

---

## 10. Key Risks

1. Poorly chosen Φ → slow learning
2. Overfitting to shaping signal
3. Computational overhead (if Φ complex)

---

## 11. Experimental Validation Plan

- Compare:
  - Binary GRPO vs Dense-GRPO
- Metrics:
  - Success rate
  - Sample efficiency
  - Training stability
- Ablations:
  - Φ = value vs learned reward vs heuristic

---

## 12. Conclusion

We propose a **principled integration of dense rewards into GRPO** using **potential-based reward shaping**, ensuring:

- Theoretical correctness (policy invariance)
- Practical gains (efficiency, stability)
- Compatibility with autoregressive VLA models

This bridges:
- Sparse RL (VLA paper)
- Dense reward shaping (2505.07395-style ideas)
- Modern RLHF/token-level reward redistribution

---