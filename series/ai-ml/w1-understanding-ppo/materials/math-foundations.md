# Mathematical Foundations for PPO

**A Beginner-Friendly Guide to the Math Behind Reinforcement Learning**

This document explains the mathematical concepts you need to understand PPO. Don't worry if you're not a math expert - we'll build everything from first principles with plenty of examples and real-world analogies!

---

## 📚 Table of Contents

1. [Probability Basics](#1-probability-basics)
2. [Expected Value](#2-expected-value)
3. [Logarithms](#3-logarithms)
4. [Gradients & Derivatives](#4-gradients--derivatives)
5. [Discount Factors](#5-discount-factors)
6. [Probability Distributions](#6-probability-distributions)
7. [Entropy](#7-entropy)
8. [KL Divergence](#8-kl-divergence)
9. [Putting It All Together](#9-putting-it-all-together)

---

## 1. Probability Basics

### What You Need to Know

**Probability** measures how likely something is to happen, from 0 (impossible) to 1 (certain).

$$
P(\text{event}) = \frac{\text{number of favorable outcomes}}{\text{total number of outcomes}}
$$

### Real-World Example

**Coin Flip:**
- $P(\text{heads}) = 0.5$ (50% chance)
- $P(\text{tails}) = 0.5$ (50% chance)
- $P(\text{heads}) + P(\text{tails}) = 1.0$ (certainty that one will happen)

**Rolling a Die:**
- $P(\text{rolling a 3}) = \frac{1}{6} \approx 0.167$
- $P(\text{rolling even}) = \frac{3}{6} = 0.5$ (2, 4, or 6)

### In PPO Context

The policy $\pi(a|s)$ outputs probabilities for each action:

$$
\pi(\text{fire left}|s) = 0.1 \quad (10\% \text{ chance})
$$
$$
\pi(\text{fire main}|s) = 0.7 \quad (70\% \text{ chance})
$$
$$
\pi(\text{do nothing}|s) = 0.2 \quad (20\% \text{ chance})
$$

These must sum to 1.0 (100% - something must happen!)

### Practice Problem

**Question:** If your policy says "fire main engine" has 70% probability and "do nothing" has 20%, what's the probability of "fire left"?

**Answer:** $1.0 - 0.7 - 0.2 = 0.1$ (10%)

---

## 2. Expected Value

### What You Need to Know

**Expected value** is the average outcome you'd get if you repeated something many times.

$$
\mathbb{E}[X] = \sum_{i} P(x_i) \cdot x_i
$$

Translation: Multiply each outcome by its probability, then add them up.

### Real-World Example

**Lottery Ticket:**

| Outcome | Probability | Value |
|---------|-------------|-------|
| Win $100 | 0.01 (1%) | $100 |
| Win $10 | 0.10 (10%) | $10 |
| Lose | 0.89 (89%) | $0 |

$$
\mathbb{E}[\text{winnings}] = (0.01 \times 100) + (0.10 \times 10) + (0.89 \times 0) = 1 + 1 + 0 = \$2
$$

**Interpretation:** On average, you win $2 per ticket.

If the ticket costs $3, you're losing money in the long run!

### Dice Rolling Example

**Rolling a standard die:**

$$
\mathbb{E}[\text{roll}] = \frac{1}{6}(1) + \frac{1}{6}(2) + \frac{1}{6}(3) + \frac{1}{6}(4) + \frac{1}{6}(5) + \frac{1}{6}(6)
$$
$$
= \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = \frac{21}{6} = 3.5
$$

You can't roll 3.5, but that's the average over many rolls!

### In PPO Context

The goal of RL is to maximize expected cumulative reward:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} r_t \right]
$$

**Translation:** "Expected total reward when following policy $\pi$"

If you play the game 1000 times with this policy, $J(\pi)$ is your average total score.

### Practice Problem

**Question:** Your agent takes an action. 50% of the time it gets +10 reward, 30% of the time +5, 20% of the time -20. What's the expected reward?

**Answer:** $(0.5 \times 10) + (0.3 \times 5) + (0.2 \times -20) = 5 + 1.5 - 4 = 2.5$

---

## 3. Logarithms

### What You Need to Know

A **logarithm** asks: "What power do I raise a base to, to get this number?"

$$
\log_b(x) = y \quad \text{means} \quad b^y = x
$$

In ML, we almost always use **natural log** ($\ln$ or $\log$), which has base $e \approx 2.718$.

### Why Logarithms?

**Key Properties:**
1. $\log(a \times b) = \log(a) + \log(b)$ (multiplication becomes addition)
2. $\log(a / b) = \log(a) - \log(b)$ (division becomes subtraction)
3. $\log(a^b) = b \cdot \log(a)$ (exponent comes down)

This makes math easier and more numerically stable!

### Real-World Examples

**Example 1: Population Growth**

If a population doubles every year:
- Year 0: 100 people
- Year 1: 200 people (2¹ × 100)
- Year 2: 400 people (2² × 100)
- Year 3: 800 people (2³ × 100)

To find how many years until 1,000,000 people:

$$
2^x \times 100 = 1,000,000 \quad \Rightarrow \quad 2^x = 10,000
$$
$$
x = \log_2(10,000) \approx 13.3 \text{ years}
$$

**Example 2: Sound Volume (Decibels)**

Decibels use log scale because our ears perceive sound logarithmically:
- 20 dB = 10× louder than 10 dB
- 30 dB = 100× louder than 10 dB
- 40 dB = 1,000× louder than 10 dB

### Log Probabilities in PPO

Probabilities are small numbers (0 to 1). Multiplying many small numbers causes numerical issues.

**Example:** $0.9 \times 0.9 \times 0.9 \times \ldots$ (1000 times) = tiny number!

**Solution:** Use log probabilities!

$$
\log(P_1 \times P_2 \times P_3) = \log(P_1) + \log(P_2) + \log(P_3)
$$

Much more stable for computers!

### In PPO Context

The policy ratio uses logs:

$$
r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} = \exp\left(\log \pi_\theta(a|s) - \log \pi_{\theta_{\text{old}}}(a|s)\right)
$$

**Why?** 
- Store log probabilities (numerically stable)
- Subtract logs instead of dividing (easier math)
- Exponentiate to get the ratio

### Practice Problem

**Question:** If $\log(P_{\text{new}}) = -0.5$ and $\log(P_{\text{old}}) = -1.0$, what's the ratio $\frac{P_{\text{new}}}{P_{\text{old}}}$?

**Answer:** 
$$
\log(P_{\text{new}}) - \log(P_{\text{old}}) = -0.5 - (-1.0) = 0.5
$$
$$
\text{ratio} = \exp(0.5) \approx 1.65
$$

The new policy is 1.65× more likely to take that action!

---

## 4. Gradients & Derivatives

### What You Need to Know

A **derivative** tells you how a function changes when you change its input slightly.

$$
\frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

**Translation:** "If I increase $x$ a tiny bit, how much does $f(x)$ change?"

A **gradient** is just a derivative for functions with multiple inputs.

### Real-World Analogy

**Hiking Up a Mountain:**
- You're at some position (x, y)
- The gradient tells you which direction is steepest uphill
- Each step, you move a little bit in that direction

**In ML:**
- You're at some parameter values $\theta$
- The gradient tells you which direction increases your reward most
- Each training step, you adjust $\theta$ in that direction

### Simple Examples

**Example 1: Linear Function**

$$
f(x) = 2x + 3
$$
$$
\frac{df}{dx} = 2
$$

**Meaning:** For every 1 unit increase in $x$, $f(x)$ increases by 2 units.

**Example 2: Quadratic Function**

$$
f(x) = x^2
$$
$$
\frac{df}{dx} = 2x
$$

**Meaning:** 
- At $x = 1$: slope is 2 (going up gently)
- At $x = 5$: slope is 10 (going up steeply)
- At $x = 0$: slope is 0 (flat - this is a minimum!)

### Gradient Descent

**Goal:** Minimize a loss function $L(\theta)$

**Algorithm:**
1. Compute gradient: $\nabla_\theta L$
2. Update parameters: $\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \nabla_\theta L$
3. Repeat!

**Why the minus sign?** The gradient points uphill, but we want to go downhill to minimize loss!

### Visual Example

Imagine $L(\theta)$ is the height of terrain:

```
      /\
     /  \
    /    \    ← You are here (θ = 3, gradient = +2)
   /      \
  /        \  ← Go left (decrease θ) to minimize L
 /          \
/            \
```

Since gradient is positive, decrease $\theta$ to go downhill.

### In PPO Context

The PPO update uses gradients:

$$
\theta_{\text{new}} = \theta_{\text{old}} + \alpha \cdot \nabla_\theta L^{\text{CLIP}}
$$

**But wait!** We're maximizing reward, so we use the **positive** gradient (climbing the reward hill).

The loss function $L^{\text{CLIP}}$ is designed so that minimizing the negative of it maximizes reward.

### Practice Problem

**Question:** If $f(x) = 3x^2 + 2x + 1$, what's the derivative? At $x = 2$, should you increase or decrease $x$ to minimize $f$?

**Answer:**
$$
\frac{df}{dx} = 6x + 2
$$

At $x = 2$: $\frac{df}{dx} = 6(2) + 2 = 14$ (positive)

To minimize, move opposite the gradient: **decrease** $x$.

---

## 5. Discount Factors

### What You Need to Know

A **discount factor** $\gamma$ (gamma) controls how much we care about future rewards vs immediate rewards.

$$
\text{Discounted Reward} = r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 r_3 + \ldots
$$

- $\gamma = 0$: Only care about immediate reward
- $\gamma = 1$: All rewards equally important
- $\gamma = 0.99$: Typical value - future matters but less than present

### Real-World Analogy

**Money Over Time:**

Would you rather have:
- **A:** $100 today
- **B:** $100 in one year

Most people choose A! Money today is worth more than money later (you could invest it, or things might change).

This is **discounting the future**.

If $\gamma = 0.9$:
- $100 today is worth $100
- $100 in 1 year is worth $100 × 0.9 = $90 (to you, now)
- $100 in 2 years is worth $100 × 0.9² = $81
- $100 in 5 years is worth $100 × 0.9⁵ ≈ $59

### Example Calculation

**Rewards:** [+1, +1, +1, +10]

**With $\gamma = 0.9$:**

$$
\text{Return} = 1 + 0.9(1) + 0.9^2(1) + 0.9^3(10)
$$
$$
= 1 + 0.9 + 0.81 + 7.29 = 10.0
$$

**With $\gamma = 0.99$:**

$$
\text{Return} = 1 + 0.99(1) + 0.99^2(1) + 0.99^3(10)
$$
$$
= 1 + 0.99 + 0.98 + 9.70 = 12.67
$$

Higher $\gamma$ = value future rewards more!

### Why Discount?

**Practical Reasons:**
1. **Uncertainty:** Future is uncertain, might never happen
2. **Time preference:** Rewards now are more useful than rewards later
3. **Mathematics:** Infinite sequences converge with $\gamma < 1$
4. **Agent behavior:** Prevents procrastination

**Without Discounting ($\gamma = 1$):**

Agent might delay rewards forever: "I'll get reward later... later... later..."

**With Discounting ($\gamma = 0.99$):**

Agent balances short-term and long-term: "Get some reward now, but also set up for future rewards"

### In PPO Context

The return (what we're trying to maximize) uses discounting:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

**Common Values:**
- **$\gamma = 0.99$:** Most common (99% value retention per step)
- **$\gamma = 0.95$:** Shorter horizon (more myopic)
- **$\gamma = 0.999$:** Very long horizon (far-sighted)

### Practice Problem

**Question:** You get rewards [+5, +5, +5] over 3 steps. What's the discounted return with $\gamma = 0.9$?

**Answer:**
$$
G = 5 + 0.9(5) + 0.9^2(5) = 5 + 4.5 + 4.05 = 13.55
$$

---

## 6. Probability Distributions

### What You Need to Know

A **probability distribution** assigns probabilities to all possible outcomes.

**Discrete distribution:** Outcomes are countable (like rolling a die)

$$
P(X = x_i) = p_i \quad \text{where} \quad \sum_i p_i = 1
$$

### Categorical Distribution (For PPO)

This is what PPO uses for discrete actions!

**Example: 4 actions in Lunar Lander**

| Action | Probability |
|--------|-------------|
| Do nothing | 0.1 |
| Fire left | 0.2 |
| Fire main | 0.6 |
| Fire right | 0.1 |

This is a **categorical distribution** over 4 categories.

### Sampling from a Distribution

**What does "sample from the distribution" mean?**

Imagine a spinner with sections proportional to the probabilities:

```
        ┌─────────────┐
       /   Fire Main  \
      |    (60%)       |
      |                |
      └────────┬───────┘
               │
    ┌──────────┴──────────┐
   /Fire Left (20%)  Fire Right\
  └──────────────────────── (10%)
              
  Do Nothing (10%)
```

Spin it, and wherever it lands is your action!

### PyTorch Example

```python
import torch
from torch.distributions import Categorical

# Action probabilities
probs = torch.tensor([0.1, 0.2, 0.6, 0.1])

# Create distribution
dist = Categorical(probs)

# Sample action
action = dist.sample()  # Returns 0, 1, 2, or 3

# Get log probability of a specific action
log_prob = dist.log_prob(torch.tensor(2))  # P(action=2)
```

### In PPO Context

The policy network outputs logits (unnormalized scores):

$$
\text{logits} = [2.3, 1.5, 4.1, 0.9]
$$

Apply **softmax** to get probabilities:

$$
P(\text{action } i) = \frac{e^{\text{logit}_i}}{\sum_j e^{\text{logit}_j}}
$$

**Example:**
$$
P(\text{fire main}) = \frac{e^{4.1}}{e^{2.3} + e^{1.5} + e^{4.1} + e^{0.9}} = \frac{60.3}{60.3 + 10.0 + 4.5 + 2.5} \approx 0.78
$$

The highest logit gets the highest probability, but others still have a chance!

### Practice Problem

**Question:** Action logits are [1.0, 2.0, 1.0]. Which action is most likely? Roughly what's its probability?

**Answer:** 
- Action 1 has highest logit (2.0), so it's most likely
- $e^{2.0} \approx 7.4$, $e^{1.0} \approx 2.7$
- $P(\text{action 1}) \approx \frac{7.4}{2.7 + 7.4 + 2.7} = \frac{7.4}{12.8} \approx 0.58$ (58%)

---

## 7. Entropy

### What You Need to Know

**Entropy** measures uncertainty or randomness in a probability distribution.

$$
H(P) = -\sum_i P(x_i) \log P(x_i)
$$

- **High entropy:** Very uncertain, many outcomes likely
- **Low entropy:** Very certain, one outcome dominates

### Real-World Analogy

**Weather Prediction:**

**High Entropy (Uncertain):**
- 25% sunny
- 25% cloudy  
- 25% rainy
- 25% snowy

You have no idea what will happen! High uncertainty = high entropy.

**Low Entropy (Certain):**
- 97% sunny
- 1% cloudy
- 1% rainy
- 1% snowy

Pretty sure it'll be sunny! Low uncertainty = low entropy.

### Entropy Calculation Examples

**Example 1: Fair Coin**

- $P(\text{heads}) = 0.5$
- $P(\text{tails}) = 0.5$

$$
H = -(0.5 \log 0.5 + 0.5 \log 0.5) = -(\!-0.35 + (\!-0.35)) = 0.69
$$

Maximum entropy for 2 outcomes!

**Example 2: Biased Coin**

- $P(\text{heads}) = 0.9$
- $P(\text{tails}) = 0.1$

$$
H = -(0.9 \log 0.9 + 0.1 \log 0.1) = -(\!-0.095 + (\!-0.23)) = 0.325
$$

Lower entropy - more predictable!

**Example 3: Unfair Die**

- $P(6) = 0.7$
- $P(\text{others}) = 0.06$ each

$$
H = -(0.7 \log 0.7 + 5 \times 0.06 \log 0.06) \approx 1.13
$$

### In PPO Context

PPO uses an **entropy bonus** to encourage exploration:

$$
L = L^{\text{CLIP}} - c \cdot H(\pi)
$$

**Why?**
- **High entropy:** Agent explores different actions
- **Low entropy:** Agent always picks same action (might be stuck in local optimum)

**During Training:**
- **Early:** Want high entropy (explore)
- **Late:** Want low entropy (exploit what you learned)

Entropy naturally decreases as agent learns!

### Entropy Examples for Actions

**Exploring Agent (High Entropy):**

| Action | Probability |
|--------|-------------|
| Do nothing | 0.25 |
| Fire left | 0.25 |
| Fire main | 0.25 |
| Fire right | 0.25 |

$$
H = -(4 \times 0.25 \log 0.25) = 1.39
$$

**Confident Agent (Low Entropy):**

| Action | Probability |
|--------|-------------|
| Do nothing | 0.05 |
| Fire left | 0.05 |
| Fire main | 0.85 |
| Fire right | 0.05 |

$$
H = -(0.85 \log 0.85 + 3 \times 0.05 \log 0.05) \approx 0.54
$$

### Practice Problem

**Question:** Which has higher entropy?
- **A:** Probabilities [0.5, 0.5]
- **B:** Probabilities [0.99, 0.01]

**Answer:** **A** has higher entropy. Uniform distributions have maximum entropy!

---

## 8. KL Divergence

### What You Need to Know

**KL Divergence** (Kullback-Leibler divergence) measures how different two probability distributions are.

$$
D_{KL}(P || Q) = \sum_i P(x_i) \log \frac{P(x_i)}{Q(x_i)}
$$

- $D_{KL} = 0$: Distributions are identical
- $D_{KL} > 0$: Distributions are different
- Not symmetric: $D_{KL}(P||Q) \neq D_{KL}(Q||P)$

### Real-World Analogy

**Language Models:**

**P = English word frequencies**
**Q = Your attempt to speak English**

High KL divergence means you don't sound like a native speaker!

**Medical Diagnosis:**

**P = Symptoms of Disease A**
**Q = Patient's symptoms**

Low KL divergence means patient likely has Disease A.

### Simple Example

**Distribution P (Old Policy):**
- $P(\text{left}) = 0.5$
- $P(\text{right}) = 0.5$

**Distribution Q (New Policy):**
- $Q(\text{left}) = 0.4$
- $Q(\text{right}) = 0.6$

$$
D_{KL}(P||Q) = 0.5 \log \frac{0.5}{0.4} + 0.5 \log \frac{0.5}{0.6}
$$
$$
= 0.5 \log(1.25) + 0.5 \log(0.833)
$$
$$
= 0.5(0.223) + 0.5(\!-0.182) = 0.112 + (\!-0.091) = 0.021
$$

Small KL divergence = policies are similar!

### In PPO Context

**TRPO (PPO's predecessor)** explicitly constrains KL divergence:

$$
\text{maximize } \mathbb{E}[\ldots] \quad \text{subject to: } D_{KL}(\pi_{\text{old}} || \pi_{\text{new}}) \leq \delta
$$

**Translation:** "Improve the policy, but don't change it too much!"

**PPO's innovation:** Instead of hard constraint, use clipping!

The clipping implicitly limits KL divergence without computing it explicitly.

### Why Limit KL Divergence?

**Problem:** If new policy is too different from old policy:
- Old data (collected with old policy) becomes invalid
- Training becomes unstable
- Policy might collapse

**Solution:** Keep new policy close to old policy
- Old data remains relevant
- Training is stable
- Gradual, consistent improvement

### Visual Intuition

```
Old Policy:        New Policy (Good):    New Policy (Bad):
     │                   │                     │
  ───┼───             ──┼──              ─────│
     │                  │                      └─── Too far!
  Similar actions    Small KL divergence     Large KL divergence
```

### Practice Problem

**Question:** If $D_{KL}(\pi_{\text{old}} || \pi_{\text{new}}) = 0$, what does that mean?

**Answer:** The policies are **identical** - no learning happened! The new policy is exactly the same as the old one.

---

## 9. Putting It All Together

### The Complete PPO Loss Function

Now that you understand all the components, let's break down the full PPO loss:

$$
L^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ L^{\text{CLIP}}_t(\theta) + c_1 L^{VF}_t(\theta) - c_2 H_t(\pi_\theta) \right]
$$

Let's decode each piece:

### Component 1: Clipped Surrogate Objective

$$
L^{\text{CLIP}}_t(\theta) = \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)
$$

**Breaking it down:**

**$r_t(\theta)$** - Probability ratio:
$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

**What it means:** "How much more likely is the new policy to take this action compared to old policy?"

**$A_t$** - Advantage:
$$
A_t = Q(s_t, a_t) - V(s_t)
$$

**What it means:** "Was this action better (+) or worse (-) than average?"

**$\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$** - Clipping:

**What it means:** "Don't let the ratio go below 0.8 or above 1.2 (if $\epsilon = 0.2$)"

**$\min(\ldots)$** - Take minimum:

**What it means:** "Be pessimistic - take the lower estimate to be conservative"

### Component 2: Value Function Loss

$$
L^{VF}_t = (V_\theta(s_t) - V_t^{\text{target}})^2
$$

**What it means:** Mean squared error between predicted value and actual return

**Why?** Train the critic to accurately predict future rewards

### Component 3: Entropy Bonus

$$
H_t(\pi_\theta) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)
$$

**What it means:** Measure of how random the policy is

**Why?** Encourage exploration by rewarding high entropy

### Putting It Together: Worked Example

**Scenario:** Agent took action "fire main engine" in some state.

**Given:**
- $\pi_{\theta_{\text{old}}}(\text{fire main} | s) = 0.3$ (old policy)
- $\pi_\theta(\text{fire main} | s) = 0.4$ (new policy)
- $A = +2.5$ (positive advantage - good action!)
- $\epsilon = 0.2$ (clip range)

**Step 1: Calculate ratio**
$$
r = \frac{0.4}{0.3} = 1.33
$$

**Step 2: Clipped ratio**
$$
\text{clip}(1.33, 0.8, 1.2) = 1.2
$$
(Ratio exceeds upper bound, so it's clipped to 1.2)

**Step 3: Two objectives**
$$
\text{surr1} = 1.33 \times 2.5 = 3.325
$$
$$
\text{surr2} = 1.2 \times 2.5 = 3.0
$$

**Step 4: Take minimum**
$$
L^{\text{CLIP}} = \min(3.325, 3.0) = 3.0
$$

**Interpretation:** 
- The ratio was clipped (1.33 → 1.2)
- This prevents the policy from changing too much
- The agent still learns, but conservatively

### What If Advantage Was Negative?

**Same scenario but $A = -2.5$ (bad action):**

**Step 1-2:** Same as before (ratio = 1.33, clipped to 1.2)

**Step 3: Two objectives**
$$
\text{surr1} = 1.33 \times (\!-2.5) = -3.325
$$
$$
\text{surr2} = 1.2 \times (\!-2.5) = -3.0
$$

**Step 4: Take minimum**
$$
L^{\text{CLIP}} = \min(\!-3.325, \!-3.0) = -3.325
$$

**Interpretation:**
- For bad actions, we want more negative objective
- No clipping happens (surr1 is already more negative)
- Agent strongly discouraged from this action

### The Beauty of PPO

PPO's clipping provides **automatic, adaptive learning rate control**:

1. **Good actions with increased probability:** Clip to prevent excessive updates
2. **Bad actions with decreased probability:** No clipping needed (want aggressive updates)
3. **Policy stays close to old policy:** Data remains valid, training is stable
4. **Simple to implement:** Just a min and clip operation!

### Comparison with TRPO

**TRPO:**
$$
\text{maximize } L(\theta) \quad \text{subject to: } D_{KL}(\pi_{\text{old}} || \pi_{\text{new}}) \leq \delta
$$

**Pros:** Strong theoretical guarantees  
**Cons:** Complex constrained optimization

**PPO:**
$$
\text{maximize } L^{\text{CLIP}}(\theta)
$$

**Pros:** Simple, efficient, works just as well  
**Cons:** Slightly weaker theoretical guarantees (but who cares in practice!)

---

## 📊 Quick Reference Guide

### Key Formulas

| Concept | Formula | What It Means |
|---------|---------|---------------|
| **Expected Value** | $\mathbb{E}[X] = \sum P(x_i) x_i$ | Average outcome |
| **Discount** | $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ | Future rewards matter less |
| **Advantage** | $A(s,a) = Q(s,a) - V(s)$ | Better or worse than average |
| **Entropy** | $H = -\sum P(x_i) \log P(x_i)$ | Measure of randomness |
| **Policy Ratio** | $r = \frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)}$ | How much policy changed |
| **PPO Clip** | $\min(r \cdot A, \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot A)$ | Stable policy update |

### Typical Values in PPO

| Parameter | Symbol | Typical Value | Purpose |
|-----------|--------|---------------|---------|
| Discount factor | $\gamma$ | 0.99 | Balance present/future |
| Clip range | $\epsilon$ | 0.2 | Prevent large policy changes |
| GAE parameter | $\lambda$ | 0.95 | Advantage estimation |
| Value coefficient | $c_1$ | 0.5 | Weight of value loss |
| Entropy coefficient | $c_2$ | 0.01 | Weight of exploration |
| Learning rate | $\alpha$ | 3e-4 | Step size for updates |

### When to Adjust Parameters

**Problem:** Reward not improving
- Decrease learning rate ($3e-4 \to 1e-4$)
- Increase batch size (2048 → 4096)

**Problem:** Training unstable
- Decrease clip range (0.2 → 0.1)
- Increase value coefficient (0.5 → 1.0)

**Problem:** Not exploring
- Increase entropy coefficient (0.01 → 0.02)

**Problem:** Policy changing too fast
- Decrease clip range (0.2 → 0.1)
- Decrease learning rate

---

## 🎓 Practice Problems

### Problem Set 1: Basics

**1.** Calculate expected reward: 30% chance of +10, 50% chance of +5, 20% chance of 0.

**2.** Given $\log(P_1) = -0.7$ and $\log(P_2) = -1.2$, calculate $P_1 / P_2$.

**3.** Calculate discounted return for rewards [2, 3, 5] with $\gamma = 0.9$.

### Problem Set 2: PPO Components

**4.** If old policy has $P(\text{action A}) = 0.4$ and new policy has $P(\text{action A}) = 0.6$, what's the ratio? Would it be clipped with $\epsilon = 0.2$?

**5.** Calculate entropy for probabilities [0.25, 0.25, 0.25, 0.25] and [0.7, 0.1, 0.1, 0.1]. Which agent is exploring more?

**6.** Given ratio = 1.5, advantage = +3, and $\epsilon = 0.2$, calculate the clipped PPO objective.

### Solutions

**1.** $(0.3)(10) + (0.5)(5) + (0.2)(0) = 3 + 2.5 + 0 = 5.5$

**2.** $\exp(-0.7 - (-1.2)) = \exp(0.5) \approx 1.65$

**3.** $2 + 0.9(3) + 0.9^2(5) = 2 + 2.7 + 4.05 = 8.75$

**4.** $r = 0.6/0.4 = 1.5$. Clipped to 1.2 since $1.5 > 1 + 0.2$.

**5.** First: $H = 1.39$ (max entropy), Second: $H \approx 0.80$. First agent explores more.

**6.** $\text{surr1} = 1.5 \times 3 = 4.5$, $\text{surr2} = 1.2 \times 3 = 3.6$, $L = \min(4.5, 3.6) = 3.6$

---

## 🎯 Conclusion

You now understand all the math needed for PPO! The key insights:

1. **Probability & Expected Value:** PPO maximizes expected cumulative reward
2. **Logarithms:** Make probability calculations numerically stable
3. **Gradients:** Tell us how to update the policy
4. **Discounting:** Balance short-term and long-term rewards
5. **Entropy:** Encourage exploration
6. **KL Divergence (conceptually):** Keep policy updates stable
7. **PPO's Clipping:** Simple alternative to complex constraints

The math might look intimidating at first, but each piece has an intuitive meaning. As you implement and experiment with PPO, these concepts will become second nature!

**Remember:** 
- Math is a tool, not the goal
- Intuition matters more than perfect mathematical rigor
- Implementation and experimentation teach you more than equations alone

**Now you're ready to dive into the PPO implementation with mathematical confidence! 🚀**

---

*Mathematical Foundations for PPO*  
*AI & ML Workshop Series | TFDevs*  
*Last Updated: February 11, 2026*
