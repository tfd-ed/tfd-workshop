# Workshop 1: Understanding Proximal Policy Optimization (PPO)

<div align="center">
  <img src="../../../assets/tfd_logo.jpeg" alt="TFD Logo" width="50"/>
</div>

**AI & ML Workshop Series | Workshop 1 of 5**

Learn one of the most successful reinforcement learning algorithms through hands-on implementation and real-world analogies.

---

## 📋 Workshop Overview

**What You'll Build:** A PPO agent that learns to land a spacecraft in the Lunar Landing environment

**Duration:** 1.5 hours  
**Level:** Beginner (with basic Python and ML knowledge)  
**Format:** Live online workshop + hands-on exercises

### What You'll Learn

✅ Reinforcement Learning fundamentals (agents, environments, rewards, policies)  
✅ Why PPO exists and what problems it solves  
✅ Complete PPO implementation from scratch in PyTorch  
✅ How to map research papers to working code  
✅ Debugging and interpreting RL training metrics  
✅ Practical tips for training RL agents

---

## 🎯 Prerequisites

### Required Knowledge

- **Python Programming:** Comfortable with functions, classes, loops
- **Basic Neural Networks:** Understand forward/backward pass (helpful but not required)
- **NumPy Basics:** Array operations, indexing

### Nice to Have (Not Required)

- PyTorch experience
- Calculus and linear algebra basics
- Previous exposure to machine learning

### Technical Requirements

**Software:**
- Python 3.9 or 3.10 (recommended for Gymnasium compatibility)
- 8GB+ RAM
- 5GB free disk space

**System:**
- macOS, Linux, or Windows 10/11 with WSL2
- GPU optional (CPU is sufficient for workshop exercises)

---

## 🚀 Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/KimangKhenng/tfd-workshop.git
cd tfd-workshop/series/ai-ml/w1-understanding-ppo
```

### Step 2: Run the Setup Script

**For Conda Users (Recommended):**
```bash
cd scripts
chmod +x lab-setup.sh
./lab-setup.sh
# Choose option 1 for Conda
```

**For venv Users:**
```bash
cd scripts
chmod +x lab-setup.sh
./lab-setup.sh
# Choose option 2 for venv
```

The setup script will:
- Create a Python environment
- Install PyTorch, Gymnasium, and dependencies
- Verify installations
- Test the Lunar Lander environment

**Expected time:** 5-10 minutes

### Step 3: Verify Installation

```bash
# Activate your environment
conda activate ppo-workshop  # or source ppo-workshop/bin/activate

# Test imports
python -c "import gymnasium; import torch; print('✓ Ready to go!')"
```

If you see `✓ Ready to go!`, you're all set!

---

## 📚 Workshop Structure

### Part 1: Reinforcement Learning Foundations (10 min)

Learn the basics of RL through real-world analogies:
- What is Reinforcement Learning?
- The Agent-Environment interaction loop
- States, actions, rewards, and policies
- Value functions and Q-functions
- The goal: maximize cumulative reward

**Real-world analogy:** Teaching a dog tricks with treats

### Part 2: Why PPO Exists (10 min)

Understand the evolution of policy gradient methods:
- The problem with vanilla policy gradients
- TRPO: Strong but complex
- PPO's clever solution: clipping
- Why PPO became the industry standard

**Key insight:** Simple clipping prevents policy collapse

### Part 3: PPO Implementation Walkthrough (20 min)

Build PPO step-by-step:
1. Actor-Critic neural network architecture
2. Collecting rollout data from the environment
3. Computing advantages with GAE
4. The PPO update step with clipping
5. Training loop and logging

**Live coding session with detailed explanations**

### Part 4: Mapping Paper to Code (10 min)

Connect the [PPO paper](https://arxiv.org/abs/1707.06347) to implementation:
- Clipped surrogate objective (Equation 7)
- Value function loss
- Entropy bonus
- Generalized Advantage Estimation
- Hyperparameter recommendations

**Learn to read RL research papers**

### Part 5: Live Demo - PPO in Action (10 min)

See PPO learning in real-time:
- Untrained agent: random movements, crashes
- During training: learning to hover
- Trained agent: smooth, confident landings
- Training curves and metrics visualization

**[Demo assets will be added by instructor]**

### Part 6: Gradient Observation & Interpretation (7 min)

Learn to debug RL training:
- What metrics to monitor
- Policy ratio distribution
- Advantage statistics
- Entropy decay
- Value loss trends
- Debugging checklist

**Practical troubleshooting guide**

### Hands-On Exercises (20 min)

Focus on core exercises:

**Exercise 1:** Explore the Lunar Lander environment  
**Exercise 2:** Build the Actor-Critic network  
**Exercise 3:** Implement GAE  
**Exercise 4:** Create the PPO loss function  
**Exercise 5:** Train your first PPO agent  
**Exercise 6:** Evaluate and analyze results

Plus optional challenge exercises for advanced learners!

---

## 📁 Workshop Materials

```
w1-understanding-ppo/
├── README.md (this file)
├── INSTRUCTOR_NOTES.md (for instructors)
├── materials/
│   └── workshop-1-content.md (main teaching content)
├── exercises/
│   └── hands-on-lab.md (structured exercises)
├── scripts/
│   ├── lab-setup.sh (environment setup)
│   └── demo-script.py (live demonstrations)
└── demo-assets/
    └── (GIFs and checkpoints - added by instructor)
```

### For Participants

- **[Workshop Content](materials/workshop-1-content.md)** - Complete teaching materials with theory and code
- **[Hands-On Lab](exercises/hands-on-lab.md)** - Step-by-step exercises with solutions
- **[Setup Script](scripts/lab-setup.sh)** - Automated environment setup

### For Instructors

- **[Instructor Notes](INSTRUCTOR_NOTES.md)** - Teaching guide with demo preparation

---

## 🛠️ Environment Setup Details

### Dependencies

The workshop uses these key libraries:

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.9 or 3.10 | Programming language |
| **PyTorch** | 2.1+ | Deep learning framework |
| **Gymnasium** | 0.29.1 | RL environments (formerly OpenAI Gym) |
| **NumPy** | 1.24.3 | Numerical computing |
| **Matplotlib** | 3.7.2 | Visualization |
| **tqdm** | 4.66.1 | Progress bars |

### Why Python 3.9 or 3.10?

OpenAI Gymnasium (the environment library) has best compatibility with Python 3.9-3.10. Python 3.11+ may work but could have dependency issues.

### GPU vs CPU

**This workshop works fine on CPU!** 
- Training time: 30-60 minutes on modern CPUs
- GPU would be: 10-20 minutes
- For learning, CPU is perfectly adequate

If you have a GPU, PyTorch will use it automatically.

---

## 🎓 Learning Path

### Before the Workshop

**Recommended (but not required):**
1. Watch [David Silver's RL Lecture 1](https://www.youtube.com/watch?v=2pWv7GOvuf0) (1 hour)
2. Read [Spinning Up: RL Intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) (30 min)
3. Review PyTorch basics if rusty

**Don't stress if you can't!** The workshop covers everything you need.

### During the Workshop

1. **Follow along with live coding** - Type the code yourself
2. **Ask questions** - No question is too basic
3. **Experiment** - Try changing parameters, see what happens
4. **Take notes** - Especially on debugging tips

### After the Workshop

1. Complete any unfinished exercises
2. Try the challenge exercises
3. Experiment with different environments (CartPole, BipedalWalker)
4. Read the PPO paper with your new understanding
5. Join our Discord for discussions

---

## 🎮 The Lunar Lander Environment

### What is it?

A spacecraft landing simulation where the agent must:
- Navigate to the landing pad
- Control descent speed
- Maintain proper angle
- Land gently between the flags

### State Space (8 dimensions)

- `[0]` Horizontal position
- `[1]` Vertical position
- `[2]` Horizontal velocity
- `[3]` Vertical velocity
- `[4]` Angle
- `[5]` Angular velocity
- `[6]` Left leg contact (0 or 1)
- `[7]` Right leg contact (0 or 1)

### Action Space (4 discrete actions)

- `0` Do nothing
- `1` Fire left orientation engine
- `2` Fire main engine
- `3` Fire right orientation engine

### Rewards

- +100 to +140 for landing in landing zone
- -100 for crashing
- -0.3 per frame for fuel consumption
- Additional rewards for moving toward landing zone

**Success threshold:** Average reward > 200 over 100 episodes

---

## 💡 Key Concepts

### Reinforcement Learning in One Sentence

*An agent learns to make decisions by trying actions and receiving feedback (rewards), gradually learning which actions lead to better outcomes.*

### PPO in One Sentence

*A policy gradient method that prevents destructive policy updates by clipping the probability ratio, making training stable and sample-efficient.*

### Why PPO Matters

PPO is used in production for:
- **OpenAI:** Fine-tuning GPT models (RLHF)
- **DeepMind:** Robot control and locomotion
- **Gaming:** Dota 2 bots, StarCraft agents
- **Robotics:** Manipulation and navigation
- **Optimization:** Resource allocation, scheduling

**It's simple, stable, and effective** - the go-to algorithm for many RL applications.

---

## 🐛 Common Issues & Solutions

### Setup Issues

**Issue:** `Python version not 3.9 or 3.10`

```bash
# Solution: Install correct version
conda install python=3.10
# or
pyenv install 3.10.0
```

**Issue:** `ModuleNotFoundError: No module named 'gymnasium'`

```bash
# Solution: Activate environment first
conda activate ppo-workshop
# Then install
pip install gymnasium[box2d]
```

**Issue:** `Box2D installation failed`

```bash
# Solution: Install SWIG first (Linux)
sudo apt-get install swig
# Then retry
pip install gymnasium[box2d]
```

### Training Issues

**Issue:** Reward not improving

- Check learning rate (try 3e-4)
- Verify advantage normalization
- Ensure GAE implementation is correct
- Monitor policy ratio distribution

**Issue:** Training is unstable

- Increase batch size (try 4096)
- Reduce learning rate (try 1e-4)
- Check gradient clipping
- Verify no gradients during data collection

See [hands-on-lab.md](exercises/hands-on-lab.md) for detailed troubleshooting.

---

## 📊 Expected Results

### Training Timeline

- **Episodes 0-50:** Random behavior, rewards around -200
- **Episodes 50-200:** Learning to hover, rewards improving to 0
- **Episodes 200-400:** Learning to land, rewards reaching +100
- **Episodes 400+:** Consistent landings, rewards above +200

### Total Training Time

- **CPU:** 30-60 minutes for 500k steps
- **GPU:** 10-20 minutes for 500k steps

### Success Criteria

✓ Average reward > 200 over last 100 episodes  
✓ Successful landings in 80%+ of episodes  
✓ Smooth, controlled descent  
✓ Centered landings between flags

---

## 🌟 What's Next?

### Workshop 2: Advanced Policy Gradient Methods

Learn about PPO's relatives and improvements:
- A2C and A3C algorithms
- Trust Region Policy Optimization (TRPO) deep dive
- Distributed training
- Comparison of different approaches

**Coming Soon!**

### Explore More Environments

After mastering Lunar Lander, try:

**Easier:**
- CartPole-v1 (balance a pole)
- MountainCar-v0 (drive up a hill)

**Harder:**
- BipedalWalker-v3 (walk on two legs)
- Atari games (Pong, Breakout)
- Custom environments

### Implement Variations

Challenge yourself:
- PPO with continuous actions
- Add recurrent layers (LSTM/GRU)
- Multi-agent PPO
- Apply to your own problem

---

## 📚 Additional Resources

### Papers

- [**PPO Paper**](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017 (original paper)
- [TRPO Paper](https://arxiv.org/abs/1502.05477) - Trust Region Policy Optimization
- [GAE Paper](https://arxiv.org/abs/1506.02438) - Generalized Advantage Estimation

### Tutorials & Guides

- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI's comprehensive guide
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Clean, minimal RL implementations
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Production-ready RL library

### Video Courses

- [David Silver's RL Course](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Classic intro to RL
- [DeepMind x UCL Lecture Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm) - Advanced topics

### Books

- [Sutton & Barto: Reinforcement Learning](http://incompleteideas.net/book/) - The RL bible
- [Grokking Deep Reinforcement Learning](https://www.manning.com/books/grokking-deep-reinforcement-learning) - Practical guide

---

## 🤝 Community & Support

### Get Help

**During the Workshop:**
- Ask questions in chat
- Use Discord for technical issues
- Direct message instructors

**After the Workshop:**
- 💬 [Discord Community](#) - Ask questions, share progress
- 📧 Email: ai-ml@tfdevs.com
- 🐛 [GitHub Issues](https://github.com/KimangKhenng/tfd-workshop/issues) - Report problems

### Share Your Progress

- Tweet with #TFDWorkshop #PPO #ReinforcementLearning
- Share trained agent GIFs
- Post in Discord
- Contribute improvements to materials

---

## 📄 License

This workshop material is licensed under the MIT License. Feel free to use, modify, and share!

---

## 🙏 Acknowledgments

This workshop is inspired by and builds upon:
- OpenAI's Spinning Up materials
- CleanRL's implementation philosophy
- The original PPO paper by Schulman et al.
- The RL community's collective knowledge

Special thanks to all contributors and participants who help improve these materials!

---

## 📞 Contact

**TFDevs AI & ML Team**

- 🌐 Website: [tfdevs.com](https://tfdevs.com)
- 📧 Email: ai-ml@tfdevs.com
- 🐦 Twitter: [@TFDevs](https://twitter.com/tfdevs)
- 💬 Discord: [Join our community](#)

**Workshop Questions:** workshops@tfdevs.com

---

<div align="center">

**Ready to start learning PPO?**

**[Read Workshop Content →](materials/workshop-1-content.md)**

**[Start Hands-On Exercises →](exercises/hands-on-lab.md)**

---

*Last Updated: February 11, 2026*  
*Workshop Version: 1.0*

[⬆ Back to AI & ML Series](../README.md) | [⬆ Back to All Workshops](../../../README.md)

</div>
