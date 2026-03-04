# Workshop 1: Git & GitHub Team Workflow

[![Workshop](https://img.shields.io/badge/Workshop-1-blue.svg)]()
[![Duration](https://img.shields.io/badge/Duration-2%20hours-green.svg)]()
[![Level](https://img.shields.io/badge/Level-Beginner--Intermediate-yellow.svg)]()

> **Master collaborative development with Git and GitHub**

Learn how professional development teams use Git and GitHub to collaborate effectively on projects. This workshop covers everything from GitHub Issues for task management to pull requests and code reviews.

---

## 🎯 What You'll Learn

By the end of this workshop, you will be able to:

- ✅ Create and manage GitHub Issues for task tracking
- ✅ Use a branch-based workflow for feature development
- ✅ Create meaningful pull requests with proper descriptions
- ✅ Review code and provide constructive feedback
- ✅ Handle merge conflicts confidently
- ✅ Apply best practices for team collaboration

---

## 📋 Prerequisites

### Required Knowledge
- Basic Git commands: `init`, `add`, `commit`, `push`, `pull`
- Basic command line navigation
- Understanding of what version control is

### Required Setup
- Git installed (version 2.x or higher)
- GitHub account (free tier)
- Text editor (VS Code recommended)
- Terminal/command line access

### Verify Your Setup

Run these commands to verify:

```bash
# Check Git version
git --version

# Check Git configuration
git config --global user.name
git config --global user.email
```

If any of these fail, follow the [setup guide](../README.md#getting-started) in the series README.

---

## 📚 Workshop Materials

### Main Content
- **[Workshop Content](./materials/workshop-1-content.md)** - Complete teaching material with examples
- **[Hands-On Lab](./exercises/hands-on-lab.md)** - Practical exercises with solutions
- **[Instructor Notes](./INSTRUCTOR_NOTES.md)** - Teaching tips and guidance

### Scripts & Tools
- **[Lab Setup Script](./scripts/lab-setup.sh)** - Automated environment setup
- **[Demo Script](./scripts/demo-script.sh)** - Live demonstration script

---

## 🚀 Quick Start

### Option 1: Follow Along (Recommended for First-Time Learners)

1. **Read the workshop content:**
   ```bash
   cd series/collab-dev/w1-git-github-team-workflow
   cat materials/workshop-1-content.md
   ```

2. **Try the examples as you read**

3. **Complete the hands-on lab:**
   ```bash
   cat exercises/hands-on-lab.md
   ```

### Option 2: Hands-On First (For Experienced Users)

1. **Run the lab setup:**
   ```bash
   cd series/collab-dev/w1-git-github-team-workflow/scripts
   ./lab-setup.sh
   ```

2. **Jump straight to exercises:**
   ```bash
   cat ../exercises/hands-on-lab.md
   ```

3. **Refer to content when needed**

---

## 📖 Workshop Outline

### Part 1: GitHub Issues (30 min)
- Understanding issue-driven development
- Creating effective issues
- Labels, milestones, and assignments
- Issue templates and best practices

### Part 2: Branch-Based Workflow (30 min)
- Why branch-based development?
- Creating and switching branches
- Naming conventions
- Feature branches and main branch protection

### Part 3: Pull Requests (30 min)
- Creating meaningful pull requests
- PR descriptions and templates
- Code review process
- Merging strategies

### Part 4: Merge Conflicts (20 min)
- Understanding merge conflicts
- Resolving conflicts step-by-step
- Tools for conflict resolution
- Preventing conflicts

### Part 5: Team Best Practices (10 min)
- Communication guidelines
- Commit message conventions
- When to create PRs vs direct commits
- Team workflow examples

---

## 🏋️ Hands-On Exercises

The workshop includes several practical exercises:

1. **Issue Management Exercise** - Create and organize issues for a project
2. **Feature Development Exercise** - Build a feature using branch workflow
3. **Pull Request Exercise** - Submit and review pull requests
4. **Conflict Resolution Exercise** - Resolve merge conflicts in a safe environment
5. **Team Simulation** - Work through a realistic team scenario

Each exercise includes:
- Clear objectives
- Step-by-step instructions
- Expected outcomes
- Solution hints

---

## 🎥 Video Recording

*Recording will be available after the live workshop session*

---

## 📝 Additional Resources

### Recommended Reading
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)
- [Oh Shit, Git!?!](https://ohshitgit.com/) - Common Git mistakes and fixes

### Tools to Explore
- [GitHub Desktop](https://desktop.github.com/)
- [GitKraken](https://www.gitkraken.com/)
- [GitHub CLI](https://cli.github.com/)
- VS Code extensions: GitLens, GitHub Pull Requests

### Practice Platforms
- [Learn Git Branching](https://learngitbranching.js.org/)
- [GitHub Learning Lab](https://lab.github.com/)

---

## 🐛 Troubleshooting

### Common Issues

**Git not found:**
```bash
# Install on macOS
brew install git

# Install on Ubuntu/Debian
sudo apt-get install git
```

**SSH authentication issues:**
```bash
# Generate new SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key and add to GitHub
cat ~/.ssh/id_ed25519.pub
```

**Permission denied on lab setup:**
```bash
chmod +x scripts/lab-setup.sh
chmod +x scripts/demo-script.sh
```

---

## 💬 Getting Help

- **During the workshop:** Raise your hand or use chat
- **After the workshop:** Open an issue in this repository
- **General questions:** Check the [Discussions](https://github.com/KimangKhenng/tfd-workshop/discussions)

---

## 🤝 Contributing

Found a typo or want to improve the content? Check our [Contributing Guide](../../../CONTRIBUTING.md)!

---

## 📬 Feedback

Your feedback helps us improve! After completing the workshop:
- Let us know what worked well
- Share what was confusing
- Suggest improvements

---

<div align="center">
  <p><strong>Ready to become a collaborative development pro?</strong></p>
  <p>Start with the <a href="./materials/workshop-1-content.md">Workshop Content</a></p>
</div>

---

**Previous:** [Series Overview](../README.md) | **Next:** [Workshop Content](./materials/workshop-1-content.md)
