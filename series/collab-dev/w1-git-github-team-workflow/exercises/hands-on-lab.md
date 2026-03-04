# Hands-On Lab: Git & GitHub Team Workflow

> **Estimated Time:** 90 minutes  
> **Difficulty:** Beginner to Intermediate  
> **Prerequisites:** Git installed, GitHub account created

---

## 🎯 Lab Objectives

By completing this lab, you will:
- Create and manage GitHub Issues
- Use branch-based workflow for feature development
- Submit and review pull requests
- Resolve merge conflicts
- Experience a realistic team collaboration scenario

---

## 📋 Prerequisites Check

Before starting, verify your environment:

```bash
# Check Git installation
git --version
# Should output: git version 2.x or higher

# Check Git configuration
git config --global user.name
git config --global user.email

# If not configured, run:
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

✅ **Have a GitHub account?** If not, create one at [github.com](https://github.com)

---

## 🏗️ Lab Setup

### Step 1: Fork the Practice Repository

We've created a practice repository for this lab:

1. **Go to:** `https://github.com/tfdevs/git-workflow-practice`
2. **Click:** "Fork" button (top right)
3. **Select:** Your GitHub account as the destination

You now have your own copy of the repository!

### Step 2: Clone Your Fork Locally

```bash
# Replace YOUR_USERNAME with your actual GitHub username
git clone https://github.com/YOUR_USERNAME/git-workflow-practice.git

# Navigate into the repository
cd git-workflow-practice

# Verify the remote
git remote -v
# Should show your fork as 'origin'
```

### Step 3: Explore the Repository Structure

```bash
# List files
ls -la

# You should see:
# - README.md (project description)
# - index.html (simple web page)
# - styles.css (stylessheet)
# - script.js (JavaScript file)
# - .gitignore
```

---

## 🧪 Exercise 1: GitHub Issues Mastery

### Objective
Practice creating well-structured GitHub Issues for project management.

### Scenario
You're joining a team working on a simple todo application. You need to create issues for upcoming work.

### Tasks

#### Task 1.1: Create a Feature Issue

1. **Navigate to your fork on GitHub**
2. **Go to the "Issues" tab**
3. **Click "New Issue"**
4. **Create a feature issue with this information:**

```markdown
Title: Add dark mode toggle button

## Description
Add a dark mode feature that allows users to switch between light and dark themes.

## Acceptance Criteria
- [ ] Add toggle button in navigation bar
- [ ] Implement dark color scheme
- [ ] Save user preference in localStorage
- [ ] Apply theme on page load based on saved preference

## Design Notes
- Toggle should show moon icon for dark mode, sun icon for light mode
- Dark mode colors:
  - Background: #1a1a1a
  - Text: #f0f0f0
  - Accent: #4a9eff

## Estimated Effort
Medium (3-4 hours)
```

5. **Add labels:** `enhancement`, `frontend`
6. **Assign it to yourself**
7. **Click "Submit New Issue"**

✅ **Checkpoint:** You now have Issue #1 (or similar number)

#### Task 1.2: Create a Bug Issue

Create another issue for a bug:

```markdown
Title: Submit button not responsive on mobile devices

## Description
The submit button on the todo form doesn't respond to taps on iOS Safari
and Chrome mobile browsers.

## Steps to Reproduce
1. Open the application on iPhone (iOS 15+) or Android device
2. Navigate to the todo input form
3. Enter a todo item
4. Tap the "Add Task" button
5. Nothing happens - button doesn't register the tap

## Expected Behavior
Button should add the task to the list and clear the input field

## Actual Behavior
Button visual feedback appears but no action is taken

## Environment
- Device: iPhone 12, Samsung Galaxy S21
- OS: iOS 15.6, Android 12
- Browsers: Safari, Chrome Mobile

## Priority
High - affects mobile users (estimated 40% of user base)

## Possible Cause
May be related to z-index or touch event handling
```

**Add labels:** `bug`, `mobile`, `high-priority`

✅ **Checkpoint:** You now have two distinct issues

#### Task 1.3: Reference Issues in Planning

Create a third issue that references the others:

```markdown
Title: Mobile optimization sprint

## Description
Collection of mobile-related improvements and bug fixes

## Tasks
- [ ] Fix submit button responsiveness (#2)
- [ ] Test dark mode on mobile devices
- [ ] Optimize touch targets (minimum 44x44px)
- [ ] Test on various screen sizes

## Related Issues
- Fixes #2
- Related to #1 (need to test dark mode on mobile)

## Sprint Goal
Improve mobile user experience and fix critical mobile bugs
```

**Add label:** `mobile`, `epic`

✅ **Checkpoint:** Three issues created with proper cross-referencing

---

## 🌿 Exercise 2: Branch-Based Feature Development

### Objective
Learn to use feature branches for isolated development.

### Scenario
You're implementing the dark mode feature from Issue #1.

### Tasks

#### Task 2.1: Create a Feature Branch

```bash
# Ensure you're on main and up to date
git checkout main
git pull origin main

# Create and switch to a new feature branch
git checkout -b feature/dark-mode-toggle

# Verify you're on the new branch
git branch
# Output should show * next to feature/dark-mode-toggle
```

#### Task 2.2: Implement the Feature

Create a new file for the dark mode functionality:

```bash
# Create a new CSS file for dark mode styles
cat > dark-mode.css << 'EOF'
/* Dark Mode Styles */
body.dark-mode {
    background-color: #1a1a1a;
    color: #f0f0f0;
    transition: background-color 0.3s ease, color 0.3s ease;
}

body.dark-mode .navbar {
    background-color: #2a2a2a;
    border-bottom: 1px solid #4a4a4a;
}

body.dark-mode .card {
    background-color: #2a2a2a;
    border-color: #4a4a4a;
}

body.dark-mode .btn-primary {
    background-color: #4a9eff;
    border-color: #4a9eff;
}

.theme-toggle {
    cursor: pointer;
    padding: 8px 12px;
    border: none;
    background: transparent;
    font-size: 1.2rem;
}

.theme-toggle:hover {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}
EOF
```

Update `index.html` to include the toggle button:

```bash
# Add dark mode script to the HTML (this is a simplified version)
cat > dark-mode.js << 'EOF'
// Dark Mode Toggle Functionality
document.addEventListener('DOMContentLoaded', function() {
    // Create toggle button
    const toggleButton = document.createElement('button');
    toggleButton.className = 'theme-toggle';
    toggleButton.innerHTML = '🌙';
    toggleButton.setAttribute('aria-label', 'Toggle dark mode');
    
    // Add to navbar (assuming navbar exists)
    const navbar = document.querySelector('.navbar') || document.body;
    navbar.appendChild(toggleButton);
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        toggleButton.innerHTML = '☀️';
    }
    
    // Toggle theme on button click
    toggleButton.addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        
        // Update button icon and save preference
        if (document.body.classList.contains('dark-mode')) {
            toggleButton.innerHTML = '☀️';
            localStorage.setItem('theme', 'dark');
        } else {
            toggleButton.innerHTML = '🌙';
            localStorage.setItem('theme', 'light');
        }
    });
});
EOF
```

#### Task 2.3: Commit Your Changes

```bash
# Check what files changed
git status

# Add the new files
git add dark-mode.css dark-mode.js

# Commit with a descriptive message referencing the issue
git commit -m "feat: add dark mode toggle button

- Created dark-mode.css with dark theme styles
- Implemented toggle functionality in dark-mode.js
- Saves user preference in localStorage
- Auto-applies theme on page load

Implements #1"

# Check your commit
git log -1
```

#### Task 2.4: Add More Improvements

Make another commit to show incremental development:

```bash
# Improve the dark mode styles
cat >> dark-mode.css << 'EOF'

/* Additional dark mode improvements */
body.dark-mode input,
body.dark-mode textarea {
    background-color: #3a3a3a;
    color: #f0f0f0;
    border-color: #4a4a4a;
}

body.dark-mode a {
    color: #4a9eff;
}

body.dark-mode a:hover {
    color: #6ab0ff;
}
EOF

# Commit the improvement
git add dark-mode.css
git commit -m "style: enhance dark mode contrast for form inputs

Improved readability of input fields in dark mode"
```

#### Task 2.5: Push Your Branch

```bash
# Push the feature branch to GitHub
git push -u origin feature/dark-mode-toggle

# The -u flag sets upstream tracking
# Future pushes can just use: git push
```

✅ **Checkpoint:** Your feature branch is now on GitHub!

---

## 🔄 Exercise 3: Pull Requests and Code Review

### Objective
Create a professional pull request and practice code review.

### Scenario
Your feature is ready for team review. Time to create a pull request!

### Tasks

#### Task 3.1: Create a Pull Request

1. **Go to your GitHub repository in browser**
2. **You'll see a banner:** "feature/dark-mode-toggle had recent pushes"
3. **Click:** "Compare & pull request"
4. **Fill out the PR template:**

```markdown
Title: Add dark mode toggle feature

## What does this PR do?
Implements a dark mode toggle that allows users to switch between light 
and dark themes. The preference is saved and persists across sessions.

## Why is this needed?
Resolves #1 - Adds requested dark mode functionality for better user 
experience in low-light environments.

## What changed?
- Created `dark-mode.css` with comprehensive dark theme styles
- Implemented toggle button in `dark-mode.js`
- Added localStorage support for theme persistence
- Included smooth transitions between themes
- Enhanced contrast for form inputs in dark mode

## How to test
1. Open `index.html` in a browser
2. Look for the 🌙 icon in the navigation
3. Click to toggle to dark mode (icon changes to ☀️)
4. Refresh the page - dark mode should persist
5. Click again to return to light mode

## Screenshots
[You would add screenshots here in a real PR]

## Checklist
- [x] Code follows project style guidelines
- [x] Feature works as expected
- [x] Theme preference persists on reload
- [x] Smooth transition between themes
- [x] Accessible (keyboard navigation works)
- [x] Tested in Chrome, Firefox, Safari
```

5. **Request yourself as a reviewer** (for practice)
6. **Add label:** `enhancement`
7. **Link to issue:** The PR description should automatically link with "Resolves #1"
8. **Click:** "Create pull request"

✅ **Checkpoint:** PR is created and visible on GitHub

#### Task 3.2: Practice Reviewing Your Own PR

This is practice for when you review others' code:

1. **Go to the "Files changed" tab in your PR**
2. **Click the "+" next to a line to add a comment**
3. **Add at least 3 comments:**

```markdown
# Example 1: Question
💡 Could we add a preference for "auto" mode that respects system theme?

# Example 2: Suggestion
Consider adding a transition to the toggle button itself for smoother UX

# Example 3: Approval
✅ Great implementation of localStorage! This will work well.
```

4. **Click "Start a review"** and add comments
5. **Submit review** with "Comment" (since you're practicing)

#### Task 3.3: Address Your Own Feedback (Practice)

Now practice responding to review feedback:

```bash
# Make an improvement based on feedback
cat >> dark-mode.css << 'EOF'

/* Smooth transition for toggle button */
.theme-toggle {
    transition: transform 0.2s ease, background-color 0.2s ease;
}

.theme-toggle:active {
    transform: scale(0.95);
}
EOF

# Commit and push
git add dark-mode.css
git commit -m "style: add transition to toggle button

Improved UX with smooth button animation"

git push
```

**On GitHub:**
- Reply to your own comments: "Great idea! Added in commit abc123"
- The PR automatically updates with new commits

✅ **Checkpoint:** PR updated with feedback addressed

#### Task 3.4: Merge Your Pull Request

1. **In your PR, click "Merge pull request"**
2. **Select "Squash and merge"** for clean history
3. **Edit the commit message if needed:**
```
feat: add dark mode toggle (#1)

- Implements theme toggle with localStorage persistence
- Includes smooth transitions and enhanced form input contrast
```
4. **Click "Confirm squash and merge"**
5. **Delete the branch** when prompted

✅ **Checkpoint:** Feature merged into main!

#### Task 3.5: Clean Up Locally

```bash
# Switch back to main
git checkout main

# Pull the merged changes
git pull origin main

# Delete local feature branch
git branch -d feature/dark-mode-toggle

# Verify it's gone
git branch
# Should only show: * main

# Check that your commit is there
git log --oneline -3
```

---

## ⚔️ Exercise 4: Merge Conflict Resolution

### Objective
Learn to confidently handle merge conflicts.

### Scenario
Two team members edited the same file. You need to resolve the conflict.

### Setup: Create a Conflict Scenario

#### Task 4.1: Create Conflicting Branches

```bash
# Ensure you're on main and up to date
git checkout main
git pull origin main

# Create first branch and make changes
git checkout -b feature/update-header
echo "<h1>Welcome to the Todo App</h1>" > header.html
git add header.html
git commit -m "feat: update header with welcome message"
git push -u origin feature/update-header

# Go back to main and create another branch
git checkout main
git checkout -b feature/add-logo
echo "<h1><img src='logo.png'> Todo App</h1>" > header.html
git add header.html
git commit -m "feat: add logo to header"
git push -u origin feature/add-logo
```

#### Task 4.2: Merge First Branch

```bash
# Merge the first feature
git checkout main
git merge feature/update-header
git push origin main
```

#### Task 4.3: Create and Resolve the Conflict

```bash
# Try to merge the second branch
git merge feature/add-logo

# You'll see:
# Auto-merging header.html
# CONFLICT (content): Merge conflict in header.html
# Automatic merge failed; fix conflicts and then commit the result.

# Check the conflict
cat header.html

# Output will show conflict markers:
# <<<<<<< HEAD
# <h1>Welcome to the Todo App</h1>
# =======
# <h1><img src='logo.png'> Todo App</h1>
# >>>>>>> feature/add-logo
```

#### Task 4.4: Resolve the Conflict

```bash
# Option 1: Manual resolution
# Open header.html in your editor and change to:
cat > header.html << 'EOF'
<h1><img src="logo.png" alt="Logo"> Welcome to the Todo App</h1>
EOF

# Check the file
cat header.html
# Should have no conflict markers!

# Mark as resolved
git add header.html

# Check status
git status
# Should say: "All conflicts fixed but you are still merging"

# Complete the merge
git commit -m "merge: combine welcome message with logo

Resolved conflict in header.html by including both logo and welcome text"

# Push the resolution
git push origin main
```

#### Task 4.5: Clean Up

```bash
# Delete merged branches
git branch -d feature/update-header
git branch -d feature/add-logo

# Delete on GitHub (optional)
git push origin --delete feature/update-header
git push origin --delete feature/add-logo
```

✅ **Checkpoint:** Conflict resolved successfully!

---

## 👥 Exercise 5: Team Collaboration Simulation

### Objective
Experience a realistic team workflow with multiple features being developed simultaneously.

### Scenario
You're part of a 3-person team. Each person is working on a different feature. Simulate all three roles.

### Tasks

#### Task 5.1: The Team Plan

Create three issues for three features:

**Issue A:** "Add task counter"
**Issue B:** "Add task search functionality"  
**Issue C:** "Add task categories"

```bash
# Create these on GitHub Issues tab
# We'll skip the detailed creation for brevity
```

#### Task 5.2: Developer A - Task Counter

```bash
# Developer A starts working
git checkout main
git pull origin main
git checkout -b feature/task-counter

# Add counter functionality
cat > counter.js << 'EOF'
// Task Counter
function updateTaskCounter() {
    const tasks = document.querySelectorAll('.task-item');
    const counter = document.getElementById('task-counter');
    if (counter) {
        counter.textContent = `Total tasks: ${tasks.length}`;
    }
}

// Initialize counter on page load
document.addEventListener('DOMContentLoaded', updateTaskCounter);
EOF

git add counter.js
git commit -m "feat: add task counter display

Shows total number of tasks in the list"
git push -u origin feature/task-counter

# Create PR on GitHub (you can do this via browser)
```

#### Task 5.3: Developer B - Task Search

```bash
# Meanwhile, Developer B starts (from main)
git checkout main
git checkout -b feature/task-search

# Add search functionality
cat > search.js << 'EOF'
// Task Search
function setupSearch() {
    const searchInput = document.getElementById('task-search');
    if (!searchInput) return;
    
    searchInput.addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const tasks = document.querySelectorAll('.task-item');
        
        tasks.forEach(task => {
            const text = task.textContent.toLowerCase();
            if (text.includes(searchTerm)) {
                task.style.display = 'block';
            } else {
                task.style.display = 'none';
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', setupSearch);
EOF

git add search.js
git commit -m "feat: add task search functionality

Users can now filter tasks by searching"
git push -u origin feature/task-search

# Create PR on GitHub
```

#### Task 5.4: Developer C - Task Categories

```bash
# Developer C also starts from main
git checkout main
git checkout -b feature/task-categories

# Add category functionality
cat > categories.js << 'EOF'
// Task Categories
const CATEGORIES = ['Work', 'Personal', 'Shopping', 'Other'];

function addCategoryDropdown() {
    const form = document.querySelector('.task-form');
    if (!form) return;
    
    const select = document.createElement('select');
    select.id = 'task-category';
    select.className = 'category-select';
    
    CATEGORIES.forEach(category => {
        const option = document.createElement('option');
        option.value = category.toLowerCase();
        option.textContent = category;
        select.appendChild(option);
    });
    
    form.appendChild(select);
}

document.addEventListener('DOMContentLoaded', addCategoryDropdown);
EOF

git add categories.js
git commit -m "feat: add task categories

Users can categorize tasks as Work, Personal, Shopping, or Other"
git push -u origin feature/task-categories

# Create PR on GitHub
```

#### Task 5.5: Review and Merge Process

**As Team Lead (or team consensus):**

1. **Review PR from Developer A (task-counter)**
   - Leave a comment: "LGTM! Simple and effective."
   - Approve and merge

2. **Review PR from Developer B (task-search)**
   - Leave feedback: "Could we add a 'clear search' button?"
   - Request changes

3. **Developer B addresses feedback:**
```bash
git checkout feature/task-search

# Add clear button
cat >> search.js << 'EOF'

// Add clear search button
function addClearButton() {
    const searchInput = document.getElementById('task-search');
    const clearBtn = document.createElement('button');
    clearBtn.textContent = '×';
    clearBtn.className = 'clear-search';
    clearBtn.onclick = () => {
        searchInput.value = '';
        searchInput.dispatchEvent(new Event('input'));
    };
    searchInput.parentNode.appendChild(clearBtn);
}

document.addEventListener('DOMContentLoaded', addClearButton);
EOF

git add search.js
git commit -m "feat: add clear search button"
git push
```

4. **Re-review and approve PR from Developer B**
   - "Great addition! Merging now."
   - Merge

5. **Review PR from Developer C (task-categories)**
   - This might now have conflicts with the merged changes!

#### Task 5.6: Developer C Syncs and Resolves

```bash
git checkout feature/task-categories

# Merge main to get latest changes
git merge main

# If conflicts:
# - Resolve them
# - Test that everything works together
# - Commit the resolution

git add .
git commit -m "merge: sync with main and resolve conflicts"
git push

# PR gets updated and can now be merged
```

✅ **Checkpoint:** All three features integrated successfully!

---

## 🎓 Bonus Challenges

### Challenge 1: GitHub Actions Setup

Create a simple GitHub Actions workflow:

```yaml
# .github/workflows/greet.yml
name: Greet Contributors

on:
  pull_request:
    types: [opened]

jobs:
  greet:
    runs-on: ubuntu-latest
    steps:
      - name: Greet
        run: echo "Thanks for your contribution!"
```

Commit and push this file, then create a new PR to see it in action!

### Challenge 2: Issue Templates

Create an issue template:

```markdown
# .github/ISSUE_TEMPLATE/feature_request.md
---
name: Feature Request
about: Suggest a new feature
title: '[FEATURE] '
labels: 'enhancement'
assignees: ''
---

## Feature Description
[Describe the feature you'd like to see]

## Use Case
[Why would this be useful?]

## Proposed Solution
[How do you envision this working?]
```

### Challenge 3: Practice with Rebase

```bash
# Create a branch
git checkout -b feature/rebase-practice

# Make multiple small commits
echo "line 1" > file.txt && git add . && git commit -m "Add line 1"
echo "line 2" >> file.txt && git add . && git commit -m "Add line 2"
echo "line 3" >> file.txt && git add . && git commit -m "Add line 3"

# Interactive rebase to squash commits
git rebase -i HEAD~3
# In the editor, change 'pick' to 'squash' for last 2 commits
```

---

## ✅ Lab Completion Checklist

Confirm you've completed:

- [ ] Created 3+ GitHub Issues with proper descriptions
- [ ] Used feature branches for all development
- [ ] Created and merged at least one Pull Request
- [ ] Practiced code review (even if self-review)
- [ ] Resolved a merge conflict successfully
- [ ] Simulated team collaboration workflow
- [ ] Cleaned up merged branches
- [ ] Understand the complete Git/GitHub workflow

---

## 🎯 What You Learned

**GitHub Issues:**
- Writing clear, actionable issues
- Using labels and assignments
- Linking issues to PRs

**Branch Workflow:**
- Creating feature branches
- Making atomic commits
- Pushing branches to GitHub

**Pull Requests:**
- Writing comprehensive PR descriptions
- Conducting code reviews
- Addressing feedback constructively

**Merge Conflicts:**
- Understanding conflict markers
- Resolving conflicts manually
- Testing after resolution

**Team Collaboration:**
- Coordinating multiple developers
- Integrating concurrent features
- Professional communication patterns

---

## 📚 Additional Practice

Want more practice? Try these:

1. **Fork a real open-source project**
   - Find a "good first issue"
   - Follow their contribution guidelines
   - Submit a real PR!

2. **Collaborate with a friend**
   - Both fork the same repository
   - Make conflicting changes
   - Practice resolving together

3. **Explore advanced Git**
   - `git cherry-pick` - Apply specific commits
   - `git bisect` - Find where bugs were introduced
   - `git stash` - Temporarily save work

4. **Set up a team project**
   - Create a real project with friends
   - Use GitHub Projects board
   - Practice the full workflow

---

## 🐛 Troubleshooting

### Common Issues

**Can't push to GitHub:**
```bash
# Make sure you're pushing to your fork
git remote -v
# Should show YOUR_USERNAME, not tfdevs

# If wrong, update:
git remote set-url origin https://github.com/YOUR_USERNAME/repo.git
```

**Merge conflict too complex:**
```bash
# Abort the merge and try again
git merge --abort

# Or get help:
git status  # See what's conflicted
# Open files in VS Code for visual merge tools
```

**Lost work:**
```bash
# Git rarely loses anything
git reflog  # See all your actions

# Find your commit and restore:
git checkout <commit-hash>
git checkout -b recovery-branch
```

---

## 💬 Get Help

Stuck? Here's how to get help:

1. **Check Git status:** `git status` often tells you what to do
2. **Read error messages:** They usually explain the problem
3. **Use Git docs:** `git help <command>`
4. **Ask in workshops:** Use the discussion forum
5. **Google the error:** Likely someone else had the same issue

---

## 🎉 Congratulations!

You've completed the Git & GitHub Team Workflow hands-on lab! You now have practical experience with:
- Professional Git workflows
- GitHub collaboration features
- Team development practices

Keep practicing these workflows in your projects!

---

**Previous:** [Workshop Content](../materials/workshop-1-content.md) | **Back to:** [Workshop README](../README.md)
