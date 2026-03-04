#!/bin/bash

# Demo Script for Git & GitHub Team Workflow Workshop
# This script demonstrates key concepts through live examples

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}   $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Function to print commands before executing
print_command() {
    echo -e "${YELLOW}$${NC} ${GREEN}$1${NC}"
}

# Function to pause for explanation
pause() {
    echo ""
    echo -e "${MAGENTA}[Press Enter to continue...]${NC}"
    read -r
}

# Function to show step description
describe_step() {
    echo ""
    echo -e "${BLUE}→ $1${NC}"
    echo ""
}

# Setup temporary demo directory
DEMO_DIR="/tmp/git-team-workflow-demo-$$"
mkdir -p "$DEMO_DIR"
cd "$DEMO_DIR"

# Introduction
clear
print_header "Git & GitHub Team Workflow - Live Demo"
echo "This demo will walk you through a complete team workflow:"
echo "  1. Creating issues"
echo "  2. Branch-based development"
echo "  3. Pull requests"
echo "  4. Merge conflicts"
echo "  5. Team collaboration"
echo ""
echo "📁 Demo directory: $DEMO_DIR"
pause

# Demo 1: Repository Setup
print_header "Demo 1: Repository Setup"

describe_step "Initialize a new Git repository"
print_command "git init demo-project"
git init demo-project
cd demo-project

print_command "git config user.name 'Demo User'"
git config user.name "Demo User"
print_command "git config user.email 'demo@example.com'"
git config user.email "demo@example.com"

describe_step "Create initial project files"
cat > README.md << 'EOF'
# Team Project Demo

This is a demo repository for learning team Git workflows.
EOF

print_command "cat README.md"
cat README.md

print_command "git add README.md"
git add README.md

print_command "git commit -m 'Initial commit: Add README'"
git commit -m "Initial commit: Add README"

print_command "git log --oneline"
git log --oneline
pause

# Demo 2: Feature Branch Workflow
print_header "Demo 2: Feature Branch Workflow"

describe_step "Create a feature branch for adding a login feature"
print_command "git checkout -b feature/user-login"
git checkout -b feature/user-login

print_command "git branch  # Shows all branches with * on current"
git branch

describe_step "Develop the login feature"
cat > login.js << 'EOF'
// User Login Functionality
function login(username, password) {
    console.log('Logging in user:', username);
    
    // Validate inputs
    if (!username || !password) {
        throw new Error('Username and password required');
    }
    
    // Authenticate
    return authenticateUser(username, password);
}

function authenticateUser(username, password) {
    // Placeholder authentication logic
    return { success: true, user: username };
}
EOF

print_command "cat login.js  # Review the code"
cat login.js

describe_step "Make a commit for this feature"
print_command "git add login.js"
git add login.js

print_command "git commit -m 'feat: add user login functionality'"
git commit -m "feat: add user login functionality"

print_command "git log --oneline --graph --all"
git log --oneline --graph --all

pause

# Demo 3: Multiple Feature Branches
print_header "Demo 3: Multiple Features in Parallel"

describe_step "Switch back to main and create another feature branch"
print_command "git checkout main"
git checkout main

print_command "git checkout -b feature/user-profile"
git checkout -b feature/user-profile

cat > profile.js << 'EOF'
// User Profile Management
function getUserProfile(userId) {
    console.log('Fetching profile for user:', userId);
    return {
        id: userId,
        name: 'Demo User',
        email: 'demo@example.com'
    };
}

function updateProfile(userId, updates) {
    console.log('Updating profile:', userId, updates);
    return { success: true };
}
EOF

print_command "git add profile.js"
git add profile.js

print_command "git commit -m 'feat: add user profile management'"
git commit -m "feat: add user profile management"

describe_step "View all branches and their commits"
print_command "git log --oneline --graph --all --decorate"
git log --oneline --graph --all --decorate

pause

# Demo 4: Merging Features
print_header "Demo 4: Merging Features"

describe_step "Merge the login feature into main"
print_command "git checkout main"
git checkout main

print_command "git merge feature/user-login"
git merge feature/user-login -m "Merge feature/user-login into main"

print_command "git log --oneline --graph"
git log --oneline --graph

describe_step "Now merge the profile feature"
print_command "git merge feature/user-profile"
git merge feature/user-profile -m "Merge feature/user-profile into main"

print_command "git log --oneline --graph --all"
git log --oneline --graph --all

describe_step "Clean up merged branches"
print_command "git branch -d feature/user-login"
git branch -d feature/user-login

print_command "git branch -d feature/user-profile"
git branch -d feature/user-profile

print_command "git branch  # Show remaining branches"
git branch

pause

# Demo 5: Creating a Merge Conflict
print_header "Demo 5: Handling Merge Conflicts"

describe_step "Create two branches that will conflict"
print_command "git checkout -b feature/update-readme-1"
git checkout -b feature/update-readme-1

cat > README.md << 'EOF'
# Team Project Demo

This is a demo repository for learning team Git workflows.

## Features
- User authentication system
- Profile management
EOF

print_command "git add README.md"
git add README.md

print_command "git commit -m 'docs: add features section to README'"
git commit -m "docs: add features section to README"

describe_step "Go back to main and create another conflicting branch"
print_command "git checkout main"
git checkout main

print_command "git checkout -b feature/update-readme-2"
git checkout -b feature/update-readme-2

cat > README.md << 'EOF'
# Team Project Demo

This is a demo repository for learning team Git workflows.

## Getting Started
1. Clone the repository
2. Run the application
EOF

print_command "git add README.md"
git add README.md

print_command "git commit -m 'docs: add getting started section'"
git commit -m "docs: add getting started section"

pause

describe_step "Merge first branch - this will work fine"
print_command "git checkout main"
git checkout main

print_command "git merge feature/update-readme-1"
git merge feature/update-readme-1 -m "Merge feature/update-readme-1"

print_command "cat README.md  # See the merged content"
cat README.md

describe_step "Try to merge second branch - this will create a conflict!"
print_command "git merge feature/update-readme-2"

set +e  # Don't exit on error
git merge feature/update-readme-2 2>&1 | tee merge_output.txt
MERGE_STATUS=$?
set -e

if [ $MERGE_STATUS -ne 0 ]; then
    echo ""
    echo -e "${RED}⚔️  Merge conflict detected!${NC}"
    echo ""
    
    describe_step "Check the status to see what's conflicted"
    print_command "git status"
    git status
    
    describe_step "Look at the conflict markers in the file"
    print_command "cat README.md"
    echo ""
    cat README.md
    echo ""
    
    describe_step "Resolve the conflict by keeping both changes"
    cat > README.md << 'EOF'
# Team Project Demo

This is a demo repository for learning team Git workflows.

## Features
- User authentication system
- Profile management

## Getting Started
1. Clone the repository
2. Run the application
EOF
    
    print_command "cat README.md  # Show resolved version"
    cat README.md
    
    describe_step "Mark conflict as resolved and complete the merge"
    print_command "git add README.md"
    git add README.md
    
    print_command "git commit -m 'merge: resolve README conflict by including both sections'"
    git commit -m "merge: resolve README conflict by including both sections"
    
    echo ""
    echo -e "${GREEN}✓ Conflict resolved successfully!${NC}"
fi

pause

# Demo 6: Team Collaboration Simulation
print_header "Demo 6: Team Collaboration Scenario"

describe_step "Simulate three developers working simultaneously"

# Developer 1: Bug fix
print_command "git checkout -b hotfix/login-validation"
git checkout -b hotfix/login-validation

cat >> login.js << 'EOF'

// Enhanced validation
function validatePassword(password) {
    if (password.length < 8) {
        throw new Error('Password must be at least 8 characters');
    }
    return true;
}
EOF

git add login.js
git commit -m "fix: add password length validation"

echo -e "${BLUE}Developer 1 completed hotfix${NC}"

# Developer 2: New feature
print_command "git checkout main"
git checkout main

print_command "git checkout -b feature/logout"
git checkout -b feature/logout

cat > logout.js << 'EOF'
// Logout functionality
function logout() {
    console.log('User logged out');
    clearSession();
    redirectToLogin();
}

function clearSession() {
    // Clear user session
}

function redirectToLogin() {
    // Redirect to login page
}
EOF

git add logout.js
git commit -m "feat: add logout functionality"

echo -e "${BLUE}Developer 2 completed feature${NC}"

# Developer 3: Refactoring
print_command "git checkout main"
git checkout main

print_command "git checkout -b refactor/auth-module"
git checkout -b refactor/auth-module

mkdir -p auth
cat > auth/index.js << 'EOF'
// Centralized authentication module
export { login } from './login.js';
export { logout } from './logout.js';
export { authenticateUser } from './auth.js';
EOF

git add auth/
git commit -m "refactor: create centralized auth module"

echo -e "${BLUE}Developer 3 completed refactoring${NC}"

pause

describe_step "Now integrate all changes (Team Lead coordinates)"
print_command "git checkout main"
git checkout main

echo ""
echo "Merging hotfix first (highest priority)..."
print_command "git merge hotfix/login-validation"
git merge hotfix/login-validation -m "Merge hotfix: login validation"

echo ""
echo "Merging logout feature..."
print_command "git merge feature/logout"
git merge feature/logout -m "Merge feature: logout"

echo ""
echo "Merging refactoring..."
print_command "git merge refactor/auth-module"
git merge refactor/auth-module -m "Merge refactor: auth module"

describe_step "View the final project history"
print_command "git log --oneline --graph --all"
git log --oneline --graph --all

describe_step "See all project files"
print_command "ls -la"
ls -la

print_command "tree -L 2  # If tree is installed"
tree -L 2 2>/dev/null || find . -maxdepth 2 -not -path '*/\.git/*' | sed 's|^\./||' | sort

pause

# Demo 7: Best Practices Summary
print_header "Demo 7: Best Practices Demonstrated"

echo "✅ What we demonstrated:"
echo ""
echo "1. Branch-based workflow"
echo "   • Created feature branches for each task"
echo "   • Kept main branch stable"
echo ""
echo "2. Clear commit messages"
echo "   • Used conventional commit format"
echo "   • feat:, fix:, docs:, refactor: prefixes"
echo ""
echo "3. Merge conflict resolution"
echo "   • Identified conflicts"
echo "   • Resolved by combining changes"
echo "   • Tested after resolution"
echo ""
echo "4. Team collaboration"
echo "   • Multiple developers working in parallel"
echo "   • Coordinated integration"
echo "   • Priority-based merging"
echo ""
echo "5. Clean history"
echo "   • Logical commit organization"
echo "   • Deleted merged branches"
echo "   • Clear project timeline"
echo ""

pause

# Cleanup
print_header "Demo Complete!"

echo "This demo covered the essential Git team workflow concepts."
echo ""
echo "📁 Demo files are in: $DEMO_DIR"
echo "   (Feel free to explore or delete this directory)"
echo ""
echo "📚 Next steps:"
echo "   1. Try the hands-on exercises in ../exercises/hands-on-lab.md"
echo "   2. Practice these workflows in your own projects"
echo "   3. Experiment with GitHub features online"
echo ""
echo "🎓 Key commands to remember:"
echo "   git branch <name>           - Create a branch"
echo "   git checkout -b <name>      - Create and switch to branch"
echo "   git merge <branch>          - Merge branch into current"
echo "   git log --oneline --graph   - Visualize history"
echo "   git status                  - Check repository state"
echo ""
echo -e "${GREEN}Happy collaborating! 🚀${NC}"
echo ""
