#!/bin/bash
set -e

# Lab Setup Script for Git & GitHub Team Workflow Workshop
# This script prepares your local environment for the hands-on exercises

echo "════════════════════════════════════════════════════════════"
echo "   Git & GitHub Team Workflow - Lab Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: Check Git installation
echo "Step 1: Checking Git installation..."
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    print_success "Git is installed (version $GIT_VERSION)"
else
    print_error "Git is not installed!"
    echo ""
    echo "Please install Git:"
    echo "  macOS:   brew install git"
    echo "  Ubuntu:  sudo apt-get install git"
    echo "  Windows: Download from https://git-scm.com/download/win"
    exit 1
fi
echo ""

# Step 2: Check Git configuration
echo "Step 2: Checking Git configuration..."
GIT_NAME=$(git config --global user.name || echo "")
GIT_EMAIL=$(git config --global user.email || echo "")

if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
    print_warning "Git is not fully configured"
    echo ""
    echo "Please configure Git with your name and email:"
    echo "  git config --global user.name \"Your Name\""
    echo "  git config --global user.email \"your.email@example.com\""
    echo ""
    read -p "Would you like to configure now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your name: " name
        read -p "Enter your email: " email
        git config --global user.name "$name"
        git config --global user.email "$email"
        print_success "Git configured successfully"
    else
        print_warning "Skipping Git configuration - you'll need to do this later"
    fi
else
    print_success "Git configured with name: $GIT_NAME"
    print_success "Git configured with email: $GIT_EMAIL"
fi
echo ""

# Step 3: Check GitHub CLI (optional but recommended)
echo "Step 3: Checking for GitHub CLI (optional)..."
if command -v gh &> /dev/null; then
    GH_VERSION=$(gh --version | head -n1 | cut -d' ' -f3)
    print_success "GitHub CLI is installed (version $GH_VERSION)"
    
    # Check if authenticated
    if gh auth status &> /dev/null; then
        print_success "GitHub CLI is authenticated"
    else
        print_warning "GitHub CLI is not authenticated"
        echo "  Run: gh auth login"
    fi
else
    print_info "GitHub CLI is not installed (optional, but recommended)"
    echo "  To install: brew install gh"
    echo "  Then authenticate: gh auth login"
fi
echo ""

# Step 4: Create practice directories
echo "Step 4: Setting up practice environment..."
PRACTICE_DIR="$HOME/git-workshop-practice"

if [ -d "$PRACTICE_DIR" ]; then
    print_warning "Practice directory already exists: $PRACTICE_DIR"
    read -p "Do you want to remove it and start fresh? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PRACTICE_DIR"
        print_success "Removed existing practice directory"
    else
        print_info "Keeping existing directory"
    fi
fi

if [ ! -d "$PRACTICE_DIR" ]; then
    mkdir -p "$PRACTICE_DIR"
    cd "$PRACTICE_DIR"
    
    # Initialize a sample repository
    git init
    print_success "Created practice directory: $PRACTICE_DIR"
    
    # Create sample files
    cat > README.md << 'EOF'
# Git Workshop Practice Repository

This is a practice repository for learning Git and GitHub workflows.

## Project: Simple Todo Application

A basic todo list application for learning purposes.

## Features
- Add tasks
- Mark tasks as complete
- Delete tasks

## Getting Started

Open `index.html` in your browser to use the app.
EOF
    
    cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Todo App</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <div class="navbar">
            <h1>Todo App</h1>
        </div>
        <div class="card">
            <form class="task-form">
                <input type="text" id="task-input" placeholder="Enter a new task...">
                <button type="submit" class="btn-primary">Add Task</button>
            </form>
            <div id="task-list"></div>
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
EOF
    
    cat > styles.css << 'EOF'
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    padding: 20px;
}

.container {
    max-width: 600px;
    margin: 0 auto;
}

.navbar {
    background-color: #4a9eff;
    color: white;
    padding: 20px;
    border-radius: 8px 8px 0 0;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 0 0 8px 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.task-form {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}

input[type="text"] {
    flex: 1;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.btn-primary {
    padding: 10px 20px;
    background-color: #4a9eff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.btn-primary:hover {
    background-color: #3a8eef;
}
EOF
    
    cat > script.js << 'EOF'
// Simple Todo App JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('.task-form');
    const input = document.getElementById('task-input');
    const taskList = document.getElementById('task-list');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const taskText = input.value.trim();
        if (taskText) {
            addTask(taskText);
            input.value = '';
        }
    });
    
    function addTask(text) {
        const taskDiv = document.createElement('div');
        taskDiv.className = 'task-item';
        taskDiv.innerHTML = `
            <span>${text}</span>
            <button onclick="this.parentElement.remove()">Delete</button>
        `;
        taskList.appendChild(taskDiv);
    }
});
EOF
    
    cat > .gitignore << 'EOF'
# OS files
.DS_Store
Thumbs.db

# Editor files
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log

# Dependencies (for future use)
node_modules/
EOF
    
    # Initial commit
    git add .
    git commit -m "Initial commit: Setup basic todo app structure"
    
    print_success "Created sample todo app files"
    print_success "Made initial commit"
else
    cd "$PRACTICE_DIR"
    print_info "Using existing practice directory"
fi
echo ""

# Step 5: Provide next steps
echo "════════════════════════════════════════════════════════════"
echo "   Setup Complete! ✓"
echo "════════════════════════════════════════════════════════════"
echo ""
print_success "Your practice environment is ready!"
echo ""
echo "📁 Practice directory: $PRACTICE_DIR"
echo ""
echo "📚 Next Steps:"
echo "   1. Navigate to practice directory:"
echo "      cd $PRACTICE_DIR"
echo ""
echo "   2. Open the workshop exercises:"
echo "      cat ../exercises/hands-on-lab.md"
echo ""
echo "   3. Follow along with the workshop or work through exercises"
echo ""
echo "   4. To see the sample app, open index.html in your browser:"
echo "      open index.html  # macOS"
echo "      xdg-open index.html  # Linux"
echo ""
echo "🔗 Helpful Commands:"
echo "   git status          - Check repository status"
echo "   git log --oneline   - View commit history"
echo "   git branch          - List branches"
echo "   git branch -a       - List all branches (including remote)"
echo ""
echo "❓ Need help?"
echo "   - Workshop materials: ../materials/workshop-1-content.md"
echo "   - Git documentation: git help <command>"
echo "   - GitHub docs: https://docs.github.com"
echo ""
echo "Happy learning! 🚀"
echo ""
