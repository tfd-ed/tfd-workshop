#!/bin/bash

################################################################################
# Frontend Optimization Workshop - Lab Setup Script
#
# This script sets up the environment for the Frontend Optimization workshop.
# It installs necessary tools and prepares the demo environment.
#
# Usage: ./lab-setup.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
fi

print_header "Frontend Optimization Workshop - Lab Setup"

echo "Detected OS: $OS"
echo ""

################################################################################
# Check Prerequisites
################################################################################

print_header "Step 1: Checking Prerequisites"

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js is installed: $NODE_VERSION"
    
    # Check if version is >= 16
    MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
    if [ "$MAJOR_VERSION" -lt 16 ]; then
        print_warning "Node.js version should be 16 or higher"
        echo "Visit https://nodejs.org/ to upgrade"
    fi
else
    print_error "Node.js is not installed"
    echo ""
    echo "Please install Node.js first:"
    if [ "$OS" == "macos" ]; then
        echo "  brew install node"
    else
        echo "  Visit https://nodejs.org/ or use your package manager"
    fi
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_success "npm is installed: v$NPM_VERSION"
else
    print_error "npm is not installed (should come with Node.js)"
    exit 1
fi

# Check Chrome/Chromium (for Lighthouse)
if command -v google-chrome &> /dev/null || command -v chromium &> /dev/null || [ -d "/Applications/Google Chrome.app" ]; then
    print_success "Chrome/Chromium is installed"
else
    print_warning "Chrome/Chromium not found - needed for Lighthouse"
    echo "Install from: https://www.google.com/chrome/"
fi

################################################################################
# Install Global Tools
################################################################################

print_header "Step 2: Installing Global Tools"

# Install Lighthouse
echo "Installing Lighthouse..."
if npm list -g lighthouse &> /dev/null; then
    print_success "Lighthouse is already installed"
else
    npm install -g lighthouse
    print_success "Lighthouse installed"
fi

# Install http-server (for local testing)
echo ""
echo "Installing http-server..."
if npm list -g http-server &> /dev/null; then
    print_success "http-server is already installed"
else
    npm install -g http-server
    print_success "http-server installed"
fi

################################################################################
# Optional Tools
################################################################################

print_header "Step 3: Checking Optional Tools"

# Check ImageMagick
if command -v convert &> /dev/null; then
    IM_VERSION=$(convert --version | head -1)
    print_success "ImageMagick is installed: $IM_VERSION"
else
    print_warning "ImageMagick not found (optional, for image optimization demos)"
    echo ""
    echo "To install ImageMagick:"
    if [ "$OS" == "macos" ]; then
        echo "  brew install imagemagick"
    else
        echo "  sudo apt-get install imagemagick"
    fi
fi

# Check webp tools
if command -v cwebp &> /dev/null; then
    print_success "WebP tools are installed"
else
    print_warning "WebP tools not found (optional, for WebP conversion)"
    echo ""
    echo "To install WebP tools:"
    if [ "$OS" == "macos" ]; then
        echo "  brew install webp"
    else
        echo "  sudo apt-get install webp"
    fi
fi

# Check jq (for JSON parsing)
if command -v jq &> /dev/null; then
    JQ_VERSION=$(jq --version)
    print_success "jq is installed: $JQ_VERSION"
else
    print_warning "jq not found (optional, for better JSON parsing)"
    echo ""
    echo "To install jq:"
    if [ "$OS" == "macos" ]; then
        echo "  brew install jq"
    else
        echo "  sudo apt-get install jq"
    fi
fi

################################################################################
# Create Lab Directory Structure
################################################################################

print_header "Step 4: Setting Up Lab Directory"

# Create workshop directory
WORKSHOP_DIR="$HOME/frontend-optimization-workshop"

if [ -d "$WORKSHOP_DIR" ]; then
    print_warning "Workshop directory already exists: $WORKSHOP_DIR"
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$WORKSHOP_DIR"
        mkdir -p "$WORKSHOP_DIR"
        print_success "Recreated workshop directory"
    fi
else
    mkdir -p "$WORKSHOP_DIR"
    print_success "Created workshop directory: $WORKSHOP_DIR"
fi

cd "$WORKSHOP_DIR"

# Create subdirectories
mkdir -p demos
mkdir -p exercises
mkdir -p reports

print_success "Created directory structure"

################################################################################
# Setup Sample Project
################################################################################

print_header "Step 5: Setting Up Sample Project"

cd "$WORKSHOP_DIR/exercises"

# Create package.json
cat > package.json << 'EOF'
{
  "name": "frontend-optimization-exercises",
  "version": "1.0.0",
  "description": "Frontend optimization workshop exercises",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "audit": "lighthouse http://localhost:5173 --view"
  },
  "devDependencies": {
    "vite": "^5.0.0"
  }
}
EOF

print_success "Created package.json"

# Install dependencies
echo ""
echo "Installing project dependencies..."
npm install > /dev/null 2>&1
print_success "Dependencies installed"

################################################################################
# Create Quick Reference
################################################################################

print_header "Step 6: Creating Quick Reference"

cd "$WORKSHOP_DIR"

cat > QUICK_REFERENCE.md << 'EOF'
# Frontend Optimization Workshop - Quick Reference

## 📊 Performance Measurement

### Run Lighthouse Audit
```bash
# In browser DevTools
F12 → Lighthouse tab → Analyze page load

# From command line
lighthouse https://example.com --view
```

### Core Web Vitals Targets
- **LCP** (Largest Contentful Paint): < 2.5s
- **INP** (Interaction to Next Paint): < 200ms
- **CLS** (Cumulative Layout Shift): < 0.1

## 🖼️ Image Optimization

### Lazy Loading
```html
<img src="image.jpg" loading="lazy" alt="Description">
```

### Responsive Images
```html
<img 
  srcset="small.jpg 400w, medium.jpg 800w, large.jpg 1200w"
  sizes="(max-width: 600px) 400px, 800px"
  src="medium.jpg"
  alt="Description"
>
```

### Modern Formats
```html
<picture>
  <source srcset="image.avif" type="image/avif">
  <source srcset="image.webp" type="image/webp">
  <img src="image.jpg" alt="Description">
</picture>
```

## ⚡ JavaScript Optimization

### Defer Loading
```html
<script defer src="app.js"></script>
```

### Code Splitting (Vue Router)
```javascript
const routes = [
  {
    path: '/dashboard',
    component: () => import('./Dashboard.vue')
  }
];
```

### Import Only What You Need
```javascript
// ❌ Bad
import _ from 'lodash';

// ✅ Good
import debounce from 'lodash/debounce';
```

## 💾 Caching Headers

### Long-term Caching (versioned assets)
```
Cache-Control: public, max-age=31536000, immutable
```

### No Caching (HTML)
```
Cache-Control: no-cache, no-store, must-revalidate
```

## 🚀 Quick Commands

### Start Dev Server
```bash
cd exercises
npm run dev
```

### Build for Production
```bash
npm run build
```

### Run Lighthouse
```bash
npm run audit
```

### Start Simple HTTP Server
```bash
python3 -m http.server 8000
# or
http-server -p 8000
```

## 🔧 Useful Tools

- **Lighthouse**: Performance auditing
- **WebPageTest**: https://www.webpagetest.org/
- **Chrome DevTools**: F12 → Performance/Network
- **web.dev**: https://web.dev/measure/

## 📚 Resources

- Web.dev: https://web.dev/performance/
- MDN Performance: https://developer.mozilla.org/en-US/docs/Web/Performance
- Chrome DevTools Docs: https://developer.chrome.com/docs/devtools/
EOF

print_success "Created QUICK_REFERENCE.md"

################################################################################
# Final Summary
################################################################################

print_header "Setup Complete! 🎉"

echo "Workshop directory: $WORKSHOP_DIR"
echo ""
echo "What's next?"
echo ""
echo "1. Navigate to workshop directory:"
echo "   cd $WORKSHOP_DIR"
echo ""
echo "2. View quick reference:"
echo "   cat QUICK_REFERENCE.md"
echo ""
echo "3. Start working on exercises:"
echo "   cd exercises"
echo "   npm run dev"
echo ""
echo "4. Run demos:"
echo "   cd demos"
echo "   # Follow instructor demos"
echo ""
echo "5. Run Lighthouse audits:"
echo "   npm run audit"
echo ""

print_success "You're all set! Happy optimizing! 🚀"

# Save setup info
cat > "$WORKSHOP_DIR/SETUP_INFO.txt" << EOF
Frontend Optimization Workshop Setup
=====================================

Setup Date: $(date)
Workshop Directory: $WORKSHOP_DIR

Installed Tools:
- Node.js: $(node --version)
- npm: v$(npm --version)
- Lighthouse: $(lighthouse --version 2>/dev/null || echo "Not installed")

Optional Tools:
- ImageMagick: $(convert --version 2>/dev/null | head -1 || echo "Not installed")
- WebP: $(cwebp -version 2>/dev/null || echo "Not installed")
- jq: $(jq --version 2>/dev/null || echo "Not installed")

Quick Start:
$ cd $WORKSHOP_DIR
$ cat QUICK_REFERENCE.md
EOF

print_success "Setup info saved to SETUP_INFO.txt"
