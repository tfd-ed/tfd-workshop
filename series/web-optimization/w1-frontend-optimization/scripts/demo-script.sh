#!/bin/bash

################################################################################
# Frontend Optimization Workshop - Demo Script
# 
# This script demonstrates key frontend optimization techniques including:
# - Performance measurement with Lighthouse
# - Image optimization
# - JavaScript optimization
# - Caching strategies
# - Real-world impact measurement
#
# Usage: ./demo-script.sh
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

pause() {
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read -r
}

################################################################################
# Demo 1: Performance Measurement
################################################################################

demo_performance_measurement() {
    print_header "Demo 1: Performance Measurement with Lighthouse"
    
    echo "We'll measure the performance of a real website using Lighthouse."
    echo ""
    
    # Check if lighthouse is installed
    if ! command -v lighthouse &> /dev/null; then
        print_warning "Lighthouse not found. Installing..."
        npm install -g lighthouse
        print_success "Lighthouse installed"
    else
        print_success "Lighthouse is already installed"
    fi
    
    echo ""
    echo "Let's audit a popular website (example.com):"
    pause
    
    # Run Lighthouse
    echo ""
    print_success "Running Lighthouse audit..."
    lighthouse https://example.com \
        --only-categories=performance \
        --output=json \
        --output-path=./lighthouse-report.json \
        --quiet
    
    # Extract key metrics
    echo ""
    print_header "Performance Results"
    
    if command -v jq &> /dev/null; then
        SCORE=$(jq '.categories.performance.score * 100' lighthouse-report.json)
        FCP=$(jq '.audits["first-contentful-paint"].numericValue / 1000' lighthouse-report.json)
        LCP=$(jq '.audits["largest-contentful-paint"].numericValue / 1000' lighthouse-report.json)
        TBT=$(jq '.audits["total-blocking-time"].numericValue' lighthouse-report.json)
        CLS=$(jq '.audits["cumulative-layout-shift"].numericValue' lighthouse-report.json)
        
        echo "Performance Score: ${SCORE}/100"
        echo "First Contentful Paint: ${FCP}s"
        echo "Largest Contentful Paint: ${LCP}s"
        echo "Total Blocking Time: ${TBT}ms"
        echo "Cumulative Layout Shift: ${CLS}"
    else
        print_warning "Install 'jq' for better JSON parsing"
        cat lighthouse-report.json | grep -E '(score|numericValue)' | head -10
    fi
    
    echo ""
    print_success "View full report: cat lighthouse-report.json | jq '.'"
    pause
}

################################################################################
# Demo 2: Image Optimization
################################################################################

demo_image_optimization() {
    print_header "Demo 2: Image Optimization"
    
    echo "We'll demonstrate the impact of image optimization."
    echo ""
    
    # Create demo directory
    mkdir -p image-demo
    cd image-demo
    
    # Download a sample image
    print_success "Downloading sample image..."
    
    if command -v curl &> /dev/null; then
        curl -s -o original.jpg https://picsum.photos/1920/1080
        print_success "Downloaded original image"
    else
        print_error "curl not found. Please install curl."
        return 1
    fi
    
    # Check original size
    ORIGINAL_SIZE=$(du -h original.jpg | cut -f1)
    echo ""
    echo "Original image size: ${ORIGINAL_SIZE}"
    
    pause
    
    # Optimize with ImageMagick (if available)
    if command -v convert &> /dev/null; then
        print_success "Optimizing image with ImageMagick..."
        
        # Reduce quality
        convert original.jpg -quality 85 optimized-85.jpg
        convert original.jpg -quality 70 optimized-70.jpg
        
        # Convert to WebP
        if command -v cwebp &> /dev/null; then
            cwebp -q 80 original.jpg -o optimized.webp
            print_success "Created WebP version"
        fi
        
        echo ""
        print_header "Size Comparison"
        echo "Original:      $(du -h original.jpg | cut -f1)"
        echo "Quality 85%:   $(du -h optimized-85.jpg | cut -f1)"
        echo "Quality 70%:   $(du -h optimized-70.jpg | cut -f1)"
        
        if [ -f optimized.webp ]; then
            echo "WebP format:   $(du -h optimized.webp | cut -f1)"
        fi
        
        # Calculate savings
        ORIG_BYTES=$(stat -f%z original.jpg 2>/dev/null || stat -c%s original.jpg)
        OPT_BYTES=$(stat -f%z optimized-85.jpg 2>/dev/null || stat -c%s optimized-85.jpg)
        SAVINGS=$((100 - (OPT_BYTES * 100 / ORIG_BYTES)))
        
        echo ""
        print_success "Savings with quality 85%: ${SAVINGS}%"
        
    else
        print_warning "ImageMagick not installed."
        echo "Install with: brew install imagemagick (macOS) or apt-get install imagemagick (Linux)"
    fi
    
    cd ..
    pause
}

################################################################################
# Demo 3: Lazy Loading Demo
################################################################################

demo_lazy_loading() {
    print_header "Demo 3: Lazy Loading Implementation"
    
    echo "We'll create an HTML page demonstrating lazy loading."
    echo ""
    
    mkdir -p lazy-loading-demo
    cd lazy-loading-demo
    
    # Create index.html
    cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lazy Loading Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .info {
            background: #f0f0f0;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .gallery img {
            width: 100%;
            height: 300px;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stats {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #333;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>🚀 Lazy Loading Demo</h1>
    
    <div class="info">
        <h2>What's happening?</h2>
        <p><strong>Open DevTools → Network tab</strong> and scroll down slowly.</p>
        <p>Notice how images only load when they're about to enter the viewport!</p>
        <p>This saves bandwidth and improves initial page load time.</p>
    </div>
    
    <div class="info">
        <h2>Above the fold content</h2>
        <p>This content is immediately visible. Scroll down to see lazy loading in action!</p>
    </div>
    
    <div class="gallery">
        <img loading="lazy" src="https://picsum.photos/400/300?random=1" alt="Image 1">
        <img loading="lazy" src="https://picsum.photos/400/300?random=2" alt="Image 2">
        <img loading="lazy" src="https://picsum.photos/400/300?random=3" alt="Image 3">
        <img loading="lazy" src="https://picsum.photos/400/300?random=4" alt="Image 4">
        <img loading="lazy" src="https://picsum.photos/400/300?random=5" alt="Image 5">
        <img loading="lazy" src="https://picsum.photos/400/300?random=6" alt="Image 6">
        <img loading="lazy" src="https://picsum.photos/400/300?random=7" alt="Image 7">
        <img loading="lazy" src="https://picsum.photos/400/300?random=8" alt="Image 8">
        <img loading="lazy" src="https://picsum.photos/400/300?random=9" alt="Image 9">
        <img loading="lazy" src="https://picsum.photos/400/300?random=10" alt="Image 10">
        <img loading="lazy" src="https://picsum.photos/400/300?random=11" alt="Image 11">
        <img loading="lazy" src="https://picsum.photos/400/300?random=12" alt="Image 12">
    </div>
    
    <div class="stats" id="stats">
        <div>Images Loaded: <span id="loaded">0</span></div>
        <div>Total Images: 12</div>
    </div>
    
    <script>
        // Track how many images have loaded
        let loadedCount = 0;
        const images = document.querySelectorAll('img[loading="lazy"]');
        
        images.forEach(img => {
            img.addEventListener('load', () => {
                loadedCount++;
                document.getElementById('loaded').textContent = loadedCount;
                console.log(`Image loaded: ${img.alt}`);
            });
        });
    </script>
</body>
</html>
EOF

    print_success "Created lazy-loading-demo/index.html"
    
    echo ""
    echo "To see the demo:"
    echo "  1. cd lazy-loading-demo"
    echo "  2. python3 -m http.server 8000"
    echo "  3. Open http://localhost:8000 in your browser"
    echo "  4. Open DevTools → Network tab"
    echo "  5. Scroll down and watch images load on demand!"
    
    cd ..
    pause
}

################################################################################
# Demo 4: JavaScript Bundle Analysis
################################################################################

demo_bundle_analysis() {
    print_header "Demo 4: JavaScript Bundle Analysis"
    
    echo "We'll create a simple app and analyze its bundle size."
    echo ""
    
    mkdir -p bundle-demo
    cd bundle-demo
    
    # Initialize npm project
    print_success "Creating demo project..."
    npm init -y > /dev/null 2>&1
    
    # Install dependencies
    print_success "Installing dependencies..."
    npm install --save lodash moment > /dev/null 2>&1
    npm install --save-dev webpack webpack-cli > /dev/null 2>&1
    
    # Create source files
    mkdir -p src
    
    # Bad example - importing entire libraries
    cat > src/bad-example.js << 'EOF'
// ❌ BAD: Importing entire libraries
import _ from 'lodash';
import moment from 'moment';

// We only use a small fraction
const debounced = _.debounce(() => console.log('Debounced'), 300);
const now = moment().format('YYYY-MM-DD');

console.log('Date:', now);
EOF

    # Good example - importing only what's needed
    cat > src/good-example.js << 'EOF'
// ✅ GOOD: Importing only what we need
import debounce from 'lodash/debounce';
import { format } from 'date-fns';

// Much smaller bundle!
const debounced = debounce(() => console.log('Debounced'), 300);
const now = format(new Date(), 'yyyy-MM-dd');

console.log('Date:', now);
EOF

    # Create webpack config
    cat > webpack.config.js << 'EOF'
const path = require('path');

module.exports = {
  mode: 'production',
  entry: {
    bad: './src/bad-example.js',
    good: './src/good-example.js'
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].bundle.js'
  }
};
EOF

    print_success "Building bundles..."
    npx webpack > /dev/null 2>&1
    
    echo ""
    print_header "Bundle Size Comparison"
    
    BAD_SIZE=$(du -h dist/bad.bundle.js | cut -f1)
    GOOD_SIZE=$(du -h dist/good.bundle.js | cut -f1)
    
    echo "❌ Bad approach (import entire library):  ${BAD_SIZE}"
    echo "✅ Good approach (import what you need): ${GOOD_SIZE}"
    
    # Calculate percentage
    BAD_BYTES=$(stat -f%z dist/bad.bundle.js 2>/dev/null || stat -c%s dist/bad.bundle.js)
    GOOD_BYTES=$(stat -f%z dist/good.bundle.js 2>/dev/null || stat -c%s dist/good.bundle.js)
    SAVINGS=$((100 - (GOOD_BYTES * 100 / BAD_BYTES)))
    
    echo ""
    print_success "Savings: ${SAVINGS}% reduction in bundle size!"
    
    echo ""
    echo "Key takeaway: Import only what you need!"
    
    cd ..
    pause
}

################################################################################
# Demo 5: Caching Demonstration
################################################################################

demo_caching() {
    print_header "Demo 5: HTTP Caching Demonstration"
    
    echo "We'll demonstrate how caching headers work."
    echo ""
    
    mkdir -p caching-demo
    cd caching-demo
    
    # Create simple server with Node.js
    cat > server.js << 'EOF'
const http = require('http');
const fs = require('fs');
const path = require('path');

const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  if (req.url === '/') {
    // HTML - no cache
    res.writeHead(200, {
      'Content-Type': 'text/html',
      'Cache-Control': 'no-cache, no-store, must-revalidate'
    });
    res.end(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Caching Demo</title>
        <link rel="stylesheet" href="/styles.css">
      </head>
      <body>
        <h1>HTTP Caching Demo</h1>
        <p>Check the Network tab in DevTools!</p>
        <p>Reload the page and see what gets cached.</p>
        <script src="/app.js"></script>
      </body>
      </html>
    `);
  }
  else if (req.url === '/styles.css') {
    // CSS - cache for 1 year
    res.writeHead(200, {
      'Content-Type': 'text/css',
      'Cache-Control': 'public, max-age=31536000, immutable'
    });
    res.end('body { font-family: Arial; padding: 20px; }');
  }
  else if (req.url === '/app.js') {
    // JS - cache for 1 year
    res.writeHead(200, {
      'Content-Type': 'application/javascript',
      'Cache-Control': 'public, max-age=31536000, immutable'
    });
    res.end('console.log("App loaded with caching!");');
  }
  else {
    res.writeHead(404);
    res.end('Not found');
  }
});

const PORT = 3000;
server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log('\nCache headers:');
  console.log('  - HTML: no-cache (always fresh)');
  console.log('  - CSS/JS: cached for 1 year (immutable)');
  console.log('\nOpen DevTools → Network and reload to see caching!');
});
EOF

    print_success "Created caching demo server"
    
    echo ""
    echo "To run the demo:"
    echo "  1. cd caching-demo"
    echo "  2. node server.js"
    echo "  3. Open http://localhost:3000"
    echo "  4. Open DevTools → Network tab"
    echo "  5. Reload page multiple times"
    echo "  6. Notice CSS/JS load from cache instantly!"
    
    cd ..
    pause
}

################################################################################
# Demo 6: Core Web Vitals Measurement
################################################################################

demo_web_vitals() {
    print_header "Demo 6: Measuring Core Web Vitals"
    
    echo "We'll create a page that measures Core Web Vitals in real-time."
    echo ""
    
    mkdir -p web-vitals-demo
    cd web-vitals-demo
    
    # Initialize project
    npm init -y > /dev/null 2>&1
    npm install web-vitals > /dev/null 2>&1
    
    # Create demo page
    cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Core Web Vitals Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .metric {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-name {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        .metric-unit {
            font-size: 14px;
            color: #999;
        }
        .good { color: #0cce6b; }
        .needs-improvement { color: #ffa400; }
        .poor { color: #ff4e42; }
        .hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 20px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <div class="hero">
        <h1>📊 Core Web Vitals Demo</h1>
        <p>Real-time performance monitoring</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-name">LCP</div>
            <div class="metric-value" id="lcp">-</div>
            <div class="metric-unit">seconds</div>
        </div>
        
        <div class="metric">
            <div class="metric-name">FID</div>
            <div class="metric-value" id="fid">-</div>
            <div class="metric-unit">milliseconds</div>
        </div>
        
        <div class="metric">
            <div class="metric-name">CLS</div>
            <div class="metric-value" id="cls">-</div>
            <div class="metric-unit">score</div>
        </div>
    </div>
    
    <div style="background: white; padding: 20px; border-radius: 8px;">
        <h2>What are Core Web Vitals?</h2>
        <p><strong>LCP (Largest Contentful Paint):</strong> Loading performance</p>
        <ul>
            <li>Good: < 2.5s</li>
            <li>Needs Improvement: 2.5-4s</li>
            <li>Poor: > 4s</li>
        </ul>
        
        <p><strong>FID (First Input Delay):</strong> Interactivity</p>
        <ul>
            <li>Good: < 100ms</li>
            <li>Needs Improvement: 100-300ms</li>
            <li>Poor: > 300ms</li>
        </ul>
        
        <p><strong>CLS (Cumulative Layout Shift):</strong> Visual stability</p>
        <ul>
            <li>Good: < 0.1</li>
            <li>Needs Improvement: 0.1-0.25</li>
            <li>Poor: > 0.25</li>
        </ul>
    </div>
    
    <script type="module">
        import {getCLS, getFID, getLCP} from 'https://unpkg.com/web-vitals@3/dist/web-vitals.js';
        
        function displayMetric(name, value) {
            const element = document.getElementById(name);
            const roundedValue = Math.round(value * 100) / 100;
            
            element.textContent = roundedValue;
            
            // Color coding
            if (name === 'lcp') {
                element.className = roundedValue < 2.5 ? 'metric-value good' :
                                   roundedValue < 4 ? 'metric-value needs-improvement' :
                                   'metric-value poor';
            } else if (name === 'fid') {
                element.className = roundedValue < 100 ? 'metric-value good' :
                                   roundedValue < 300 ? 'metric-value needs-improvement' :
                                   'metric-value poor';
            } else if (name === 'cls') {
                element.className = roundedValue < 0.1 ? 'metric-value good' :
                                   roundedValue < 0.25 ? 'metric-value needs-improvement' :
                                   'metric-value poor';
            }
            
            console.log(`${name.toUpperCase()}: ${roundedValue}`);
        }
        
        getCLS((metric) => displayMetric('cls', metric.value));
        getFID((metric) => displayMetric('fid', metric.value));
        getLCP((metric) => displayMetric('lcp', metric.value / 1000));
    </script>
</body>
</html>
EOF

    print_success "Created Core Web Vitals demo"
    
    echo ""
    echo "To run the demo:"
    echo "  1. cd web-vitals-demo"
    echo "  2. python3 -m http.server 8080"
    echo "  3. Open http://localhost:8080"
    echo "  4. Interact with the page to trigger FID measurement"
    echo "  5. Watch metrics update in real-time!"
    
    cd ..
    pause
}

################################################################################
# Main Menu
################################################################################

main_menu() {
    while true; do
        clear
        print_header "Frontend Optimization Workshop - Demo Menu"
        
        echo "Choose a demo to run:"
        echo ""
        echo "  1) Performance Measurement with Lighthouse"
        echo "  2) Image Optimization"
        echo "  3) Lazy Loading Demo"
        echo "  4) JavaScript Bundle Analysis"
        echo "  5) HTTP Caching Demonstration"
        echo "  6) Core Web Vitals Measurement"
        echo "  7) Run All Demos"
        echo "  0) Exit"
        echo ""
        read -p "Enter your choice: " choice
        
        case $choice in
            1) demo_performance_measurement ;;
            2) demo_image_optimization ;;
            3) demo_lazy_loading ;;
            4) demo_bundle_analysis ;;
            5) demo_caching ;;
            6) demo_web_vitals ;;
            7)
                demo_performance_measurement
                demo_image_optimization
                demo_lazy_loading
                demo_bundle_analysis
                demo_caching
                demo_web_vitals
                ;;
            0)
                print_success "Thanks for attending the workshop!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please try again."
                sleep 2
                ;;
        esac
    done
}

################################################################################
# Script Entry Point
################################################################################

print_header "Frontend Optimization Workshop"
echo "This script will demonstrate various frontend optimization techniques."
echo ""
echo "Prerequisites:"
echo "  - Node.js and npm installed"
echo "  - Internet connection (for downloading resources)"
echo ""
pause

# Run main menu
main_menu
