# Hands-On Lab: Frontend Optimization

**Duration**: 60-90 minutes  
**Difficulty**: Intermediate  
**Prerequisites**: Basic knowledge of HTML, CSS, JavaScript, and command line

## 🎯 Lab Objectives

By completing this hands-on lab, you will:

1. Analyze website performance using Lighthouse and Chrome DevTools
2. Optimize JavaScript bundles with code splitting
3. Implement image lazy loading and modern formats
4. Configure caching headers
5. Measure the impact of your optimizations

## 📋 Prerequisites

### Required Tools

- Modern web browser (Chrome recommended)
- Node.js v16+ and npm
- Text editor (VS Code recommended)
- Terminal access

### Verify Installation

```bash
# Check Node.js version
node --version  # Should be v16 or higher

# Check npm
npm --version
```

## 🚀 Lab Setup

### Step 1: Create Lab Project

```bash
# Create project directory
mkdir frontend-optimization-lab
cd frontend-optimization-lab

# Initialize npm project
npm init -y

# Install dependencies
npm install --save-dev vite lighthouse
```

### Step 2: Create Sample Website

Create the following files:

**index.html**:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Frontend Optimization Lab</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="header">
    <h1>Welcome to Performance Lab</h1>
    <nav>
      <a href="#home">Home</a>
      <a href="#gallery">Gallery</a>
      <a href="#about">About</a>
    </nav>
  </header>

  <main>
    <section id="home" class="hero">
      <h2>Unoptimized Website</h2>
      <p>This website has performance problems. Let's fix them!</p>
      <img src="https://picsum.photos/1200/800" alt="Hero image">
    </section>

    <section id="gallery" class="gallery">
      <h2>Image Gallery</h2>
      <div class="gallery-grid">
        <!-- 20 large images - performance problem! -->
        <img src="https://picsum.photos/800/600?random=1" alt="Gallery 1">
        <img src="https://picsum.photos/800/600?random=2" alt="Gallery 2">
        <img src="https://picsum.photos/800/600?random=3" alt="Gallery 3">
        <img src="https://picsum.photos/800/600?random=4" alt="Gallery 4">
        <img src="https://picsum.photos/800/600?random=5" alt="Gallery 5">
        <img src="https://picsum.photos/800/600?random=6" alt="Gallery 6">
        <img src="https://picsum.photos/800/600?random=7" alt="Gallery 7">
        <img src="https://picsum.photos/800/600?random=8" alt="Gallery 8">
        <img src="https://picsum.photos/800/600?random=9" alt="Gallery 9">
        <img src="https://picsum.photos/800/600?random=10" alt="Gallery 10">
        <img src="https://picsum.photos/800/600?random=11" alt="Gallery 11">
        <img src="https://picsum.photos/800/600?random=12" alt="Gallery 12">
        <img src="https://picsum.photos/800/600?random=13" alt="Gallery 13">
        <img src="https://picsum.photos/800/600?random=14" alt="Gallery 14">
        <img src="https://picsum.photos/800/600?random=15" alt="Gallery 15">
        <img src="https://picsum.photos/800/600?random=16" alt="Gallery 16">
        <img src="https://picsum.photos/800/600?random=17" alt="Gallery 17">
        <img src="https://picsum.photos/800/600?random=18" alt="Gallery 18">
        <img src="https://picsum.photos/800/600?random=19" alt="Gallery 19">
        <img src="https://picsum.photos/800/600?random=20" alt="Gallery 20">
      </div>
    </section>

    <section id="about">
      <h2>About Performance</h2>
      <p>Performance matters for user experience and business success.</p>
    </section>
  </main>

  <footer>
    <p>&copy; 2026 Performance Lab</p>
  </footer>

  <!-- Large unoptimized JavaScript -->
  <script src="app.js"></script>
</body>
</html>
```

**styles.css**:

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: Arial, sans-serif;
  line-height: 1.6;
  color: #333;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 2rem;
  text-align: center;
}

.header nav {
  margin-top: 1rem;
}

.header a {
  color: white;
  text-decoration: none;
  margin: 0 1rem;
  font-weight: bold;
}

.hero {
  padding: 4rem 2rem;
  text-align: center;
  background: #f4f4f4;
}

.hero img {
  max-width: 100%;
  height: auto;
  margin-top: 2rem;
  border-radius: 8px;
}

.gallery {
  padding: 4rem 2rem;
}

.gallery h2 {
  text-align: center;
  margin-bottom: 2rem;
}

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
}

.gallery-grid img {
  width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

#about {
  padding: 4rem 2rem;
  background: #f9f9f9;
  text-align: center;
}

footer {
  background: #333;
  color: white;
  text-align: center;
  padding: 2rem;
}
```

**app.js**:

```javascript
// Unoptimized JavaScript with blocking operations

console.log('App loaded');

// Simulate heavy computation (blocking)
function heavyComputation() {
  let result = 0;
  for (let i = 0; i < 100000000; i++) {
    result += Math.sqrt(i);
  }
  return result;
}

// Run blocking operation on load (bad!)
document.addEventListener('DOMContentLoaded', () => {
  console.log('Starting heavy computation...');
  const result = heavyComputation();
  console.log('Computation result:', result);
  
  // Add click handlers
  const links = document.querySelectorAll('a[href^="#"]');
  links.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.querySelector(e.target.getAttribute('href'));
      target.scrollIntoView({ behavior: 'smooth' });
    });
  });
  
  // Simulate analytics (should be async!)
  setTimeout(() => {
    console.log('Analytics loaded');
  }, 2000);
});

// Unused functions (dead code)
function unusedFunction1() {
  return 'This is never called';
}

function unusedFunction2() {
  return 'Neither is this';
}

function unusedFunction3() {
  return 'Or this one';
}
```

### Step 3: Add Package Scripts

Update **package.json**:

```json
{
  "name": "frontend-optimization-lab",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "audit": "lighthouse http://localhost:5173 --view"
  },
  "devDependencies": {
    "lighthouse": "^11.0.0",
    "vite": "^5.0.0"
  }
}
```

---

## 📊 Exercise 1: Baseline Performance Measurement

### Objective

Measure the current performance of the unoptimized website.

### Tasks

#### Task 1.1: Start Development Server

```bash
npm run dev
```

The site should be available at `http://localhost:5173`

#### Task 1.2: Run Lighthouse Audit

Open Chrome DevTools (F12) → Lighthouse tab → Click "Analyze page load"

**Record your baseline scores**:

```
Performance Score: _____/100
LCP: _____ seconds
INP: _____ ms
CLS: _____
Total Page Size: _____ MB
Number of Requests: _____
```

#### Task 1.3: Analyze Network Waterfall

1. Open Chrome DevTools (F12)
2. Go to Network tab
3. Refresh page (Cmd/Ctrl + R)
4. Review the waterfall

**Identify problems**:

```
Largest resources:
1. ___________________ (_____ KB)
2. ___________________ (_____ KB)
3. ___________________ (_____ KB)

Longest requests:
1. ___________________ (_____ ms)
2. ___________________ (_____ ms)
3. ___________________ (_____ ms)
```

#### Task 1.4: Performance Tab Analysis

1. Open Performance tab in DevTools
2. Click Record
3. Refresh page
4. Stop recording after page loads
5. Analyze main thread activity

**Questions**:
- How long is the main thread blocked?
- What's causing the blocking?
- When does the page become interactive?

### Expected Findings

You should discover:

🔴 **Images not lazy loaded** - All 20 gallery images load immediately  
🔴 **Blocking JavaScript** - Heavy computation blocks rendering  
🔴 **No caching headers** - Resources re-downloaded every time  
🔴 **No image optimization** - Large image files  
🔴 **Dead code** - Unused functions in JavaScript

---

## 🔧 Exercise 2: Image Optimization

### Objective

Implement lazy loading and optimize images.

### Task 2.1: Add Lazy Loading

**Modify index.html** - Add `loading="lazy"` to gallery images:

```html
<section id="gallery" class="gallery">
  <h2>Image Gallery</h2>
  <div class="gallery-grid">
    <!-- Add loading="lazy" to all gallery images -->
    <img src="https://picsum.photos/800/600?random=1" 
         alt="Gallery 1" 
         loading="lazy">
    <img src="https://picsum.photos/800/600?random=2" 
         alt="Gallery 2" 
         loading="lazy">
    <!-- Continue for all images... -->
  </div>
</section>
```

**Keep hero image eager**:

```html
<!-- Hero image should load immediately -->
<img src="https://picsum.photos/1200/800" 
     alt="Hero image" 
     loading="eager">
```

### Task 2.2: Add Image Dimensions

Add explicit dimensions to prevent layout shifts:

```html
<img src="https://picsum.photos/800/600?random=1" 
     alt="Gallery 1" 
     loading="lazy"
     width="800"
     height="600">
```

### Task 2.3: Implement Responsive Images

Update hero image with responsive sizes:

```html
<img 
  src="https://picsum.photos/800/600" 
  srcset="
    https://picsum.photos/400/300 400w,
    https://picsum.photos/800/600 800w,
    https://picsum.photos/1200/900 1200w
  "
  sizes="(max-width: 600px) 400px, (max-width: 1200px) 800px, 1200px"
  alt="Hero image"
  loading="eager"
  width="1200"
  height="800"
>
```

### Task 2.4: Test Improvements

1. Refresh the page
2. Open Network tab
3. Scroll slowly through the page

**Observations**:
- Gallery images load only when scrolled into view
- Fewer requests on initial load
- Faster initial page load

### Task 2.5: Measure Impact

Run Lighthouse again and compare:

```
Before lazy loading:
Requests: _____
Page size: _____ MB
LCP: _____ s

After lazy loading:
Requests: _____
Page size: _____ MB
LCP: _____ s

Improvement: _____ %
```

✅ **Expected improvement**: 40-60% reduction in initial requests

---

## ⚡ Exercise 3: JavaScript Optimization

### Objective

Optimize JavaScript to reduce blocking time and bundle size.

### Task 3.1: Defer JavaScript Loading

Update script tag in **index.html**:

```html
<!-- Before: Blocks rendering -->
<script src="app.js"></script>

<!-- After: Non-blocking -->
<script defer src="app.js"></script>
```

### Task 3.2: Move Heavy Computation

Modify **app.js** to make computation async:

```javascript
// Before: Blocking
function heavyComputation() {
  let result = 0;
  for (let i = 0; i < 100000000; i++) {
    result += Math.sqrt(i);
  }
  return result;
}

const result = heavyComputation(); // Blocks!

// After: Non-blocking with Web Worker or chunked
async function heavyComputationAsync() {
  let result = 0;
  const chunkSize = 1000000;
  const total = 100000000;
  
  for (let i = 0; i < total; i += chunkSize) {
    // Process in chunks, yielding to browser
    await new Promise(resolve => setTimeout(resolve, 0));
    
    for (let j = i; j < Math.min(i + chunkSize, total); j++) {
      result += Math.sqrt(j);
    }
  }
  
  return result;
}

// Run async
document.addEventListener('DOMContentLoaded', async () => {
  console.log('Starting non-blocking computation...');
  const result = await heavyComputationAsync();
  console.log('Computation result:', result);
  
  // Rest of code...
});
```

### Task 3.3: Remove Dead Code

Delete unused functions from **app.js**:

```javascript
// DELETE these unused functions:
function unusedFunction1() {
  return 'This is never called';
}

function unusedFunction2() {
  return 'Neither is this';
}

function unusedFunction3() {
  return 'Or this one';
}
```

### Task 3.4: Code Splitting (Advanced)

Create separate file **analytics.js**:

```javascript
// analytics.js - Load separately
export function initAnalytics() {
  console.log('Analytics initialized');
  // Analytics code here
}
```

Update **app.js**:

```javascript
// Dynamically import analytics when needed
document.addEventListener('DOMContentLoaded', async () => {
  // Main app code...
  
  // Load analytics after main content
  setTimeout(async () => {
    const { initAnalytics } = await import('./analytics.js');
    initAnalytics();
  }, 2000);
});
```

### Task 3.5: Test Improvements

Check Performance tab:

```
Before optimization:
- Total Blocking Time: _____ ms
- Main thread blocked for: _____ s

After optimization:
- Total Blocking Time: _____ ms
- Main thread blocked for: _____ s

Improvement: _____ %
```

✅ **Expected improvement**: 70-90% reduction in blocking time

---

## 💾 Exercise 4: Caching and Compression

### Objective

Implement effective caching strategies.

### Task 4.1: Add Vite Build Configuration

Create **vite.config.js**:

```javascript
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        // Add hash to filenames for cache busting
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash].[ext]'
      }
    },
    // Enable minification
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true // Remove console.logs in production
      }
    }
  }
});
```

### Task 4.2: Build for Production

```bash
npm run build
```

Check the `dist/` directory - files now have hashes!

### Task 4.3: Add Cache Headers (Server Configuration)

For local testing, create **public/_headers** (Netlify format):

```
# Cache static assets for 1 year
/assets/*
  Cache-Control: public, max-age=31536000, immutable

# Never cache HTML
/*.html
  Cache-Control: no-cache, no-store, must-revalidate
  
# Cache images for 30 days
/*.jpg
  Cache-Control: public, max-age=2592000
/*.png
  Cache-Control: public, max-age=2592000
```

**For NGINX**, create this configuration:

```nginx
# Cache versioned assets aggressively
location ~* \.(js|css)$ {
  add_header Cache-Control "public, max-age=31536000, immutable";
}

# Don't cache HTML
location ~* \.html$ {
  add_header Cache-Control "no-cache, no-store, must-revalidate";
}

# Cache images moderately
location ~* \.(jpg|jpeg|png|gif|ico|svg|webp)$ {
  add_header Cache-Control "public, max-age=2592000";
}

# Enable gzip compression
gzip on;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
gzip_min_length 1000;
```

### Task 4.4: Test Caching

1. Build and preview:
   ```bash
   npm run build
   npm run preview
   ```

2. Open DevTools → Network
3. Load the page
4. Check response headers for Cache-Control
5. Reload page - resources should load from cache

---

## 📈 Exercise 5: Final Performance Audit

### Objective

Measure the cumulative impact of all optimizations.

### Task 5.1: Complete Performance Test

Run Lighthouse on optimized site:

```bash
npm run audit
```

### Task 5.2: Compare Results

Fill in the comparison table:

```
┌─────────────────────────────────────────────────────────┐
│ Performance Comparison                                  │
├─────────────────────────┬────────────┬─────────────────┤
│ Metric                  │   Before   │     After       │
├─────────────────────────┼────────────┼─────────────────┤
│ Performance Score       │   ___/100  │    ___/100      │
│ LCP (seconds)           │   _____    │    _____        │
│ INP (milliseconds)      │   _____    │    _____        │
│ CLS                     │   _____    │    _____        │
│ Total Page Size (MB)    │   _____    │    _____        │
│ Number of Requests      │   _____    │    _____        │
│ Load Time (seconds)     │   _____    │    _____        │
└─────────────────────────┴────────────┴─────────────────┘

Overall Improvement: _____ %
```

### Task 5.3: Test on Slow Connection

1. Open DevTools → Network tab
2. Select "Slow 3G" from throttling dropdown
3. Reload page
4. Measure load time

```
Load time on Slow 3G:
Before: _____ seconds
After: _____ seconds
Improvement: _____ %
```

---

## 🏆 Bonus Challenges

### Challenge 1: Implement Service Worker

Create **sw.js**:

```javascript
const CACHE_NAME = 'v1';

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll([
        '/',
        '/styles.css',
        '/app.js'
      ]);
    })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});
```

Register in **app.js**:

```javascript
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js')
    .then(reg => console.log('Service Worker registered'))
    .catch(err => console.error('SW registration failed', err));
}
```

### Challenge 2: Add Critical CSS

Extract above-the-fold CSS and inline it:

```html
<head>
  <!-- Critical CSS inline -->
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .hero { padding: 4rem 2rem; text-align: center; }
  </style>
  
  <!-- Load full CSS async -->
  <link rel="preload" href="styles.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="styles.css"></noscript>
</head>
```

### Challenge 3: Implement Font Optimization

```html
<head>
  <!-- Preload critical font -->
  <link rel="preload" 
        href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" 
        as="style">
  
  <style>
    @font-face {
      font-family: 'Roboto';
      font-display: swap; /* Show fallback immediately */
      src: url('...') format('woff2');
    }
  </style>
</head>
```

---

## ✅ Lab Completion Checklist

Mark completed tasks:

- [ ] Measured baseline performance with Lighthouse
- [ ] Implemented image lazy loading
- [ ] Added responsive images
- [ ] Deferred JavaScript loading
- [ ] Optimized heavy computations
- [ ] Removed dead code
- [ ] Configured build optimization
- [ ] Set up caching headers
- [ ] Ran final performance audit
- [ ] Documented improvements

---

## 🎓 Key Learnings

After completing this lab, you should understand:

✅ How to measure performance using browser tools  
✅ The impact of lazy loading on initial page load  
✅ How JavaScript blocking affects user experience  
✅ The importance of caching for repeat visits  
✅ How to systematically optimize web performance

---

## 📚 Additional Resources

- [Web.dev Performance](https://web.dev/performance/)
- [Chrome DevTools Documentation](https://developer.chrome.com/docs/devtools/)
- [Lighthouse Scoring Guide](https://web.dev/performance-scoring/)
- [MDN Image Optimization](https://developer.mozilla.org/en-US/docs/Learn/Performance/Multimedia)

---

## 🐛 Troubleshooting

### Issue: Lighthouse shows different scores each time

**Solution**: 
- Close other tabs and applications
- Use incognito mode
- Disable browser extensions
- Run multiple tests and average the results

### Issue: Lazy loading not working

**Solution**:
- Check browser support (Chrome 77+, Firefox 75+)
- Verify `loading="lazy"` attribute is present
- Test with slow network throttling

### Issue: Images not displaying

**Solution**:
- Check console for errors
- Verify image URLs are correct
- Check CORS headers if using external images

---

**Congratulations on completing the lab! 🎉**

You've learned practical techniques to significantly improve website performance. Apply these to your real-world projects!
