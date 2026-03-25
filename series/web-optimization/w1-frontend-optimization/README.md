# Workshop 1: Frontend Optimization

**Duration**: 1.5 hours  
**Level**: Intermediate  
**Series**: Web Optimization Workshop Series

## 🎯 Overview

Learn essential frontend optimization techniques to drastically improve your website's loading speed, user experience, and conversion rates. This workshop covers Core Web Vitals, JavaScript optimization, image handling, caching strategies, and modern performance tools.

## 📚 What You'll Learn

- **Performance Metrics**: Master Core Web Vitals (LCP, INP, CLS) and other key metrics
- **JavaScript Optimization**: Reduce bundle sizes through tree shaking and code splitting
- **Image Optimization**: Implement lazy loading and modern image formats
- **Caching & CDN**: Design effective caching strategies and leverage CDNs
- **Framework Optimization**: Apply framework-specific techniques (Vue, React, Nuxt, Next.js)
- **Production Monitoring**: Track real user performance metrics

## 🎓 Learning Objectives

By the end of this workshop, you will be able to:

1. ✅ Measure frontend performance using Lighthouse and Chrome DevTools
2. ✅ Optimize JavaScript bundles and implement code splitting
3. ✅ Configure lazy loading for images and components
4. ✅ Set up effective caching headers and CDN delivery
5. ✅ Reduce render-blocking resources
6. ✅ Monitor performance in production environments

## 📋 Prerequisites

### Required Knowledge
- Solid understanding of HTML, CSS, and JavaScript
- Experience with at least one modern framework (Vue, React, Angular, etc.)
- Basic understanding of HTTP and web browsers
- Familiarity with Chrome DevTools

### Required Tools
- **Web Browser**: Chrome or Firefox (latest version)
- **Node.js**: v16 or higher
- **Package Manager**: npm or yarn
- **Text Editor**: VS Code recommended
- **Terminal**: Command line access

### Recommended
- Experience with build tools (Webpack, Vite, or similar)
- Understanding of basic networking concepts

## 🚀 Quick Start

### 1. Environment Setup

Run the setup script to install necessary tools:

```bash
cd scripts
./lab-setup.sh
```

### 2. Review Workshop Materials

- [Workshop Content](materials/workshop-1-content.md) - Detailed teaching material
- [Instructor Notes](INSTRUCTOR_NOTES.md) - Teaching tips and time management

### 3. Hands-On Practice

Work through the practical exercises:

```bash
cd exercises
# Follow instructions in hands-on-lab.md
```

## 📖 Workshop Structure

### Part 1: Introduction (10 minutes)
- Why frontend optimization matters
- Common performance problems
- Real-world impact on business metrics

### Part 2: Performance Metrics (20 minutes)
- Core Web Vitals explained
- Measurement tools and techniques
- Lighthouse, WebPageTest, Chrome DevTools

### Part 3: Optimization Techniques (40 minutes)
- JavaScript bundle reduction
- Lazy loading strategies
- Image optimization
- Asset minification and compression
- CDN integration
- Caching strategies
- Render-blocking resource elimination
- Font optimization

### Part 4: Network Optimization (10 minutes)
- Reducing HTTP requests
- HTTP/2 benefits
- Prefetch and preload strategies

### Part 5: Framework Optimization (5 minutes)
- SSR and static generation
- Route-based code splitting

### Part 6: Production Monitoring (5 minutes)
- Real user monitoring tools
- Performance tracking strategies

### Part 7: Practical Workflow (5 minutes)
- Step-by-step optimization process

### Part 8: Conclusion & Q&A (5 minutes)
- Key takeaways
- Next steps

## 🔧 Workshop Materials

- **Content**: [materials/workshop-1-content.md](materials/workshop-1-content.md)
- **Exercises**: [exercises/hands-on-lab.md](exercises/hands-on-lab.md)
- **Demo Script**: [scripts/demo-script.sh](scripts/demo-script.sh)
- **Setup Script**: [scripts/lab-setup.sh](scripts/lab-setup.sh)

## 💡 Key Concepts

### Performance Impact
- **1 second delay** → up to 7% conversion loss
- **53% of users** abandon pages that take >3 seconds to load
- Page speed affects SEO rankings significantly

### Core Web Vitals
- **LCP** (Largest Contentful Paint): < 2.5s
- **INP** (Interaction to Next Paint): < 200ms
- **CLS** (Cumulative Layout Shift): < 0.1

## 🎬 Live Demo Examples

Optional demonstrations include:
- Running a Lighthouse audit
- Analyzing network waterfall
- Implementing lazy loading
- Comparing image compression formats

## 📚 Additional Resources

- [Web.dev - Performance](https://web.dev/performance/)
- [Chrome DevTools Performance Documentation](https://developer.chrome.com/docs/devtools/performance/)
- [MDN - Web Performance](https://developer.mozilla.org/en-US/docs/Web/Performance)
- [WebPageTest](https://www.webpagetest.org/)

## ❓ Troubleshooting

Common issues and solutions:

**Issue**: Lighthouse score varies between runs  
**Solution**: Run multiple tests, disable browser extensions, use incognito mode

**Issue**: Can't reproduce performance issues locally  
**Solution**: Use Chrome DevTools throttling to simulate slower connections

## 🤝 Contributing

Found an issue or want to improve the workshop? See [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## 📝 License

MIT License - see [LICENSE](../../../LICENSE) for details.

---

**Next Steps**: After completing this workshop, explore backend optimization and full-stack performance monitoring in upcoming workshops.

**Teaching for Development (TFD)** - Practical workshops for modern developers.
