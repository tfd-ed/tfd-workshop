# Assets Directory

This directory contains visual assets for the Frontend Optimization workshop.

## Structure

```
assets/
└── gifs/
    ├── lighthouse-audit.gif       # Demo of running Lighthouse
    ├── lazy-loading-demo.gif      # Lazy loading in action
    ├── network-waterfall.gif      # Network tab analysis
    ├── bundle-comparison.gif      # Bundle size before/after
    ├── web-vitals-demo.gif        # Core Web Vitals monitoring
    └── performance-improvement.gif # Before/after comparison
```

## Creating GIFs

### Tools

- **macOS**: QuickTime Player + Gifski
- **Windows**: ScreenToGif
- **Linux**: Peek or SimpleScreenRecorder + ffmpeg
- **Cross-platform**: OBS Studio + ffmpeg

### Recording Guidelines

1. **Resolution**: 1920x1080 or 1280x720
2. **Frame Rate**: 10-15 fps for smaller file size
3. **Duration**: Keep under 30 seconds
4. **File Size**: Aim for < 5MB per GIF
5. **Content**: Show clear, step-by-step actions

### Example: Creating a Lighthouse Audit GIF

```bash
# 1. Record screen with OBS or QuickTime
# 2. Convert to GIF with ffmpeg

ffmpeg -i input.mov -vf "fps=10,scale=1280:-1:flags=lanczos" \
  -c:v gif lighthouse-audit.gif

# Or use Gifski for better quality
gifski -o lighthouse-audit.gif --fps 10 --width 1280 input.mov
```

### Optimization

```bash
# Install gifsicle
brew install gifsicle

# Optimize GIF
gifsicle -O3 --lossy=80 -o optimized.gif input.gif
```

## Asset Naming Convention

- Use lowercase
- Use hyphens for spaces
- Be descriptive
- Include context if needed

Examples:
- `✅ lazy-loading-demo.gif`
- `✅ lighthouse-mobile-audit.gif`
- `❌ demo.gif`
- `❌ Test_Recording.gif`

## Usage in Documentation

```markdown
![Lighthouse Audit Demo](assets/gifs/lighthouse-audit.gif)

*Running a Lighthouse audit in Chrome DevTools*
```

## Contributing

When adding new assets:

1. Follow naming conventions
2. Optimize file size
3. Update this README
4. Include alt text in documentation
5. Test on different screen sizes

## Current Assets

| Asset | Description | Size | Created |
|-------|-------------|------|---------|
| TBD   | Placeholder | TBD  | TBD     |

*Note: GIFs to be added during workshop preparation or by contributors.*

## Future Assets Needed

- [ ] Lighthouse audit walkthrough
- [ ] Lazy loading network tab demo
- [ ] Code splitting before/after
- [ ] Image optimization comparison
- [ ] Cache headers demonstration
- [ ] Web Vitals real-time monitoring
- [ ] Performance improvement timeline

## Copyright & Licensing

All assets created for this workshop are licensed under MIT License.

External assets should be properly attributed and comply with their respective licenses.
