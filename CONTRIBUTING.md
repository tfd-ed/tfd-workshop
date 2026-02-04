# Contributing to Container Security Workshop Series 🤝

Thank you for your interest in contributing! This project aims to make container security accessible to everyone, and your contributions help achieve that mission.

## 🌟 Ways to Contribute

### 1. Report Issues 🐛
Found a bug, typo, or technical error?
- Open an issue with a clear description
- Include steps to reproduce (if applicable)
- Mention which workshop is affected
- Provide your environment details (OS, Docker version)

### 2. Suggest Improvements 💡
Have ideas for better explanations or examples?
- Open an issue describing your suggestion
- Explain why it would be helpful
- Provide examples if possible

### 3. Fix Bugs or Typos 🔧
- Fork the repository
- Create a branch: `git checkout -b fix/your-fix-name`
- Make your changes
- Test thoroughly
- Submit a pull request

### 4. Improve Documentation 📝
Documentation is crucial!
- Fix typos or unclear explanations
- Add missing steps or clarifications
- Improve code comments
- Add troubleshooting tips

### 5. Add New Content 🎓
Want to contribute a new workshop or exercise?
- Open an issue first to discuss your idea
- Follow the existing workshop structure
- Ensure hands-on exercises are included
- Test all commands and scripts

## 📋 Contribution Guidelines

### Code Style
- **Shell scripts:** Use `#!/bin/bash` and follow shellcheck guidelines
- **Markdown:** Use proper headings hierarchy (h1 → h2 → h3)
- **Code blocks:** Always specify language for syntax highlighting
- **Line length:** Keep lines under 100 characters when possible

### Workshop Structure
Each workshop should include:
```
wX-workshop-name/
├── README.md                   # Workshop overview
├── materials/
│   ├── workshop-X-content.md  # Detailed teaching content
│   ├── slides-outline.md      # Presentation outline
│   └── instructor-guide.md    # Quick reference
├── scripts/
│   ├── demo-script.sh         # Automated demos
│   └── lab-setup.sh           # Environment setup
└── exercises/
    └── hands-on-lab.md        # Student exercises
```

### Commit Messages
Use clear, descriptive commit messages:
- ✅ `Fix typo in Workshop 1 exercises`
- ✅ `Add troubleshooting section for Alpine images`
- ✅ `Update demo script with better error handling`
- ❌ `fix stuff`
- ❌ `update`

### Pull Request Process

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/container-security-workshop-series.git
   cd container-security-workshop-series
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the style guidelines
   - Test all code and commands
   - Update documentation if needed

4. **Test Your Changes**
   - Run all scripts to ensure they work
   - Verify commands in exercises
   - Check markdown formatting

5. **Commit & Push**
   ```bash
   git add .
   git commit -m "Add: clear description of changes"
   git push origin feature/your-feature-name
   ```

6. **Open Pull Request**
   - Provide a clear title and description
   - Reference any related issues
   - Explain what you changed and why
   - Wait for review and feedback

## ✅ Checklist Before Submitting

- [ ] All commands have been tested
- [ ] Scripts are executable (`chmod +x`)
- [ ] Markdown is properly formatted
- [ ] Code blocks have language specified
- [ ] No sensitive information included
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Changes follow existing structure

## 🎯 Priority Areas

We especially need help with:
- 🚧 **Workshop 2-7 content** - Help develop future workshops
- 🔧 **Troubleshooting guides** - Common issues and solutions
- 🌍 **Translations** - Make content accessible globally
- 🎨 **Visual aids** - Diagrams, infographics, illustrations
- 📹 **Video tutorials** - Supplement written materials

## 🚫 What NOT to Contribute

Please avoid:
- ❌ Copyrighted content without permission
- ❌ Malicious or dangerous code
- ❌ Promotional content or spam
- ❌ Offensive or inappropriate material
- ❌ Secrets, keys, or credentials (even for testing)

## 📞 Questions?

If you're unsure about anything:
- 💬 Open a discussion on GitHub
- 📧 Email us at contact@tfdevs.com
- 🐦 Tweet at [@tfdevs](https://twitter.com/tfdevs)

## 🏆 Recognition

All contributors will be:
- Listed in the README acknowledgments
- Mentioned in workshop materials (if significant contribution)
- Given credit in relevant sections

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for making container security education better for everyone!** 🙏

Made with ❤️ by the TFDevs community
