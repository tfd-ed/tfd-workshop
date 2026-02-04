# Container Security Workshop Series 🔐🐳

[![Workshop Series](https://img.shields.io/badge/Workshops-7%20Part%20Series-blue.svg)](https://github.com/tfdevs/container-security-workshop-series)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Required-2496ED?logo=docker)](https://www.docker.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> A comprehensive hands-on workshop series covering practical container security from basics to advanced topics.

Transform your understanding of Docker and container security through 7 progressive workshops with real-world scenarios, live demonstrations, and hands-on exercises.

---

## 🎯 About This Series

Container security is critical in modern DevOps, yet many developers and engineers lack proper knowledge of security best practices. This workshop series bridges that gap with:

- **Practical, hands-on learning** - Real commands, real scenarios
- **Progressive curriculum** - From basics to advanced topics
- **Live demonstrations** - See security issues in action
- **Industry best practices** - Production-ready knowledge
- **Free and open-source** - All materials available to the community

---

## 📚 Workshop Overview

### Workshop 1: Container Security Basics ✅ COMPLETED
**Date:** February 4, 2026 | **Duration:** 2.5 hours | **Participants:** 300+

**What We Covered:**
- Containers vs VMs (Security Perspective)
- Shared Kernel Risks & Implications
- Container Isolation Boundaries
- Common Security Myths Debunked
- Hands-on Security Demonstrations

**Status:** Materials available in [`/w1-container-security-basics`](./w1-container-security-basics/)

**Recording:** [Watch on YouTube](#) | [Download Materials](./w1-container-security-basics/)

---

### Workshop 2: Image Security & Attack Surface
**Status:** 🚧 Coming Soon

**Topics:**
- How vulnerable images happen
- The `latest` tag problem
- Alpine vs Ubuntu vs Distroless
- CVE scanning & vulnerability detection
- Building minimal secure images

**Duration:** 1 hour | **Level:** Beginner

---

### Workshop 3: Runtime Security & Privileged Containers
**Status:** 🚧 Coming Soon

**Topics:**
- Linux capabilities explained
- Why `--privileged` is dangerous
- Container escape scenarios
- Running containers as non-root
- Capability dropping

**Duration:** 1-1.5 hours | **Level:** Intermediate

---

### Workshop 4: Secrets & Configuration Security
**Status:** 🚧 Coming Soon

**Topics:**
- Why secrets in images are dangerous
- Environment variables vs mounted secrets
- Docker secrets & Kubernetes secrets
- Secret rotation strategies
- Avoiding Git leaks

**Duration:** 1 hour | **Level:** Intermediate

---

### Workshop 5: Network & Access Control
**Status:** 🚧 Coming Soon

**Topics:**
- Container networking security
- Exposed ports & attack surface
- Network isolation patterns
- Service mesh basics
- Zero-trust networking

**Duration:** 1 hour | **Level:** Intermediate

---

### Workshop 6: Supply Chain & CI/CD Risks
**Status:** 🚧 Coming Soon

**Topics:**
- Image poisoning attacks
- Dependency vulnerabilities
- Tag immutability
- Container signing & verification
- CI/CD security best practices

**Duration:** 1 hour | **Level:** Advanced

---

### Workshop 7: Final Project - Secure the Broken App
**Status:** 🚧 Coming Soon

**Format:** Hands-on Security Challenge

**Scenario:** Fix a deliberately insecure containerized application

**Tasks:**
- Harden vulnerable Dockerfiles
- Remove excessive privileges
- Implement proper secret management
- Configure network isolation
- Apply defense-in-depth

**Duration:** 1-1.5 hours | **Level:** All levels

---

## 🗂️ Repository Structure

```
container-security-workshop-series/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── CONTRIBUTING.md                     # Contribution guidelines
├── .gitignore                          # Git ignore rules
│
├── w1-container-security-basics/       # Workshop 1 - ✅ COMPLETED
│   ├── README.md                       # Workshop overview
│   ├── materials/                      # Teaching materials
│   │   ├── workshop-1-content.md      # Detailed teaching notes
│   │   ├── slides-outline.md          # Presentation slides
│   │   └── instructor-guide.md        # Instructor reference
│   ├── scripts/                        # Demo & setup scripts
│   │   ├── demo-script.sh             # Automated demos
│   │   └── lab-setup.sh               # Environment setup
│   ├── exercises/                      # Student materials
│   │   └── hands-on-lab.md            # Lab exercises
│   ├── poster.html                     # Workshop poster
│   └── facebook-post.md               # Social media content
│
├── w2-image-security/                  # Workshop 2 - 🚧 Coming Soon
├── w3-runtime-security/                # Workshop 3 - 🚧 Coming Soon
├── w4-secrets-management/              # Workshop 4 - 🚧 Coming Soon
├── w5-network-security/                # Workshop 5 - 🚧 Coming Soon
├── w6-supply-chain/                    # Workshop 6 - 🚧 Coming Soon
├── w7-final-project/                   # Workshop 7 - 🚧 Coming Soon
│
├── resources/                          # Shared resources
│   ├── tools/                         # Security tools & scripts
│   ├── references/                    # Documentation & links
│   └── templates/                     # Templates for exercises
│
└── docs/                              # Additional documentation
    ├── setup-guide.md                 # Environment setup
    ├── troubleshooting.md             # Common issues
    └── faq.md                         # Frequently asked questions
```

---

## 🚀 Getting Started

### Prerequisites

- **Docker** installed and running
- **Basic Linux command line** knowledge
- **Terminal** access
- **Text editor** (VS Code, Vim, etc.)
- **Internet connection** for pulling images

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tfdevs/container-security-workshop-series.git
   cd container-security-workshop-series
   ```

2. **Choose a workshop:**
   ```bash
   cd w1-container-security-basics
   ```

3. **Review the README:**
   ```bash
   cat README.md
   ```

4. **Run the setup script:**
   ```bash
   chmod +x scripts/lab-setup.sh
   ./scripts/lab-setup.sh
   ```

5. **Follow the exercises:**
   ```bash
   cat exercises/hands-on-lab.md
   ```

### Verify Your Environment

```bash
# Check Docker version
docker --version

# Test Docker is working
docker run --rm hello-world

# Check sudo access (optional but helpful)
sudo -v
```

---

## 🎓 Who Should Use This?

This workshop series is perfect for:

- ✅ **Developers** using Docker in projects
- ✅ **DevOps Engineers** managing containerized workloads
- ✅ **Security Professionals** learning container security
- ✅ **CS Students** studying cloud technologies
- ✅ **System Administrators** migrating to containers
- ✅ **Tech Leads** implementing DevSecOps

---

## 📖 Learning Path

### Beginner Track
1. Workshop 1: Container Security Basics
2. Workshop 2: Image Security
3. Workshop 4: Secrets Management

### Intermediate Track
1. Workshop 3: Runtime Security
2. Workshop 5: Network Security
3. Workshop 6: Supply Chain

### Advanced Track
1. Complete Workshops 1-6
2. Workshop 7: Final Project
3. Apply in real-world scenarios

---

## 🛠️ Tools & Technologies Covered

- **Docker** - Container runtime
- **Linux** - Namespaces, cgroups, capabilities
- **Security Tools** - Trivy, Docker Bench, Falco
- **Best Practices** - CIS Benchmarks, NIST guidelines
- **Kubernetes** - Security concepts (where applicable)

---

## 📊 Workshop Statistics

### Workshop 1 (February 4, 2026)
- **Registrations:** 300+ participants
- **Countries:** 15+ countries represented
- **Satisfaction:** 98% would recommend
- **Completion Rate:** 85% completed hands-on labs
- **Platform:** Google Meet

---

## 🤝 Contributing

We welcome contributions! Whether it's:

- 🐛 **Bug reports** - Found an issue? Let us know
- 💡 **Feature requests** - Have an idea? Share it
- 📝 **Documentation** - Improve our docs
- 🔧 **Code** - Submit a PR with improvements
- 🎓 **Teaching** - Share your expertise

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use these materials for personal learning
- ✅ Use them in your own workshops (with attribution)
- ✅ Modify and adapt the content
- ✅ Share with your team or community

---

## 🌟 Support & Community

### Get Help
- 📧 **Email:** contact@tfdevs.com
- 💬 **Discord:** [Join our community](#)
- 🐦 **Twitter:** [@tfdevs](https://twitter.com/tfdevs)
- 💼 **LinkedIn:** [TFDevs](https://linkedin.com/company/tfdevs)

### Stay Updated
- 🔔 **Watch** this repo for updates
- ⭐ **Star** if you find it helpful
- 🔄 **Fork** to create your own version
- 📢 **Share** with your network

---

## 📚 Additional Resources

### Official Documentation
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)
- [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)

### Recommended Reading
- [Container Security by Liz Rice](https://www.oreilly.com/library/view/container-security/9781492056690/)
- [Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [Kubernetes Security Best Practices](https://kubernetes-security.info/)

### Security Tools
- [Trivy](https://github.com/aquasecurity/trivy) - Vulnerability scanner
- [Docker Bench Security](https://github.com/docker/docker-bench-security) - Security audit
- [Falco](https://falco.org/) - Runtime security
- [Anchore](https://anchore.com/) - Container analysis

---

## 🙏 Acknowledgments

Special thanks to:
- All **300+ participants** of Workshop 1
- **Contributors** who helped improve the materials
- **Open source community** for tools and resources
- **Docker & Kubernetes** communities for documentation

---

## 📅 Upcoming Workshops

Stay tuned for announcements:
- **Workshop 2:** Image Security (TBA)
- **Workshop 3:** Runtime Security (TBA)
- **Workshop 4:** Secrets Management (TBA)

Follow us on social media for updates! 📢

---

## 🎯 Our Mission

**"Making container security accessible, practical, and understandable for everyone."**

We believe that security should not be an afterthought. Through hands-on education and practical examples, we empower developers and engineers to build secure containerized applications from day one.

---

## 📞 Contact

**TFDevs - Teaching for Development**

- 🌐 Website: [tfdevs.com](https://tfdevs.com)
- 📧 Email: contact@tfdevs.com
- 🎥 YouTube: [@tfdevs](https://youtube.com/@tfdevs)
- 📘 Facebook: [TFDevs](https://facebook.com/teachingfordevelopment)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

**🔔 Watch for updates on new workshops**

**🤝 Contribute to help others learn**

Made with ❤️ by [TFDevs](https://tfdevs.com)

[⬆ Back to Top](#container-security-workshop-series-)

</div>
