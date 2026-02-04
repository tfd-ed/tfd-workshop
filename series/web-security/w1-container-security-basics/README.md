# Workshop 1: Container Security Basics

[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](https://github.com/tfdevs/container-security-workshop-series)
[![Docker](https://img.shields.io/badge/Docker-Required-2496ED?logo=docker)](https://www.docker.com/)
[![Duration](https://img.shields.io/badge/Duration-2.5%20hours-blue.svg)]()

**Date:** February 4, 2026 | **Participants:** 300+ | **Completion Rate:** 85%

---

## 🎯 Workshop Overview

This workshop introduces the fundamental concepts of container security, focusing on understanding the security implications of containerization technology. Through hands-on demonstrations and exercises, participants learn why containers are not a security boundary and how to identify common security misconceptions.

### What You'll Learn

- **Containers vs VMs** - Security perspective on isolation models
- **Shared Kernel Risks** - Understanding kernel vulnerability implications
- **Container Isolation** - How Linux namespaces and cgroups work
- **Security Myths** - Common misconceptions debunked
- **Practical Skills** - Hands-on security demonstrations and exercises

---

## 📂 Workshop Structure

```
w1-container-security-basics/
├── materials/
│   └── workshop-1-content.md      # Complete workshop content with Mermaid diagrams
├── exercises/
│   └── hands-on-lab.md            # 5 hands-on exercises with solutions
├── scripts/
│   ├── lab-setup.sh               # Environment setup script
│   └── demo-script.sh             # Automated demonstration script
└── README.md                      # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Docker installed and running
- Basic Linux command line knowledge
- Terminal access
- Text editor
- Internet connection for pulling images

### Quick Start

1. **Navigate to workshop directory:**
   ```bash
   cd series/web-security/w1-container-security-basics
   ```

2. **Run the setup script:**
   ```bash
   chmod +x scripts/lab-setup.sh
   ./scripts/lab-setup.sh
   ```

3. **Review the materials:**
   ```bash
   # Read workshop content
   cat materials/workshop-1-content.md
   
   # Or open in your favorite editor/viewer
   ```

4. **Start the exercises:**
   ```bash
   cat exercises/hands-on-lab.md
   ```

### Verify Your Environment

```bash
# Check Docker version
docker --version

# Test Docker is working
docker run --rm hello-world

# Check sudo access (helpful for some exercises)
sudo -v
```

---

## 📚 Workshop Content

### Part 1: Containers vs VMs (Security View)
Learn the fundamental differences between VMs and containers from a security perspective, including:
- Virtual machine isolation model
- Container shared kernel architecture
- Security boundary comparison
- Visual diagrams using Mermaid

### Part 2: Shared Kernel Risk
Understand the security implications of kernel sharing:
- Attack flow diagrams
- Real-world CVE examples (e.g., Dirty Pipe)
- Syscall sharing implications
- Kernel compromise scenarios

### Part 3: Containers Are NOT a Security Boundary
Official positions from Docker and Kubernetes:
- What containers ARE good for
- What containers are NOT designed for
- Real-world security scenarios
- Better security approaches

### Part 4: Common Security Myths
Debunking 5 major misconceptions:
1. "Docker is secure by default"
2. "Containers can't access the host"
3. "Using Alpine Linux makes containers secure"
4. "Private registry = Secure images"
5. "Kubernetes adds security"

### Part 5: Live Demonstrations
4 hands-on demonstrations:
- Proving shared kernel
- Inspecting container processes
- Exploring namespaces
- Shared kernel access

---

## 🔬 Hands-On Exercises

### Exercise 1: Verify Shared Kernel
Prove that all containers share the same host kernel by checking kernel versions across different containers.

### Exercise 2: Inspect Process Tree
View container processes from both host and container perspectives, understanding PID namespaces.

### Exercise 3: Explore Namespaces
Investigate the 6 types of Linux namespaces that provide container isolation.

### Exercise 4: What's Shared? What's Isolated?
Identify exactly what containers can and cannot access from the host system.

### Exercise 5: Understanding Container Boundaries
Learn about container permissions and security boundaries through safe demonstrations.

---

## 🎬 Automated Demonstrations

The `demo-script.sh` provides automated demonstrations of all key concepts:

```bash
chmod +x scripts/demo-script.sh
./scripts/demo-script.sh
```

The script includes:
- Color-coded output for clarity
- Interactive pauses between demonstrations
- Automatic cleanup on exit
- 6 comprehensive demos

---

## 📊 Workshop Statistics

**February 4, 2026 Session:**

- **Registrations:** 300+ participants
- **Countries:** 15+ represented
- **Satisfaction:** 98% would recommend
- **Completion Rate:** 85% completed hands-on labs
- **Platform:** Google Meet
- **Recording:** [Watch on YouTube](#)

---

## 🎯 Learning Objectives

By the end of this workshop, you will be able to:

✅ Explain the difference between VMs and containers from a security perspective  
✅ Understand why containers share the host kernel and its implications  
✅ Identify common container security misconceptions  
✅ Inspect container isolation boundaries using Linux tools  
✅ Recognize dangerous container configurations  
✅ Apply defense-in-depth principles to container security  

---

## 💡 Key Takeaways

1. **Containers share the host kernel** - This is the fundamental security limitation
2. **Containers ≠ VMs** - Different isolation models with different security properties
3. **Docker is not secure by default** - You must actively harden containers
4. **Containers are not a security boundary** - Don't rely on them for isolation alone
5. **Defense in depth** - Layer multiple security controls

---

## 🎓 Additional Resources

### Official Documentation
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)

### Recommended Tools
- [Trivy](https://github.com/aquasecurity/trivy) - Vulnerability scanner
- [Docker Bench Security](https://github.com/docker/docker-bench-security) - Security audit
- [Falco](https://falco.org/) - Runtime security

### Further Reading
- [Container Security by Liz Rice](https://www.oreilly.com/library/view/container-security/9781492056690/)
- [Understanding Linux Namespaces](https://www.nginx.com/blog/what-are-namespaces-cgroups-how-do-they-work/)

---

## 🐛 Troubleshooting

**Issue:** "Can't see container processes from host"
```bash
# Make sure you're checking the right PID
docker inspect -f '{{.State.Pid}}' <container_name>
```

**Issue:** "Permission denied when accessing /proc"
```bash
# Some operations need sudo
sudo ls -la /proc/<pid>/ns/
```

**Issue:** "Container doesn't show up in ps"
```bash
# Container might have exited
docker ps -a  # Show all containers
```

**Issue:** "Docker command not found"
```bash
# Ensure Docker is installed and running
docker --version
```

---

## 🔄 What's Next?

### Continue Your Learning Journey

**Workshop 2: Image Security & Attack Surface** (Coming Soon)
- How vulnerable images happen
- Building minimal secure images
- CVE scanning and vulnerability detection

### Apply Your Knowledge
- Review your own container configurations
- Implement security hardening in your projects
- Share learnings with your team

---

## 🤝 Contributing

Found an issue or have suggestions? We welcome contributions!

- 🐛 [Report bugs](https://github.com/tfdevs/container-security-workshop-series/issues)
- 💡 [Request features](https://github.com/tfdevs/container-security-workshop-series/issues)
- 📝 [Improve documentation](https://github.com/tfdevs/container-security-workshop-series/pulls)

---

## 📞 Support

Need help or have questions?

- 📧 Email: contact@tfdevs.com
- 💬 Discord: [Join our community](#)
- 🐦 Twitter: [@tfdevs](https://twitter.com/tfdevs)

---

## 📜 License

This workshop is part of the TFD Workshop Series and is licensed under the MIT License.

---

<div align="center">

**[⬅ Back to Main README](../../../README.md)** | **[Next Workshop: Image Security ➡](#)**

Part of the [TFD Workshop Series](https://github.com/tfdevs/container-security-workshop-series)

</div>
