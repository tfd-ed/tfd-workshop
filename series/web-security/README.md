# Container Security Workshop Series

## 📚 Complete Workshop Series

A comprehensive hands-on workshop series covering practical container security from basics to advanced topics.

**Total Duration:** 6-8 hours (7 workshops)  
**Format:** Live online workshops  
**Audience:** Developers, DevOps beginners, CS students  
**Requirement:** Laptop with Docker installed

---

## 🗂️ Workshop Structure

```
web-security/
├── README.md (this file)
├── w1-container-security-basics/
├── w2-image-security-attack-surface/
├── w3-runtime-security-privileged-containers/
├── w4-secrets-configuration-security/
├── w5-network-access-control/
├── w6-supply-chain-cicd-risks/
└── w7-final-secure-broken-app/
```

Each workshop folder contains:
- `README.md` - Workshop overview
- `materials/` - Teaching content and slides
- `scripts/` - Demo and setup scripts
- `exercises/` - Hands-on lab guides

---

## 📖 Workshops

### [Workshop 1: Container Security Basics](w1-container-security-basics/)
**Duration:** 1 hour  
**Status:** ✅ Complete

**Topics:**
- Containers vs VMs (security perspective)
- Shared kernel risks
- Security boundaries
- Common security myths

**Key Takeaway:** Containers share the host kernel - understand the fundamental security limitations.

---

### Workshop 2: Image Security & Attack Surface
**Duration:** 1 hour  
**Status:** 🚧 Coming soon

**Topics:**
- How vulnerable images happen
- `latest` tag problems
- Alpine vs Ubuntu vs distroless
- CVE scanning basics

**Key Takeaway:** Reduce attack surface before containers even run.

---

### Workshop 3: Runtime Security & Privileged Containers
**Duration:** 1-1.5 hours  
**Status:** 🚧 Coming soon

**Topics:**
- Linux capabilities explained
- Why `--privileged` is dangerous
- Container escape examples
- Running as non-root

**Key Takeaway:** Understand and mitigate runtime security risks.

---

### Workshop 4: Secrets & Configuration Security
**Duration:** 1 hour  
**Status:** 🚧 Coming soon

**Topics:**
- Why secrets in images are dangerous
- Environment variables vs mounted files
- Secret management best practices
- Basic secret rotation

**Key Takeaway:** Never commit secrets - use proper secret management.

---

### Workshop 5: Network & Access Control
**Duration:** 1 hour  
**Status:** 🚧 Coming soon

**Topics:**
- Container networking recap
- Port exposure risks
- Network isolation
- Service segmentation

**Key Takeaway:** Limit blast radius through network isolation.

---

### Workshop 6: Supply Chain & CI/CD Risks
**Duration:** 1 hour  
**Status:** 🚧 Coming soon

**Topics:**
- Image poisoning
- Dependency risks
- Tag immutability
- CI/CD security

**Key Takeaway:** Secure the entire container supply chain.

---

### Workshop 7: Secure the Broken App (Final Project)
**Duration:** 1-1.5 hours  
**Status:** 🚧 Coming soon

**Format:** Hands-on security challenge

Students receive:
- Insecure Dockerfile
- Exposed secrets
- Privileged container
- Open network access

**Tasks:**
- Harden the image
- Remove privileges
- Secure secrets
- Isolate network

**Key Takeaway:** Apply all learned concepts to secure a real application.

---

## 🎯 Learning Path

```
Workshop 1 (Basics)
    ↓
Workshop 2 (Images)
    ↓
Workshop 3 (Runtime)
    ↓
Workshop 4 (Secrets)
    ↓
Workshop 5 (Network)
    ↓
Workshop 6 (Supply Chain)
    ↓
Workshop 7 (Final Project)
```

**Recommended:** Complete workshops in order as each builds on previous concepts.

---

## 🔄 Workshop Format

Each workshop follows the same structure:

1. **Concept** (15-20 min) - Teaching with visuals
2. **Live Demo** (15 min) - Instructor demonstration
3. **Hands-On Lab** (20-30 min) - Student exercises
4. **Discussion** (10 min) - Q&A and wrap-up

---

## 🛠️ Prerequisites

### Technical Requirements
- Docker installed and running
- Basic command line knowledge
- Terminal access
- Internet connection

### Verify Your Setup
```bash
docker --version
docker ps
docker run --rm hello-world
```

### Recommended Knowledge
- Basic understanding of containers
- Linux command line basics
- Basic networking concepts

---

## 🎓 Teaching Style

- ✅ **Hands-on first** - Learn by doing
- ✅ **Break things safely** - Understand vulnerabilities
- ✅ **Real-world focus** - Practical, not theoretical
- ✅ **Attacker mindset** - Think "what if attacker..."
- ✅ **No heavy theory** - Concepts explained simply

---

## 📚 Additional Resources

### Official Documentation
- [Docker Security](https://docs.docker.com/engine/security/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)
- [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)

### Security Tools
- [Docker Bench Security](https://github.com/docker/docker-bench-security)
- [Trivy](https://github.com/aquasecurity/trivy) - Vulnerability scanner
- [Falco](https://falco.org/) - Runtime security
- [Anchore](https://anchore.com/) - Container analysis

### Further Reading
- [Container Security by Liz Rice](https://www.oreilly.com/library/view/container-security/9781492056690/)
- [Kubernetes Security](https://kubernetes-security.info/)
- [Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

---

## 🚀 Getting Started

### For Instructors

1. Review the workshop materials in each folder
2. Test demo scripts before the workshop
3. Prepare your environment
4. Set up screen sharing for online delivery

### For Students

1. Ensure Docker is installed and working
2. Start with Workshop 1
3. Complete hands-on exercises
4. Ask questions during discussion time

---

## 🎯 Learning Outcomes

After completing all workshops, students will be able to:

- ✅ Explain container security fundamentals
- ✅ Build secure container images
- ✅ Configure runtime security controls
- ✅ Manage secrets properly
- ✅ Implement network isolation
- ✅ Secure CI/CD pipelines
- ✅ Apply defense-in-depth strategies

---

## 💡 Workshop Tips

### For Online Delivery

1. **Engagement**
   - Use polls and questions
   - Encourage chat participation
   - Share your screen with large fonts

2. **Pacing**
   - Build in buffer time
   - Check for questions frequently
   - Don't rush hands-on sections

3. **Support**
   - Monitor chat for issues
   - Provide clear troubleshooting steps
   - Have backup plans for technical issues

---

## 📧 Feedback & Contributions

- Found an issue? Open an issue
- Have improvements? Submit a pull request
- Questions? Start a discussion

---

## 📄 License

Educational use - feel free to adapt and share with attribution.

---

**Ready to start?** 🚀

Begin with [Workshop 1: Container Security Basics](w1-container-security-basics/)
