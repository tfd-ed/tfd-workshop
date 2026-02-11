# Workshop 1: Container Security Basics - Instructor Notes

## 📋 Overview

**Duration:** 2.5 hours  
**Level:** Beginner  
**Format:** Live online workshop with hands-on exercises  
**Target Audience:** Developers, DevOps engineers, CS students with basic Docker knowledge

---

## 🎯 Workshop Objectives

By the end of this workshop, participants will:
- Understand containers vs VMs from a security perspective
- Recognize shared kernel security implications
- Identify common container security misconceptions
- Inspect container isolation boundaries using Linux tools

---

## ⏱️ Suggested Timeline

| Section | Duration | Activity |
|---------|----------|----------|
| **Introduction** | 5 min | Welcome, objectives, agenda |
| **Part 1: Containers vs VMs** | 15 min | Concept + Discussion |
| **Part 2: Shared Kernel Risk** | 15 min | Teaching + Examples |
| **Part 3: Security Boundaries** | 15 min | Concept + Real-world scenarios |
| **Part 4: Common Myths** | 20 min | Interactive debunking |
| **Break** | 10 min | Screen break |
| **Live Demonstrations** | 25 min | 4 demos with explanation |
| **Hands-On Lab Setup** | 5 min | Students run setup script |
| **Hands-On Exercises** | 30 min | Students work independently |
| **Discussion & Q&A** | 15 min | Key takeaways, questions |
| **Wrap-up** | 5 min | Next workshop preview, resources |

**Total:** ~2.5 hours (adjust based on pace and questions)

---

## 🚀 Pre-Workshop Preparation

### Technical Setup (30 min before)

1. **Test your environment:**
   ```bash
   docker --version
   docker run --rm hello-world
   docker run --rm ubuntu uname -r
   ```

2. **Run the demo script in dry-run:**
   ```bash
   cd scripts/
   ./demo-script.sh
   ```

3. **Have backup containers ready:**
   ```bash
   docker pull ubuntu:latest
   docker pull alpine:latest
   docker pull nginx:latest
   ```

4. **Test screen sharing** with large fonts and contrast

### Materials to Have Ready

- Workshop content (materials/workshop-1-content.md)
- Demo script (scripts/demo-script.sh)
- Exercise guide (exercises/hands-on-lab.md)
- Browser tabs:
  - Docker Security Docs
  - CVE-2022-0847 (Dirty Pipe) article
  - CIS Docker Benchmark

---

## 📚 Teaching Tips by Section

### Part 1: Containers vs VMs

**Key Message:** Different isolation models = Different security properties

**Teaching Approach:**
- Start with the kernel/hypervisor definitions - many students won't know these
- Use the diagrams - walk through each layer
- Emphasize: "VMs virtualize hardware, containers virtualize OS"

**Common Questions:**
- Q: "Why not just use VMs then?"
  - A: Performance, resource efficiency, DevOps workflows. We're not saying don't use VMs - we're saying understand the trade-offs.

- Q: "What about Windows containers?"
  - A: Same concept, but Windows kernel. Less common in production. This workshop focuses on Linux containers.

**Talking Points:**
- Hypervisor provides strong isolation boundary
- Each VM has complete kernel = more overhead but better isolation
- Containers share kernel = lighter but weaker isolation

**Windows/Mac Clarification:**
- Bring up Docker Desktop VM early
- Explain it's still Linux containers sharing a Linux kernel
- Emphasize production scenarios run directly on Linux servers

---

### Part 2: Shared Kernel Risk

**Key Message:** One kernel vulnerability = game over for all containers

**Teaching Approach:**
- Use the attack flow diagram - walk through each step
- Reference real CVE (Dirty Pipe) for credibility
- Make it concrete: "If I'm in Container 1 and exploit kernel..."

**Demo Highlight:**
- Show `uname -r` from multiple containers
- Let students see it's literally the same kernel version
- Drive home: "This isn't theoretical - this is how it works"

**Common Questions:**
- Q: "How often do kernel vulnerabilities happen?"
  - A: Regularly. That's why kernel updates are critical. Show CVE databases.

- Q: "Can we isolate the kernel somehow?"
  - A: Not with standard containers. That's when you need VMs, gVisor, or Kata Containers (mention briefly for advanced students).

**Talking Points:**
- Kernel is the single point of failure
- All containers make syscalls through same kernel
- Kernel exploit = host access + all containers
- This is by design, not a bug

---

### Part 3: Security Boundaries

**Key Message:** Containers were NOT designed for security isolation

**Teaching Approach:**
- Quote Docker/Kubernetes official docs
- Use the good/bad comparison diagram
- Give real-world CI/CD example

**Important Emphasis:**
- Containers are great for what they do
- They're not broken - they're just not security tools
- Need defense-in-depth approach

**Common Questions:**
- Q: "So should we not use containers?"
  - A: No! Use them for what they're good at. Just understand the security model and add appropriate controls.

- Q: "What do you mean by 'untrusted workloads'?"
  - A: Code you don't control: customer uploads, public CI/CD, multi-tenant platforms

**Better Alternatives Discussion:**
- VMs for untrusted workloads
- gVisor (mention briefly)
- Kata Containers (mention briefly)
- Firecracker (mention briefly)
- Don't dive deep - just awareness

---

### Part 4: Common Myths

**Key Message:** Challenge assumptions, think critically

**Teaching Approach:**
- Interactive: Ask students to raise hands if they thought each myth was true
- Go through each myth methodically
- Use humor - these are common misconceptions

**Myth-by-Myth Tips:**

**Myth #1: "Docker is secure by default"**
- Show: `docker run ubuntu whoami` → root
- Explain: Convenience over security by design
- Not criticism - just reality

**Myth #2: "Containers can't access the host"**
- Demo the volume mount: `-v /:/host`
- Let them see the entire host filesystem
- Emphasize: "This is configuration, not a bug"

**Myth #3: "Alpine = Secure"**
- Acknowledge: Smaller surface is better
- But: Still shares kernel, still runs as root by default
- Size ≠ Security

**Myth #4: "Private registry = Secure"**
- Private means access control, not security
- Still need scanning, verification
- Trust but verify

**Myth #5: "Kubernetes adds security"**
- K8s is orchestration, not security
- Provides tools (network policies, RBAC)
- But misconfiguration can make things worse
- Security is your responsibility

---

## 🎬 Live Demonstrations

### Demo Execution Tips

**General Guidelines:**
- Explain what you're about to do BEFORE running commands
- Run commands slowly, let output be visible
- Pause after key observations
- Use large terminal fonts (16pt+)
- High contrast color scheme

**Demo 1: Shared Kernel**
- **Time:** 3 minutes
- **Key Point:** All containers show same kernel version
- **Tip:** Run from host first, then containers - contrast is powerful
- **Watch For:** On Mac/Windows, explain Docker Desktop VM

**Demo 2: Container Processes**
- **Time:** 5 minutes
- **Key Point:** Containers are just processes with namespaces
- **Tip:** Show both perspectives - host and container
- **Highlight:** PID 1 inside container, real PID on host

**Demo 3: Namespaces**
- **Time:** 5 minutes
- **Key Point:** 6 types of namespaces provide isolation
- **Tip:** Use sudo if needed, mention this upfront
- **Explain:** What each namespace does (use the table)

**Demo 4: Shared Kernel Access**
- **Time:** 2 minutes
- **Key Point:** Containers read host kernel info via /proc
- **Tip:** Quick demo, reinforces earlier concepts

### If Demos Fail

**Backup Plans:**
- Have pre-recorded screen recordings ready
- Use screenshots in slides
- Explain what should happen and why
- Move to exercises - students can see it themselves

---

## 🔬 Hands-On Lab Management

### Setup Phase (5 minutes)

**Instructions:**
1. Share the setup script command in chat
2. Ask students to run it now
3. Monitor chat for errors
4. Have common troubleshooting ready

**Common Setup Issues:**

**"Docker not found"**
```bash
# Solution: Install Docker first
# Point to: https://docs.docker.com/get-docker/
```

**"Permission denied"**
```bash
# Solution: Add user to docker group or use sudo
sudo usermod -aG docker $USER
# Then logout/login
```

**"Port already in use"**
```bash
# Solution: Stop conflicting containers
docker ps
docker stop <conflicting_container>
```

### Exercise Phase (30 minutes)

**Facilitation Tips:**
- Encourage students to work at their own pace
- Monitor chat actively for questions
- Don't rush through - let them explore
- Some will finish early, some will need more time

**Exercise Checkpoints:**

**5 minutes in:** Check if everyone completed Exercise 1
**15 minutes in:** Most should be on Exercise 3
**25 minutes in:** Give 5-minute warning
**30 minutes:** Bring everyone back for discussion

**Circulate and Help:**
- Watch for common errors
- Share solutions in chat
- Encourage peer help
- Note questions for Q&A

---

## 💭 Discussion & Q&A Tips

### Discussion Questions

**When to use Containers vs VMs?**
- Guide toward: VMs for untrusted, containers for trusted
- Discuss hybrid approaches
- Real-world examples from your experience

**Shared Kernel Exploitation**
- Review the attack flow
- Discuss defense strategies
- Mention: patching, monitoring, segmentation

**Security Measures**
- Collect ideas from students first
- Validate good suggestions
- Correct misconceptions
- Preview Workshop 2-7 topics

### Handling Questions

**If You Don't Know:**
- "Great question! I'm not 100% certain. Let me research and follow up"
- Never make up answers
- Use it as learning opportunity

**If It's Off-Topic:**
- "That's a great question for Workshop X"
- "Let's discuss that in the break/after"
- Note it for future reference

**If It's Too Advanced:**
- Acknowledge the depth
- Give brief high-level answer
- Offer to discuss offline
- Share resources

---

## 🎯 Key Messages to Reinforce

Throughout the workshop, repeatedly emphasize:

1. **Containers share the host kernel** - This is fundamental
2. **Shared kernel = shared security fate** - Vulnerability affects all
3. **Not a bug, it's the design** - Containers are what they are
4. **Defense in depth is essential** - No single security control
5. **Production vs Development** - What matters in real deployments

---

## 🚨 Common Pitfalls to Avoid

### For Instructors

1. **Don't bash Docker** - It's a tool, not inherently bad
2. **Don't oversimplify** - Security is nuanced
3. **Don't assume prior knowledge** - Define terms
4. **Don't rush through demos** - Slow is smooth, smooth is fast
5. **Don't ignore questions** - Engage with students

### For Students (Watch For)

1. **Thinking Docker Desktop = Production** - Clarify early and often
2. **Confusing containers with VMs** - Use diagrams repeatedly
3. **Believing myths are true** - Challenge assumptions gently
4. **Not running labs** - Encourage hands-on participation
5. **Getting lost in details** - Keep big picture in mind

---

## 📊 Assessment & Feedback

### Quick Knowledge Checks

**Throughout Workshop:**
- "Why do all containers show the same kernel version?"
- "What happens if the kernel is compromised?"
- "Can you name the 6 namespace types?"

**End of Workshop:**
- "In one sentence, what's the main security limitation of containers?"
- Expected: "They share the host kernel"

### Feedback Collection

**During Workshop:**
- Read chat sentiment
- Watch for confusion signals
- Adjust pace as needed

**After Workshop:**
- Send feedback survey
- Ask: What worked? What didn't?
- Collect suggestions for improvement

---

## 📚 Additional Resources to Share

### Essential Reading
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [NIST Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)

### Tools to Mention
- Trivy (vulnerability scanning)
- Docker Bench Security (security audit)
- Falco (runtime security)

### Deep Dives
- Container Security by Liz Rice (book)
- Linux namespaces man pages
- Kernel CVE databases

---

## 🔄 Workshop Variations

### For Different Audiences

**Beginners (More Docker Basics):**
- Spend more time on Docker fundamentals
- Simplify security concepts
- More guided exercises

**Intermediate (Faster Pace):**
- Skip Docker basics
- Dive deeper into namespaces
- More challenging exercises

**Advanced (Additional Topics):**
- Discuss gVisor, Kata Containers
- Kernel security modules
- Real CVE exploitation (safely)

### For Different Formats

**2-hour Version:**
- Cut one demonstration
- Reduce exercise time to 20 minutes
- Shorter Q&A

**3-hour Version:**
- Add more exercises
- Deeper demo explanations
- Extended Q&A and discussion

**Self-Paced:**
- Pre-record demonstrations
- Provide written walkthrough
- Add more explanatory text

---

## 🎓 Success Metrics

### Workshop is Successful If:

- ✅ 80%+ students complete exercises
- ✅ Students can explain shared kernel concept
- ✅ Active participation in discussions
- ✅ Positive feedback (4+/5 rating)
- ✅ Students express interest in Workshop 2

### Red Flags:

- ❌ Many students lost/confused
- ❌ Questions about basic Docker concepts
- ❌ Technical issues consuming time
- ❌ Dead chat, no engagement
- ❌ Running significantly over time

---

## 🔧 Troubleshooting Guide

### Technical Issues

**Screen sharing frozen:**
- Stop and restart share
- Use backup browser/tool
- Continue with verbal explanation

**Demo fails:**
- Use pre-recorded backup
- Explain what should happen
- Move forward, don't debug live

**Student environment issues:**
- Address common ones in chat
- Direct to exercise troubleshooting section
- Offer to help in break

### Engagement Issues

**Low participation:**
- Ask direct questions
- Use polls/reactions
- Make it interactive

**Too many questions:**
- Acknowledge and defer some
- "Great question for later"
- Keep on schedule

**Pace problems:**
- Check in regularly: "Too fast? Too slow?"
- Be flexible
- Skip optional content if needed

---

## 📝 Post-Workshop Follow-Up

### Immediately After

1. Save chat log (questions for FAQ)
2. Note what worked/didn't work
3. Update materials based on feedback
4. Thank participants

### Within 24 Hours

1. Send recording link (if recorded)
2. Share additional resources
3. Send Workshop 2 announcement
4. Respond to follow-up questions

### Within a Week

1. Analyze feedback survey
2. Update materials
3. Prepare for next workshop
4. Document lessons learned

---

## 💡 Pro Tips

1. **Start on time** - Respect everyone's schedule
2. **Use humor** - Security can be dry, lighten it up
3. **Share real stories** - Personal anecdotes are powerful
4. **Encourage questions** - No stupid questions
5. **Stay positive** - Security should empower, not scare
6. **Be humble** - Admit when you don't know
7. **Follow up** - Build the community
8. **Iterate** - Every workshop teaches you something

---

## 📞 Support Resources

### During Workshop
- Co-instructor (if available) monitors chat
- Have Docker documentation open
- Keep troubleshooting guide handy

### For Students
- Discord/Slack channel for questions
- GitHub issues for material problems
- Email for private inquiries

---

## 🎯 Preparing for Workshop 2

**Preview Topics:**
- Image layers and vulnerabilities
- CVE scanning
- Minimal base images
- Secure image building

**Transition Statement:**
"Now that you understand why containers share the kernel and the security implications, in Workshop 2 we'll learn how to build secure images before they even run."

---

## 📋 Pre-Workshop Checklist

**1 Week Before:**
- [ ] Test all demos
- [ ] Update materials if needed
- [ ] Send reminder email
- [ ] Prepare backup materials

**1 Day Before:**
- [ ] Test complete setup
- [ ] Charge devices
- [ ] Prepare workspace
- [ ] Print/bookmark materials

**1 Hour Before:**
- [ ] Join meeting early
- [ ] Test audio/video
- [ ] Share welcome slide
- [ ] Calm and focus

**During Workshop:**
- [ ] Record (if applicable)
- [ ] Monitor chat
- [ ] Take notes
- [ ] Engage participants

**After Workshop:**
- [ ] Save materials
- [ ] Send follow-up
- [ ] Update notes
- [ ] Plan improvements

---

## 🌟 Remember

You're teaching critical skills that will help developers build more secure systems. Your passion and clarity will inspire them. Stay patient, be supportive, and enjoy the process!

**Good luck with your workshop! 🚀**

---

*Last Updated: February 4, 2026*  
*Workshop Version: 1.0*  
*For questions or suggestions, contact: info@tfdevs.com*
