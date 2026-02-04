# Workshop 1: Container Security Basics - Hands-On Lab Guide

## 🎯 Lab Objectives

- Verify that containers share the host kernel
- Inspect container processes from both inside and outside
- Explore Linux namespaces
- Understand what's shared vs isolated
- See how easy it is to misconfigure container security

**Time:** 25-30 minutes

---

## 🚀 Setup

Run the setup script:
```bash
chmod +x lab-setup.sh
./lab-setup.sh
```

This will:
- Pull necessary images (ubuntu, alpine, nginx)
- Start demo containers (web1, web2, alpine-demo)

---

## 📝 Exercise 1: Verify Shared Kernel (5 minutes)

### Goal
Prove that all containers share the same host kernel.

### Steps

**Step 1:** Check your host kernel version
```bash
uname -r
```

**Expected output:** Something like `5.15.0-56-generic` or `6.1.0-25-amd64`

**Step 2:** Check kernel version from different containers
```bash
# Ubuntu container
docker run --rm ubuntu uname -r

# Alpine container
docker run --rm alpine uname -r

# Existing nginx container
docker exec web1 uname -r
```

### ✅ Success Criteria
All commands should show the **exact same kernel version**.

### 💭 Questions to Consider
1. Why do they all show the same kernel?
2. What are the security implications of this?
3. If there's a kernel vulnerability, what's affected?

<details>
<summary>Click for answers</summary>

1. **Why same kernel?** Containers are processes running on the host OS, using the host's kernel. Unlike VMs, they don't have their own kernel.

2. **Security implications:** A kernel vulnerability in one container affects ALL containers on the host, plus the host itself.

3. **What's affected?** The host and every single container running on it.

</details>

---

## 📝 Exercise 2: Inspect Process Tree (8 minutes)

### Goal
Understand how container processes appear from both host and container perspectives.

### Part A: View from Host

**Step 1:** Get the container's host PID
```bash
docker inspect -f '{{.State.Pid}}' web1
```

**Expected output:** A number like `15234`

**Step 2:** Save it to a variable and view the process
```bash
HOST_PID=$(docker inspect -f '{{.State.Pid}}' web1)
echo "Container PID on host: $HOST_PID"

# View the process details
ps -p $HOST_PID -o pid,ppid,user,command
```

**Step 3:** View all nginx processes from host
```bash
ps aux | grep nginx | grep -v grep
```

### Part B: View from Inside Container

**Step 4:** View processes from inside the container
```bash
docker exec web1 ps aux
```

### ✅ Success Criteria
- You see the nginx process from both host and container
- PID is different in each view (e.g., PID 15234 on host, PID 1 in container)
- You understand this is the same process, just different perspectives

### 💭 Questions to Consider
1. Why does the container think it's PID 1?
2. What Linux feature enables this?
3. Can the host kill the container process?

<details>
<summary>Click for answers</summary>

1. **Why PID 1?** PID namespace makes the container's main process appear as PID 1 inside its namespace.

2. **Linux feature:** Linux namespaces (specifically PID namespace).

3. **Can host kill it?** Yes! The host has full control:
   ```bash
   kill $HOST_PID
   # Or
   docker stop web1
   ```

</details>

---

## 📝 Exercise 3: Explore Namespaces (7 minutes)

### Goal
Understand the Linux namespaces that provide container isolation.

### Part A: View Container Namespaces

**Step 1:** Get container PID
```bash
CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' alpine-demo)
echo "Container PID: $CONTAINER_PID"
```

**Step 2:** List the container's namespaces (may need sudo)
```bash
sudo ls -la /proc/$CONTAINER_PID/ns/
```

**Expected output:**
```
lrwxrwxrwx 1 root root 0 mnt -> 'mnt:[4026532574]'
lrwxrwxrwx 1 root root 0 net -> 'net:[4026532577]'
lrwxrwxrwx 1 root root 0 pid -> 'pid:[4026532575]'
...
```

### Part B: Compare with Host Namespaces

**Step 3:** View host namespaces (PID 1)
```bash
sudo ls -la /proc/1/ns/
```

### Part C: Understand Each Namespace

**Step 4:** Enter the container and explore isolation
```bash
docker exec -it alpine-demo sh
```

Inside the container, run:
```bash
# Network namespace (isolated)
hostname
ip addr

# Mount namespace (isolated)
mount | head -5

# Process namespace (isolated)
ps aux

# UTS namespace (isolated hostname)
cat /etc/hostname

# But kernel info is shared!
cat /proc/version
cat /proc/cpuinfo | head -10
```

Type `exit` to leave the container.

### ✅ Success Criteria
- You see different namespace IDs for container vs host
- You understand which aspects are isolated vs shared

### 📊 Namespace Summary

| Namespace | What It Isolates | Shared or Isolated? |
|-----------|------------------|---------------------|
| PID | Process tree | Isolated |
| NET | Network interfaces, IPs | Isolated |
| MNT | Filesystem mounts | Isolated |
| UTS | Hostname | Isolated |
| IPC | Inter-process communication | Isolated |
| USER | User/Group IDs | Usually Isolated |
| Kernel | The actual kernel | **SHARED** ⚠️ |

---

## 📝 Exercise 4: What's Shared? What's Isolated? (5 minutes)

### Goal
Identify exactly what containers can and cannot access.

### Commands to Run

**Step 1:** Enter the alpine container
```bash
docker exec -it alpine-demo sh
```

**Step 2:** Inside the container, check these:
```bash
# Check kernel (SHARED)
uname -r
cat /proc/version

# Check CPU info (SHARED)
cat /proc/cpuinfo | head -15

# Check memory (SHARED but limited)
cat /proc/meminfo | head -10

# Check hostname (ISOLATED)
hostname

# Check network (ISOLATED)
ip addr

# Check processes (ISOLATED)
ps aux

# Try to see host processes
ps aux | wc -l  # Should be very few

# Check filesystem (ISOLATED)
ls /
df -h
```

**Step 3:** Exit and compare with host
```bash
exit

# Host perspective
hostname
ip addr
ps aux | wc -l  # Many more processes!
```

### ✅ Fill in the Table

| Item | Container Value | Host Value | Shared or Isolated? |
|------|----------------|------------|---------------------|
| Kernel version | | | |
| Hostname | | | |
| Number of processes | | | |
| IP address | | | |
| CPU cores | | | |

<details>
<summary>Click for answers</summary>

| Item | Container Value | Host Value | Shared or Isolated? |
|------|----------------|------------|---------------------|
| Kernel version | Same | Same | **SHARED** |
| Hostname | Different | Different | Isolated |
| Number of processes | ~5-10 | 100+ | Isolated (PID namespace) |
| IP address | 172.17.x.x | Real IP | Isolated (NET namespace) |
| CPU cores | Same | Same | **SHARED** |

</details>

---

## 📝 Exercise 5: Break the Illusion (CAREFULLY!) (5 minutes)

### Goal
⚠️ **WARNING:** This exercise shows dangerous configurations. **NEVER do this in production!**

Understand how easy it is to break container isolation with misconfigurations.

### Part A: Mount Host Filesystem

**Step 1:** Run container with host mount
```bash
docker run -it --rm -v /:/host ubuntu bash
```

**Step 2:** Inside this container:
```bash
# You can see the ENTIRE host filesystem!
ls /host

# Look at host's root directory
ls /host/root

# View host's password file
cat /host/etc/passwd

# See what's in /home
ls /host/home

# Exit
exit
```

### Part B: Try Privileged Mode (VERY DANGEROUS)

**Step 3:** Run with --privileged flag
```bash
docker run -it --rm --privileged -v /:/host ubuntu bash
```

**Step 4:** Inside this container:
```bash
# Now you have almost full host access
ls /host/root

# You could even access host devices
ls /host/dev

# You could modify host files
# (DON'T actually do this!)
# echo "pwned" > /host/etc/motd

exit
```

### ✅ Success Criteria
- You understand how volume mounts can expose the host
- You see why `--privileged` is dangerous
- You recognize these as security misconfigurations

### 💭 Questions to Consider
1. What could a malicious container do with these permissions?
2. When (if ever) would you use `--privileged`?
3. How can you prevent this in production?

<details>
<summary>Click for answers</summary>

1. **What could malicious container do?**
   - Steal sensitive data
   - Modify system files
   - Install backdoors
   - Pivot to other systems on the network
   - Cryptomining
   - Data exfiltration

2. **When to use --privileged?**
   - Docker-in-Docker (CI/CD systems)
   - System administration containers
   - Hardware access (GPU, USB devices)
   - **But there are usually better alternatives!**

3. **Prevention:**
   - Never use `-v /:/host` in production
   - Avoid `--privileged` unless absolutely necessary
   - Use security policies (AppArmor, SELinux, seccomp)
   - Implement admission controllers in Kubernetes
   - Use read-only mounts when possible: `-v /data:/data:ro`

</details>

---

## 🧹 Cleanup

After completing the lab:

```bash
# Stop and remove all demo containers
docker stop web1 web2 alpine-demo
docker rm web1 web2 alpine-demo

# Or use Docker's prune command
docker container prune -f
```

---

## 🎯 Lab Summary & Key Takeaways

### What You Learned

✅ **All containers share the host kernel**
- Verified by checking `uname -r` from multiple containers
- Security implication: kernel exploits affect everything

✅ **Container processes are visible from the host**
- Same process, different PID in different namespaces
- Host has full control over container processes

✅ **Namespaces provide isolation, not security**
- 6 types: PID, NET, MNT, UTS, IPC, USER
- They create separate views, but all use the same kernel

✅ **Misconfigurations can expose the entire host**
- Volume mounts: `-v /:/host` is extremely dangerous
- Privileged mode: `--privileged` removes most protections

✅ **Containers are NOT a security boundary**
- Think of them as process isolation, not security isolation
- Defense in depth is essential

### Security Checklist

When running containers in production:

- [ ] Never use `--privileged` unless absolutely necessary
- [ ] Be very careful with volume mounts
- [ ] Don't run containers as root (we'll cover this in Workshop 3)
- [ ] Use minimal base images (Workshop 2)
- [ ] Scan images for vulnerabilities (Workshop 2)
- [ ] Implement resource limits
- [ ] Use security profiles (seccomp, AppArmor, SELinux)
- [ ] Keep the host kernel updated

---

## 🤔 Challenge Questions (Optional)

Try to answer these without looking at documentation:

1. **Can a container access another container's processes?**
   <details>
   <summary>Answer</summary>
   
   No, not by default. PID namespaces prevent this. However:
   - The host can see all container processes
   - Containers sharing a PID namespace can (rare configuration)
   - Container with `--privileged` might be able to break out
   </details>

2. **What happens if you run `docker run --pid=host ubuntu ps aux`?**
   <details>
   <summary>Answer</summary>
   
   The container will see ALL host processes! The `--pid=host` flag makes the container share the host's PID namespace. This is dangerous and breaks isolation.
   </details>

3. **Can you have different kernel versions in different containers?**
   <details>
   <summary>Answer</summary>
   
   No. All containers must use the host's kernel. This is a fundamental limitation. If you need different kernels, use VMs, or specialized solutions like gVisor or Kata Containers.
   </details>

---

## 🚀 Next Steps

- Complete the discussion questions with your instructor
- Think about how these concepts apply to your own applications
- Get ready for **Workshop 2: Image Security & Attack Surface**

---

## 📚 Additional Experiments (For Advanced Students)

### Experiment 1: Namespace Sharing
```bash
# Create a container
docker run -d --name c1 alpine sleep 1000

# Create another container sharing its network namespace
docker run -it --rm --net=container:c1 alpine sh

# Inside: you'll share the same network as c1
ip addr  # Same IP as c1
```

### Experiment 2: Kernel Module Check
```bash
# List kernel modules from container
docker run --rm ubuntu lsmod

# Compare with host
lsmod

# Same modules = shared kernel!
```

### Experiment 3: Cgroup Limits
```bash
# Run container with memory limit
docker run -it --rm --memory="100m" ubuntu bash

# Inside:
cat /sys/fs/cgroup/memory/memory.limit_in_bytes

# This shows how cgroups limit resources
```

---

## ❓ Troubleshooting

**Problem:** "Can't see namespace information"
```bash
# Solution: Need sudo for /proc access
sudo ls -la /proc/<pid>/ns/
```

**Problem:** "Permission denied when running docker commands"
```bash
# Solution: Add your user to docker group (requires logout)
sudo usermod -aG docker $USER
# Then logout and login again
```

**Problem:** "Container exited immediately"
```bash
# Solution: Check logs
docker logs <container_name>

# Containers need a long-running process
docker run -d alpine sleep 1000  # ✓ Correct
docker run -d alpine echo "hi"   # ✗ Wrong (exits immediately)
```

**Problem:** "Cannot remove container"
```bash
# Solution: Force removal
docker rm -f <container_name>
```
