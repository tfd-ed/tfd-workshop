#!/bin/bash

# Workshop 1: Container Security Basics - Demo Script
# This script demonstrates key container security concepts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function for section headers
section() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

pause() {
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up demo containers...${NC}"
    docker rm -f demo-container web1 web2 alpine-demo 2>/dev/null || true
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT

echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Workshop 1: Container Security Basics    ║${NC}"
echo -e "${GREEN}║           Live Demo Script                ║${NC}"
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"

# ============================================================
# Demo 1: Prove Shared Kernel
# ============================================================
section "Demo 1: Proving All Containers Share the Same Kernel"

echo -e "${YELLOW}Host Kernel Version:${NC}"
uname -r
cat /proc/version

pause

echo -e "\n${YELLOW}Container 1 (Ubuntu):${NC}"
docker run --rm ubuntu uname -r

echo -e "\n${YELLOW}Container 2 (Alpine):${NC}"
docker run --rm alpine uname -r

echo -e "\n${YELLOW}Container 3 (Nginx):${NC}"
docker run --rm nginx uname -r

echo -e "\n${RED}💡 Notice: All containers show THE SAME kernel version!${NC}"
echo -e "${RED}   This proves they're all sharing the host's kernel.${NC}"

pause

# ============================================================
# Demo 2: Container Processes from Host
# ============================================================
section "Demo 2: Viewing Container Processes from Host"

echo -e "${YELLOW}Starting a demo container...${NC}"
docker run -d --name demo-container alpine sleep 3600
echo -e "${GREEN}Container started!${NC}"

pause

echo -e "\n${YELLOW}Container's process ID (from Docker's perspective):${NC}"
CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' demo-container)
echo -e "${GREEN}Container PID: $CONTAINER_PID${NC}"

pause

echo -e "\n${YELLOW}Viewing this process from the HOST:${NC}"
ps -p $CONTAINER_PID -o pid,ppid,user,command

echo -e "\n${RED}💡 The container process is visible from the host!${NC}"
echo -e "${RED}   Containers are just regular processes with special isolation.${NC}"

pause

echo -e "\n${YELLOW}Inside the container, what does it think its PID is?${NC}"
docker exec demo-container ps aux

echo -e "\n${RED}💡 Inside the container, it thinks it's PID 1!${NC}"
echo -e "${RED}   This is namespace isolation in action.${NC}"

pause

# ============================================================
# Demo 3: Inspect Namespaces
# ============================================================
section "Demo 3: Linux Namespaces Inspection"

echo -e "${YELLOW}Container's namespaces:${NC}"
echo -e "${YELLOW}(These are the isolation mechanisms Docker uses)${NC}\n"

if [ -d "/proc/$CONTAINER_PID/ns" ]; then
    ls -la /proc/$CONTAINER_PID/ns/ 2>/dev/null || {
        echo -e "${RED}Need sudo to view namespaces. Running with sudo:${NC}"
        sudo ls -la /proc/$CONTAINER_PID/ns/
    }
else
    echo -e "${RED}Cannot access namespace information${NC}"
fi

pause

echo -e "\n${YELLOW}Compare with host (PID 1) namespaces:${NC}"
sudo ls -la /proc/1/ns/ 2>/dev/null || ls -la /proc/1/ns/

echo -e "\n${RED}💡 Notice the namespace IDs are DIFFERENT!${NC}"
echo -e "${RED}   Each namespace type has a unique ID for the container.${NC}"
echo -e "${RED}   Types: mnt, net, pid, uts, ipc, user${NC}"

pause

# ============================================================
# Demo 4: Shared Kernel Information Access
# ============================================================
section "Demo 4: Container Access to Kernel Information"

echo -e "${YELLOW}Container reading CPU information from host kernel:${NC}"
docker run --rm alpine cat /proc/cpuinfo | head -20

pause

echo -e "\n${YELLOW}Container reading memory information:${NC}"
docker run --rm alpine cat /proc/meminfo | head -15

pause

echo -e "\n${YELLOW}Container reading kernel parameters:${NC}"
docker run --rm alpine sysctl kernel.hostname kernel.version 2>/dev/null || {
    docker run --rm alpine cat /proc/sys/kernel/hostname
}

echo -e "\n${RED}💡 Containers can read kernel information because they share it!${NC}"
echo -e "${RED}   This is why kernel security is critical.${NC}"

pause

# ============================================================
# Demo 5: Process Tree Visibility
# ============================================================
section "Demo 5: Understanding Process Isolation"

echo -e "${YELLOW}Starting multiple containers...${NC}"
docker run -d --name web1 nginx
docker run -d --name web2 nginx
docker run -d --name alpine-demo alpine sleep 3600

echo -e "${GREEN}Containers started!${NC}"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"

pause

echo -e "\n${YELLOW}All container processes from host perspective:${NC}"
ps aux | grep -E "nginx|sleep 3600" | grep -v grep

echo -e "\n${RED}💡 All container processes are visible from the host!${NC}"
echo -e "${RED}   The host can see everything happening inside containers.${NC}"

pause

echo -e "\n${YELLOW}Inside container 'web1', can it see other containers?${NC}"
docker exec web1 ps aux

echo -e "\n${RED}💡 Container only sees its own processes!${NC}"
echo -e "${RED}   PID namespace provides this isolation.${NC}"

pause

# ============================================================
# Demo 6: Breaking the Illusion (Safely)
# ============================================================
section "Demo 6: How Easy It Is To Break Container Isolation"

echo -e "${RED}⚠️  WARNING: This demo shows dangerous configurations!${NC}"
echo -e "${RED}⚠️  NEVER do this in production!${NC}\n"

pause

echo -e "${YELLOW}Running container with host filesystem mounted:${NC}"
echo -e "${YELLOW}Command: docker run --rm -v /:/host ubuntu ls /host${NC}\n"

docker run --rm -v /:/host ubuntu ls /host

echo -e "\n${RED}💡 The container can see the ENTIRE host filesystem!${NC}"
echo -e "${RED}   With write access, it could modify anything.${NC}"

pause

echo -e "\n${YELLOW}What about accessing sensitive files?${NC}"
docker run --rm -v /:/host ubuntu ls /host/etc/passwd

echo -e "\n${RED}💡 Access to sensitive host files!${NC}"
echo -e "${RED}   This is why volume mounts must be carefully controlled.${NC}"

pause

# ============================================================
# Summary
# ============================================================
section "Demo Summary - Key Security Insights"

echo -e "${GREEN}✓ All containers share the same host kernel${NC}"
echo -e "${GREEN}✓ Container processes are visible from the host${NC}"
echo -e "${GREEN}✓ Namespaces provide isolation, not security${NC}"
echo -e "${GREEN}✓ Improper configuration can expose the entire host${NC}"
echo -e "${GREEN}✓ Containers are processes, not virtual machines${NC}"

echo -e "\n${YELLOW}Key Takeaway:${NC}"
echo -e "${RED}Containers provide CONVENIENCE, not automatic SECURITY!${NC}"

echo -e "\n${BLUE}Next: Hands-on lab time!${NC}"

# Cleanup happens automatically via trap
