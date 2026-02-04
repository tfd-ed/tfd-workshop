#!/bin/bash

# Workshop 1: Container Security Basics - Lab Setup Script
# Run this before the hands-on lab section

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Workshop 1: Lab Environment Setup        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}\n"

echo -e "${YELLOW}Pulling required Docker images...${NC}"
docker pull ubuntu:latest
docker pull alpine:latest
docker pull nginx:latest

echo -e "\n${YELLOW}Starting demo containers...${NC}"
docker run -d --name web1 nginx
docker run -d --name web2 nginx
docker run -d --name alpine-demo alpine sleep 3600

echo -e "\n${GREEN}✓ Lab setup complete!${NC}\n"

echo -e "${BLUE}Available containers:${NC}"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"

echo -e "\n${GREEN}You're ready to start the hands-on exercises!${NC}"
