## How to Setup Development Environment (Linux)

This guide will help you set up the development environment on Linux.

### Prerequisites

- Make sure you have sudo privileges
- Terminal/Command line access

### Installation Steps

1. Install Rye (Python Package Manager):
```bash
curl -sSf https://rye.astral.sh/get | bash
```

2. Install Required System Packages:
```bash
# Install C compiler
sudo apt install -y clang

# Install C++ development libraries
sudo apt install libstdc++-12-dev
```

3. Setup Python Environment:
```bash
# Sync Python dependencies
rye sync

# Install package in development mode
python setup.py develop
```