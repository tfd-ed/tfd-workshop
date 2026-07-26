#!/bin/bash
# Lab Setup Script for Workshop 2: AI Token-Efficient Usage
# TFD Workshop Series - AI/ML Track

set -e

echo "======================================================================"
echo "Workshop 2: AI Token-Efficient Usage - Lab Setup"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or later from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"
echo ""

# Create lab directory
LAB_DIR="token-efficiency-lab"
echo "Creating lab directory: $LAB_DIR"

if [ -d "$LAB_DIR" ]; then
    echo -e "${YELLOW}⚠️  Directory $LAB_DIR already exists${NC}"
    read -p "Do you want to remove it and start fresh? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$LAB_DIR"
        echo "Removed existing directory"
    else
        echo "Keeping existing directory"
    fi
fi

if [ ! -d "$LAB_DIR" ]; then
    mkdir "$LAB_DIR"
    echo -e "${GREEN}✅ Created $LAB_DIR${NC}"
fi

cd "$LAB_DIR"

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install required packages
echo ""
echo "Installing required packages..."
echo "  - tiktoken (token counting)"
echo "  - flask (API examples)"
echo "  - pytest (testing)"

pip install -q tiktoken flask pytest

echo -e "${GREEN}✅ All packages installed${NC}"

# Create directory structure
echo ""
echo "Creating directory structure..."

mkdir -p exercises
mkdir -p demos
mkdir -p scripts

# Create token counter utility
echo ""
echo "Creating token_counter.py utility..."
cat > scripts/token_counter.py << 'EOF'
#!/usr/bin/env python3
"""
Token Counter Utility
Count tokens in text using OpenAI's tiktoken library.
"""

import tiktoken
import sys


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text for a given model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def main():
    if len(sys.argv) < 2:
        print("Usage: python token_counter.py <text>")
        print("   or: python token_counter.py --file <filename>")
        print("   or: cat file.py | python token_counter.py --stdin")
        return
    
    if sys.argv[1] == "--file" and len(sys.argv) >= 3:
        with open(sys.argv[2], 'r') as f:
            text = f.read()
    elif sys.argv[1] == "--stdin":
        text = sys.stdin.read()
    else:
        text = ' '.join(sys.argv[1:])
    
    tokens = count_tokens(text)
    chars = len(text)
    lines = len(text.splitlines())
    
    print(f"Tokens: {tokens}")
    print(f"Characters: {chars}")
    print(f"Lines: {lines}")
    print(f"Ratio: {chars / tokens if tokens > 0 else 0:.2f} chars/token")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/token_counter.py
echo -e "${GREEN}✅ Created token_counter.py${NC}"

# Create example README
echo ""
echo "Creating lab README..."
cat > README.md << 'EOF'
# Token Efficiency Lab

Welcome to the hands-on lab for Workshop 2: AI Token-Efficient Usage!

## Setup Complete ✅

Your lab environment is ready with:
- Python virtual environment
- tiktoken (token counting library)
- Flask (for API examples)
- pytest (for testing)

## Getting Started

### 1. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 2. Test Token Counter

```bash
python scripts/token_counter.py "Hello, world!"
```

### 3. Count Tokens in a File

```bash
python scripts/token_counter.py --file yourfile.py
```

### 4. Try the Exercises

See the full hands-on lab exercises:
[exercises/hands-on-lab.md](../exercises/hands-on-lab.md)

## Quick Examples

### Count tokens in code:

```python
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4")

code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""

tokens = len(encoder.encode(code))
print(f"Tokens: {tokens}")
```

### Compare prompt efficiency:

```python
vague = "Create a user registration system"
specific = "Create a function register_user(email, password) that validates email format and returns a dict"

print(f"Vague: {len(encoder.encode(vague))} tokens")
print(f"Specific: {len(encoder.encode(specific))} tokens")
```

## Resources

- Workshop Content: [materials/workshop-2-content.md](../materials/workshop-2-content.md)
- Demo Script: [scripts/demo-script.py](../scripts/demo-script.py)
- Exercises: [exercises/hands-on-lab.md](../exercises/hands-on-lab.md)

## Tips

1. **Measure everything**: Use the token counter before and after optimization
2. **Start small**: Practice with simple examples first
3. **Compare approaches**: Try vague vs. specific prompts
4. **Track savings**: Calculate your token and cost reductions

Happy learning! 🚀
EOF

echo -e "${GREEN}✅ Created README.md${NC}"

# Create a simple example file
echo ""
echo "Creating example files..."
cat > demos/example.py << 'EOF'
# Example: Simple vs. Verbose Code
# Compare token counts

def add_simple(a, b):
    return a + b


def add_with_types(a: int, b: int) -> int:
    return a + b


def add_with_docs(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b


def add_comprehensive(a: int, b: int) -> int:
    """Add two numbers with comprehensive type checking.
    
    This function performs addition on integer inputs with proper
    type validation and error handling.
    
    Args:
        a: First integer number
        b: Second integer number
        
    Returns:
        Sum of a and b as an integer
        
    Raises:
        TypeError: If either argument is not an integer
        
    Examples:
        >>> add_comprehensive(2, 3)
        5
    """
    if not isinstance(a, int):
        raise TypeError(f"First argument must be int, got {type(a)}")
    if not isinstance(b, int):
        raise TypeError(f"Second argument must be int, got {type(b)}")
    return a + b


# Use the token counter to see the difference!
# python ../scripts/token_counter.py --file example.py
EOF

echo -e "${GREEN}✅ Created demos/example.py${NC}"

# Create exercises directory structure
mkdir -p exercises/exercise1
mkdir -p exercises/exercise2
mkdir -p exercises/exercise3

# Print completion message
echo ""
echo "======================================================================"
echo -e "${GREEN}✅ Lab setup complete!${NC}"
echo "======================================================================"
echo ""
echo "📁 Lab directory: $(pwd)"
echo ""
echo "🚀 Next steps:"
echo ""
echo "  1. Read the lab README:"
echo "     cat README.md"
echo ""
echo "  2. Try the token counter:"
echo "     python scripts/token_counter.py 'Hello, world!'"
echo ""
echo "  3. Count tokens in example file:"
echo "     python scripts/token_counter.py --file demos/example.py"
echo ""
echo "  4. Start the exercises:"
echo "     See: ../exercises/hands-on-lab.md"
echo ""
echo "  5. Run the demo script:"
echo "     python ../scripts/demo-script.py"
echo ""
echo "💡 Tips:"
echo "  - Keep the virtual environment activated"
echo "  - Use the token counter frequently"
echo "  - Compare before/after for every optimization"
echo ""
echo "📚 Resources:"
echo "  - Workshop content: ../materials/workshop-2-content.md"
echo "  - Hands-on lab: ../exercises/hands-on-lab.md"
echo "  - Instructor notes: ../INSTRUCTOR_NOTES.md"
echo ""
echo "======================================================================"
echo "Virtual environment is activated. To deactivate, run: deactivate"
echo "======================================================================"
echo ""
