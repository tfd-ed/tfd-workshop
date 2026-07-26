# Hands-On Lab: Token-Efficient AI Workflows

**Duration**: 30-45 minutes  
**Difficulty**: Intermediate

## 🎯 Objectives

By completing this lab, you will:

1. Measure token usage for different prompting approaches
2. Practice divide-and-conquer task breakdown
3. Compare monolithic vs. incremental workflows
4. Apply token-efficient prompt patterns
5. Use agentic features effectively

## 📋 Prerequisites

- AI coding assistant with agentic capabilities (GitHub Copilot, Cursor, etc.)
- Python 3.8+ or Node.js 16+
- Git
- Basic command line familiarity

## 🔧 Lab Setup

### Option 1: Python Project
```bash
mkdir token-efficiency-lab
cd token-efficiency-lab
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install flask sqlalchemy pytest tiktoken
```

### Option 2: Node.js Project
```bash
mkdir token-efficiency-lab
cd token-efficiency-lab
npm init -y
npm install express sqlite3 jest
```

We'll use Python for the examples below, but principles apply to any language.

---

## Exercise 1: Token Awareness

**Goal**: Understand what tokens actually are and how they're counted.

### Task 1.1: Measure Your Code

Create a file `token_counter.py`:

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text for a given model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Test different code samples
samples = {
    "Simple function": """
def add(a, b):
    return a + b
""",
    
    "Function with types": """
def add(a: int, b: int) -> int:
    return a + b
""",
    
    "Function with docstring": """
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    \"\"\"
    return a + b
""",
    
    "Full implementation": """
from typing import Union

def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    \"\"\"Add two numbers with type checking.
    
    Args:
        a: First number (int or float)
        b: Second number (int or float)
        
    Returns:
        Sum of a and b
        
    Raises:
        TypeError: If inputs are not numbers
    \"\"\"
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers")
    return a + b
"""
}

print("Token counts for different code styles:\n")
for name, code in samples.items():
    tokens = count_tokens(code)
    print(f"{name:25s}: {tokens:4d} tokens")
    print(f"  ({len(code):4d} characters, {len(code.splitlines()):2d} lines)")
    print()
```

**Run it:**
```bash
python token_counter.py
```

### Expected Output

```
Token counts for different code styles:

Simple function           :   14 tokens
  (  37 characters,  3 lines)

Function with types       :   22 tokens
  (  53 characters,  3 lines)

Function with docstring   :   89 tokens
  ( 227 characters, 13 lines)

Full implementation       :  156 tokens
  ( 478 characters, 20 lines)
```

### 📝 Reflection Questions

1. How much do type hints increase token count?
2. How expensive are docstrings in tokens?
3. For AI generation, which version would you request for a simple task?

<details>
<summary>💡 Discussion Points</summary>

- Type hints: ~50% increase (14 → 22 tokens) - worth it for clarity
- Docstrings: ~4x increase (22 → 89 tokens) - don't ask AI to generate unless needed
- Full implementation: ~7x increase (22 → 156 tokens) - only request when necessary

**Best practice**: Ask for working code first, add documentation in a second step if needed.
</details>

---

## Exercise 2: The Cost of Vague Prompts

**Goal**: Experience the token waste from imprecise prompts.

### Task 2.1: Vague vs. Specific

**Scenario**: You need to create a user registration endpoint.

#### Round 1: Vague Prompt (DO THIS FIRST)

Prompt your AI:
```
"Create a user registration system"
```

**Observe:**
- How many files did it create?
- How many lines of code?
- Did it include things you didn't need?
- Estimate tokens (characters ÷ 4 is a rough approximation)

#### Round 2: Specific Prompt

Delete the generated code. Now prompt:
```
"Create a single function register_user(email, password) that:
1. Validates email format (must contain @)
2. Validates password length (min 8 characters)
3. Returns dict with 'success' and 'message' keys
4. Max 20 lines of code
5. No database, no hashing yet - just validation"
```

**Compare:**
- Lines of code generated
- Token count (use your `token_counter.py`)
- Time to review
- Likelihood of errors

### Task 2.2: Measure the Difference

Use the token counter:

```python
# Save each AI response to a file
with open("vague_response.txt", "r") as f:
    vague = f.read()

with open("specific_response.txt", "r") as f:
    specific = f.read()

print(f"Vague prompt response:    {count_tokens(vague)} tokens")
print(f"Specific prompt response: {count_tokens(specific)} tokens")
print(f"Difference: {count_tokens(vague) - count_tokens(specific)} tokens")
print(f"Savings: {100 * (1 - count_tokens(specific) / count_tokens(vague)):.1f}%")
```

### 📝 Expected Results

```
Vague prompt response:    1247 tokens
Specific prompt response:  243 tokens
Difference: 1004 tokens
Savings: 80.5%
```

---

## Exercise 3: Divide and Conquer

**Goal**: Practice breaking down complex tasks.

### Task 3.1: Build a TODO API (Incremental Approach)

Build this incrementally, **one prompt at a time**. Track tokens for each.

#### Step 1: Data Model (Prompt 1)
```
"Create a Task class with these attributes:
- id (int)
- title (str)
- completed (bool, default False)
- created_at (datetime, auto-set to now)

Include __init__ and __repr__ methods. Max 15 lines."
```

**Test it:**
```python
from datetime import datetime

task = Task(1, "Learn about tokens")
print(task)
# Expected: Task(id=1, title='Learn about tokens', completed=False)
```

#### Step 2: Storage (Prompt 2)
```
"Create a TaskStore class with:
- __init__(): creates empty list self.tasks
- add(task): appends task to list
- get(task_id): returns task with matching id, or None
- get_all(): returns all tasks

Max 20 lines."
```

**Test it:**
```python
store = TaskStore()
task = Task(1, "Test task")
store.add(task)
print(store.get(1))
print(len(store.get_all()))  # Should be 1
```

#### Step 3: Flask Endpoint (Prompt 3)
```
"Create a Flask app with POST /tasks endpoint that:
- Accepts JSON with 'title' field
- Creates a Task with auto-incremented id
- Adds to global TaskStore instance
- Returns the task as JSON

Use this pattern:
@app.post('/tasks')
def create_task():
    # your implementation

Include the Flask app setup. Max 25 lines total."
```

**Test it:**
```bash
flask run
curl -X POST http://localhost:5000/tasks -H "Content-Type: application/json" -d '{"title":"Buy groceries"}'
```

#### Step 4: Validation (Prompt 4)
```
"Update the create_task endpoint to:
- Return 400 if 'title' is missing
- Return 400 if 'title' is empty or > 100 chars
- Return proper JSON error messages

Show only the updated create_task function."
```

#### Step 5: GET Endpoint (Prompt 5)
```
"Add GET /tasks endpoint that returns all tasks as JSON array.
Follow the same pattern as POST /tasks."
```

### Task 3.2: Track Your Tokens

After each step, record:

| Step | Prompt Tokens (approx) | Response Tokens | Total |
|------|------------------------|-----------------|-------|
| 1. Task class | ~60 | ~150 | 210 |
| 2. TaskStore | ~70 | ~180 | 250 |
| 3. Flask endpoint | ~90 | ~220 | 310 |
| 4. Validation | ~80 | ~120 | 200 |
| 5. GET endpoint | ~60 | ~100 | 160 |
| **Total** | **360** | **770** | **1,130** |

### Task 3.3: Compare to Monolithic Approach

Now try the monolithic approach (in a new directory):

```
"Create a complete Flask TODO API with:
- Task model with id, title, completed, created_at
- In-memory storage
- POST /tasks endpoint with validation
- GET /tasks endpoint
- Error handling
- JSON responses"
```

**Measure:**
- Total tokens (likely 2,500-4,000)
- Number of files generated
- Issues found during testing
- Time spent debugging

### 📝 Reflection

**Questions:**
1. Which approach resulted in fewer total tokens?
2. Which was easier to test and debug?
3. Which gave you more control over the result?

<details>
<summary>💡 Expected Findings</summary>

**Incremental:**
- ~1,130 tokens total
- Each step testable
- Easy to debug
- More control

**Monolithic:**
- ~3,000+ tokens
- All-or-nothing testing
- Harder to debug
- AI makes assumptions

**Token savings: ~60%**
**Time savings: Similar total time, but less debugging**
**Quality improvement: Incremental gives cleaner, more predictable code**
</details>

---

## Exercise 4: Token-Efficient Patterns

**Goal**: Practice patterns that minimize token usage.

### Task 4.1: Pattern Reuse

You need three similar endpoints. Compare approaches:

#### ❌ Approach A: Regenerate Each Time

```
Prompt 1: "Create GET /users endpoint that returns all users from users list"
Prompt 2: "Create GET /products endpoint that returns all products from products list"
Prompt 3: "Create GET /orders endpoint that returns all orders from orders list"
```

Each prompt generates ~200 tokens of similar code.  
Total: ~600 tokens

#### ✅ Approach B: Pattern Then Apply

```
Prompt 1: "Create a generic function make_list_endpoint(resource_name, data_list) 
that returns a Flask endpoint function for GET /{resource_name}"

Prompt 2: "Use make_list_endpoint to create GET /users, /products, and /orders endpoints"
```

Total: ~300 tokens

**Task**: Implement both approaches and measure actual token counts.

### Task 4.2: Diffs vs. Full Regeneration

You have a 50-line file and need to change one function.

#### ❌ Approach A: Full Regeneration

```
"Update the validate_email function in utils.py to also check for + signs"
```

AI regenerates entire 50-line file = ~600 tokens

#### ✅ Approach B: Request Just the Function

```
"Show only the updated validate_email function from utils.py"
```

AI returns just the function = ~80 tokens

**Savings: 87%**

### Task 4.3: Batch Related Changes

#### ❌ Sequential Single Changes

```
Prompt 1: "Add type hint to calculate_total function"
Prompt 2: "Add type hint to calculate_tax function"  
Prompt 3: "Add type hint to calculate_discount function"
```

Each loads context separately = ~800 tokens × 3 = 2,400 tokens

#### ✅ Batched Changes

```
"Add type hints to these functions in calculator.py: 
calculate_total, calculate_tax, calculate_discount"
```

Single context load = ~900 tokens

**Savings: 62%**

---

## Exercise 5: Agentic AI Practice

**Goal**: Learn to control agentic features effectively.

### Task 5.1: Supervised vs. Autonomous

#### Scenario A: Autonomous (Measure the Cost)

```
"Refactor my app to use proper error handling throughout"
```

**What might happen:**
- Agent searches entire codebase
- Reads 20+ files
- Generates changes across 10+ files
- Runs tests multiple times
- Token cost: 50k+

#### Scenario B: Supervised

```
Step 1: "List all functions in app.py that don't have try-except blocks"
Step 2: [Review list]
Step 3: "Add try-except to these 3 functions: login, register, update_profile"
Step 4: [Test]
Step 5: "Continue with the next 3 functions"
```

**Token cost**: ~8k

**Task**: Try both approaches on your TODO API. Measure the difference.

### Task 5.2: Set Boundaries

Practice controlling the agent:

```
"Add logging to the create_task function.
DO NOT:
- Modify any other functions
- Install new packages
- Create new files
- Run tests

ONLY:
- Add logging statements to create_task
- Show me the updated function"
```

**Compare to:**
```
"Add logging to create_task"
```

Observe how the constrained prompt prevents over-generation.

---

## Exercise 6: Real-World Scenario

**Goal**: Apply all techniques to a realistic task.

### Scenario: Add Authentication to Your TODO API

Use **all** the token-efficient techniques you've learned:

#### Your Approach

1. **Break it down** (Divide & Conquer)
2. **Be specific** (Clear prompts)
3. **Incremental** (One feature at a time)
4. **Supervised** (Review before executing)
5. **Reuse patterns** (DRY principle)

#### Suggested Steps

```markdown
Step 1: Plan
"I need to add authentication to my TODO API. 
List the components needed:
- User model
- Password hashing
- Login endpoint
- Protected routes
Break it into 5-6 incremental tasks."

Step 2: User Model
"Create a User class with id, email, password_hash.
Include a check_password(plain_password) method.
Max 20 lines."

Step 3: Password Hashing
"Create hash_password(password) and verify_password(plain, hashed) functions 
using hashlib and secrets. Max 15 lines total."

Step 4: Registration
"Add POST /register endpoint that:
- Accepts email and password
- Validates email format and password length (8+)
- Hashes password
- Stores user
- Returns user id
Max 25 lines."

Step 5: Login
"Add POST /login endpoint that:
- Accepts email and password
- Finds user by email
- Verifies password
- Returns success/failure
Max 20 lines."

Step 6: Protection (later)
"I'll add route protection next, but stop here for now."
```

### Track Your Metrics

| Metric | Your Result |
|--------|-------------|
| Total prompts | |
| Total tokens (estimate) | |
| Working features | |
| Bugs encountered | |
| Time spent | |

### Compare to Monolithic

Try the same with one big prompt:
```
"Add complete authentication to my TODO API with user registration, 
login, password hashing, and protected routes"
```

Record the same metrics.

---

## Exercise 7: Token Budget Challenge

**Goal**: Build a feature within a token budget.

### The Challenge

**Budget**: 2,000 tokens total (prompt + response)  
**Task**: Add a "mark task complete" feature to your TODO API

**Requirements:**
- PATCH /tasks/<id> endpoint
- Updates task.completed to True
- Returns updated task
- Handles task not found (404)
- Includes basic test

### Strategy

Plan your prompts to stay under budget:

```markdown
Prompt 1 (~400 tokens): Create the endpoint
Prompt 2 (~300 tokens): Add error handling
Prompt 3 (~500 tokens): Create pytest test
Prompt 4 (~200 tokens): Fix any issues

Total: ~1,400 tokens (600 token buffer)
```

Use `token_counter.py` to track your actual usage.

### Success Criteria

- ✅ Feature works correctly
- ✅ Under 2,000 token budget
- ✅ Code is clean and testable

---

## Bonus Challenges

### Challenge 1: Optimize an Existing Prompt

Take a prompt you've used recently. Rewrite it to use ≤50% of the original tokens while getting equivalent or better results.

### Challenge 2: Framework-Specific Skill

Pick a framework you use (Django, React, FastAPI, etc.). Research its "skills" or best practices. Write a prompt that leverages framework-specific patterns effectively.

### Challenge 3: Build a Token Tracker

Create a simple tool that:
- Logs each AI interaction
- Tracks tokens used
- Shows cumulative daily/weekly usage
- Alerts when approaching budget limits

---

## 📊 Lab Completion Checklist

- [ ] Completed token counting exercise
- [ ] Compared vague vs. specific prompts
- [ ] Practiced divide-and-conquer with TODO API
- [ ] Applied token-efficient patterns
- [ ] Controlled agentic behavior
- [ ] Completed real-world authentication scenario
- [ ] Stayed within token budget challenge
- [ ] (Bonus) Attempted at least one bonus challenge

---

## 🎓 Key Takeaways

After completing this lab, you should understand:

1. **Token costs are real**: Small inefficiencies add up quickly
2. **Specificity wins**: Clear prompts use fewer tokens and get better results
3. **Incremental > Monolithic**: Breaking tasks down saves tokens and improves quality
4. **Control agentic features**: Supervision prevents runaway token usage
5. **Patterns over repetition**: Reuse code patterns to minimize generation

---

## 📈 Next Steps

1. **Apply to your projects**: Use these techniques in your daily work
2. **Track your usage**: Monitor token consumption over the next week
3. **Share with team**: Teach these patterns to colleagues
4. **Iterate and improve**: Find what works best for your workflow

---

## 🔗 Resources

- [Token counter code](../scripts/demo/token_counter.py)
- [Complete solution](../scripts/demo/todo_api_solution.py)
- [Workshop content](../materials/workshop-2-content.md)

---

## ❓ Need Help?

If you get stuck:
1. Check the [workshop content](../materials/workshop-2-content.md) for examples
2. Review the [instructor notes](../INSTRUCTOR_NOTES.md)
3. Ask in the workshop discussion forum

Happy token-efficient coding! 🚀
