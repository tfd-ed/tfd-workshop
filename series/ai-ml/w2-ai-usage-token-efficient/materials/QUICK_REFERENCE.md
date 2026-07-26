# Token-Efficient AI: Quick Reference Cheat Sheet

**Workshop 2: AI Is Getting Expensive — Stop Wasting Tokens!**

---

## 🎯 Core Principles

1. **Tokens = Money** → Every token counts
2. **Clarity > Cleverness** → Simple beats elaborate
3. **Small Tasks > Big Tasks** → Break it down
4. **Iteration > Perfection** → Refine, don't predict
5. **Supervision > Autonomy** → Guide, don't let loose

---

## 💰 Token Economics

| Cost Component | Example | Token Count |
|---------------|---------|-------------|
| Simple prompt | "Create a login function" | ~10 tokens |
| Specific prompt | "Create function check_credentials(user, pass)..." | ~30 tokens |
| Over-engineered | "Act as a senior engineer with 20 years..." | ~150 tokens |
| Code (minimal) | `def add(a, b): return a + b` | ~15 tokens |
| Code (documented) | Same function + docstring | ~90 tokens |

**Pricing (GPT-4 Turbo)**:
- Input: $10 per 1M tokens
- Output: $30 per 1M tokens
- Output is 3x more expensive!

---

## ✅ Efficient Prompt Patterns

### ❌ Avoid

```
"Act as an expert senior software engineer with 20+ years of experience.
Think step by step. Explain your reasoning. Consider all edge cases.
Write production-ready code with comprehensive error handling and tests."
```
→ 150+ tokens, vague requirements

### ✅ Use

```
"Create a function validate_email(email: str) -> bool that:
- Checks for @ symbol
- Returns True/False
- Max 5 lines"
```
→ 30 tokens, clear requirements

---

## 🔧 Token-Saving Techniques

### 1. Be Specific, Not Elaborate
- ❌ "Make it better"
- ✅ "Add error handling for null email"

### 2. Minimize Context
- ❌ Paste entire file in prompt
- ✅ "In auth.py, update line 45"

### 3. Request Diffs
- ❌ "Update the file"
- ✅ "Show only the changed function"

### 4. Batch Changes
- ❌ 3 prompts: "Add type hint to foo", "to bar", "to baz"
- ✅ 1 prompt: "Add type hints to: foo, bar, baz"

### 5. Break Down Tasks
- ❌ "Build complete TODO API"
- ✅ "Step 1: Create Task model. Step 2: Add GET endpoint. Step 3: ..."

### 6. Use Constraints
- ❌ "Write good code"
- ✅ "Max 20 lines, type hints required, no external libraries"

### 7. Provide Examples
- ❌ "Follow best practices"
- ✅ "Follow this pattern: [show example]"

### 8. Limit Output Scope
- ❌ "Create a REST API"
- ✅ "Create ONE endpoint: GET /users/:id"

---

## 📊 Workflow Optimization

### Incremental Development

```
✅ Small Steps:
1. Create model (200 tokens)
2. Add validation (150 tokens)
3. Create endpoint (250 tokens)
4. Add tests (200 tokens)
Total: 800 tokens

❌ Monolithic:
1. "Build everything" (5000 tokens × 2-3 iterations)
Total: 10,000-15,000 tokens
```

**Savings: 92%**

---

## 🤖 Agentic AI Control

### Set Boundaries

```
✅ "Add logging to create_task function.
DO NOT:
- Modify other functions
- Install new packages
- Run tests automatically"
```

### Request Plans First

```
✅ "List all files you'll modify for adding auth.
Wait for my approval before making changes."
```

### Limit File Access

```
✅ "Read only: auth.py, config.py
Generate changes but don't write yet."
```

---

## 🎓 Real-World Patterns

### Pattern 1: Explain Then Generate
```
Round 1: "How would you add caching to this API?"
→ Review strategy

Round 2: "Implement Redis caching for /users endpoint"
→ Correct on first try
```

### Pattern 2: Reference Implementation
```
"Here's my existing GET /users endpoint: [code]
Create a similar POST /products endpoint."
→ Consistent style, less guessing
```

### Pattern 3: Incremental Refinement
```
Step 1: "Basic function"
Step 2: "Add validation"
Step 3: "Add error handling"
→ Test each step
```

### Pattern 4: Error-Driven
```
Round 1: Generate code
Round 2: Run it, see error
Round 3: "Fix JSONDecodeError when file is empty"
→ Precise fix
```

---

## 📈 Measuring Success

### Track These Metrics

```python
# Use tiktoken to measure
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4")
tokens = len(encoder.encode(your_prompt))
```

### Before/After Comparison

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| Tokens/request | 8,000 | 2,000 | 75% |
| Requests/day | 50 | 50 | - |
| Monthly tokens | 12M | 3M | 75% |
| Monthly cost | $120 | $30 | $90 |

---

## 🚫 Common Anti-Patterns

### 1. The Novel
Over-explaining requirements → Wasted tokens

### 2. The Roleplay
"Act as a 10x engineer..." → Doesn't improve output

### 3. The Everything Request
"Build X with Y and Z and..." → Hard to debug

### 4. The Context Dump
Pasting entire files → Use file paths instead

### 5. The Perfect First Try
Trying to specify everything → Iterate instead

---

## ⚡ Quick Wins

**Today:**
1. Use specific prompts instead of vague ones
2. Add constraints (max lines, no docs)
3. Request diffs, not full files

**This Week:**
1. Break down next complex task into 3-5 steps
2. Measure token usage before/after
3. Batch similar changes

**This Month:**
1. Track monthly token consumption
2. Compare to previous month
3. Calculate savings

---

## 🔢 Cost Calculator

```python
# Your usage
requests_per_day = 50
avg_input_tokens = 2000  # Your optimized amount
avg_output_tokens = 800

# Monthly
monthly_input = requests_per_day * avg_input_tokens * 30
monthly_output = requests_per_day * avg_output_tokens * 30

# Cost (GPT-4 Turbo)
input_cost = (monthly_input / 1_000_000) * 10
output_cost = (monthly_output / 1_000_000) * 30

total_cost = input_cost + output_cost
print(f"Monthly cost: ${total_cost:.2f}")
```

---

## 📚 Resources

- **Workshop Recording**: [Link]
- **Hands-On Lab**: [exercises/hands-on-lab.md](exercises/hands-on-lab.md)
- **Demo Script**: [scripts/demo-script.py](scripts/demo-script.py)
- **Token Counter**: `pip install tiktoken`
- **OpenAI Tokenizer**: https://platform.openai.com/tokenizer

---

## 🎯 Remember

**The Goal**: Not to save every token, but to:
- ✅ Build good efficiency habits
- ✅ Get better results faster
- ✅ Reduce unnecessary waste
- ✅ Work smarter with AI

**Think of AI as a junior developer:**
- Give clear, specific instructions
- Review their work
- Iterate and refine
- Guide, don't micromanage

---

## 💡 One-Line Takeaways

1. "Tokens are the new compute cost"
2. "Specific beats clever every time"
3. "Small tasks compound into big results"
4. "Measure, optimize, repeat"
5. "AI is a tool, you're the engineer"

---

**Keep this handy!** Print it out or save it for quick reference.

---

*Part of the TFD Workshop Series - Teaching for Development*  
*Workshop 2: AI Token-Efficient Usage*
