# Workshop Assets

This directory contains visual aids and diagrams to enhance understanding of token efficiency and context windows in AI systems.

## 📊 Available Visual Aids

### 1. Context Windows Diagram (`context-windows.png`)

**Purpose**: Visualize how context windows work and what counts toward the token limit.

**Shows**:
- Components that fill the context window (system prompt, history, files, user input, AI response)
- Visual representation of token allocation
- How context accumulates over conversation turns
- When context window overflow occurs

**When to Use**:
- Part 1: Understanding the Foundation → "Understanding Context Windows" section
- Explaining why token usage compounds over time
- Demonstrating the hidden costs of long conversations
- Teaching context window management strategies

**Key Teaching Points**:
- Everything counts toward the limit
- History accumulates with each turn
- Larger windows enable (and hide) inefficient behavior
- Context overflow leads to lost information

---

### 2. LLM No Memory Diagram (`llm-no-memory.png`)

**Purpose**: Illustrate that LLMs are stateless and don't remember between requests.

**Shows**:
- Each request is independent
- Context must be explicitly provided in each request
- Why conversation history grows (to maintain context)
- The relationship between statelessness and token costs

**When to Use**:
- Part 1: Understanding the Foundation → "How LLMs Work: A Quick Overview"
- Explaining why context windows exist
- Teaching why history is re-sent with each request
- Demonstrating the token cost of maintaining conversation state

**Key Teaching Points**:
- LLMs have no memory between requests
- All context must be re-provided each time
- This is why tokens compound in long conversations
- Understanding this helps design token-efficient workflows

---

## 🎓 How to Use These in Your Workshop

### During Presentation

1. **Context Windows Diagram**:
   ```markdown
   ![Context Window Components](../assets/context-windows.png)
   
   As you can see, everything counts toward your context window limit:
   - System instructions (always present)
   - Previous conversation (grows with each turn)
   - Loaded files (can be huge!)
   - Your actual prompt (usually tiny!)
   - AI's response (as it generates)
   ```

2. **LLM No Memory Diagram**:
   ```markdown
   ![LLMs are Stateless](../assets/llm-no-memory.png)
   
   This diagram shows why token costs compound:
   - Each request is independent
   - To "remember" context, we resend the history
   - After 10 turns, you're sending 10x the information
   - This is the hidden cost of long conversations
   ```

### In Hands-On Exercises

Reference these diagrams when students ask:
- "Why does it cost more after many messages?"
  → Show the context windows diagram
  
- "Can't the AI just remember what I said?"
  → Show the LLM no memory diagram

### In Documentation

Both images are referenced in:
- `materials/workshop-2-content.md` - Main teaching content
- `exercises/hands-on-lab.md` - Student exercises
- `INSTRUCTOR_NOTES.md` - Teaching guide

---

## 🎨 Creating Additional Diagrams

If you want to create more visual aids for this workshop, consider:

### Suggested Diagrams

1. **Token Cost Comparison Chart**
   - Bar chart showing cost per model
   - Input vs output pricing
   - Monthly cost scenarios

2. **Efficient vs Inefficient Workflow**
   - Side-by-side comparison
   - Show token consumption over time
   - Highlight savings

3. **Agentic AI Architecture**
   - How agent, tools, and planning work together
   - Token flow through the system
   - Where costs accumulate

4. **Prompt Engineering Impact**
   - Before/after examples
   - Token reduction visualization
   - Quality vs quantity trade-offs

### Tools for Creating Diagrams

- **Mermaid** (text-based diagrams in markdown)
- **Excalidraw** (hand-drawn style diagrams)
- **Figma** (professional design tool)
- **Draw.io / diagrams.net** (free diagramming)
- **Python + Matplotlib** (data visualization)

### Diagram Best Practices

✅ **Do**:
- Use clear, large fonts (readable from distance)
- Include legends and labels
- Use consistent colors (e.g., red = cost, green = savings)
- Keep it simple (one concept per diagram)
- Add visual hierarchy (bigger = more important)

❌ **Don't**:
- Overcrowd with too much information
- Use unclear abbreviations
- Mix multiple concepts in one diagram
- Use colors that don't show up on projectors
- Forget to add titles and context

---

## 📝 Diagram Usage Guidelines

### In Markdown Files

Reference images using relative paths:

```markdown
![Alt Text](../assets/context-windows.png)
```

### In Presentations

- Place diagrams on slides with minimal text
- Explain verbally while showing the diagram
- Use animations to reveal parts progressively
- Allow time for participants to study the diagram

### In Hands-On Labs

- Include diagrams in exercise instructions
- Reference specific parts: "As shown in the diagram above..."
- Use diagrams to confirm understanding
- Ask students to sketch their own versions

---

## 🔄 Updating Visual Aids

If you update or replace these diagrams:

1. **Keep the filename** for consistency with references
2. **Update this README** with new descriptions
3. **Test all references** in workshop materials
4. **Archive old versions** (optional, for reference)

### Version History

- **v1.0** (Initial): Basic context windows and stateless diagrams
- Future versions will be documented here

---

## 📚 Additional Resources

### For Understanding Context Windows
- [OpenAI Documentation: Managing Tokens](https://platform.openai.com/docs/guides/text-generation/managing-tokens)
- [Anthropic: Context Window Optimization](https://docs.anthropic.com/claude/docs/context-window-optimization)

### For Token Counting
- [tiktoken library](https://github.com/openai/tiktoken)
- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)

### For Visual Learning
- Research shows visual aids improve retention by 65%
- Diagrams help bridge abstract concepts to concrete understanding
- Multiple representations (text + visual) cater to different learning styles

---

## 💡 Tips for Instructors

**When presenting diagrams**:
1. Introduce the concept verbally first
2. Show the diagram
3. Walk through each component
4. Connect back to real-world scenarios
5. Ask for questions before moving on

**Common questions about context windows**:
- "Why not just make windows infinite?" → Cost and processing time
- "Can I control what's in the window?" → Yes! This is the whole workshop
- "Do all models have the same window?" → No, show the comparison table

**Common questions about statelessness**:
- "Why don't they just remember?" → Technical architecture choice
- "Isn't this inefficient?" → Yes, that's why token management matters
- "Can future models remember?" → Possible, but current ones don't

---

Remember: These visual aids are tools to enhance understanding. The real learning happens when students connect the concepts to their own AI usage patterns!
