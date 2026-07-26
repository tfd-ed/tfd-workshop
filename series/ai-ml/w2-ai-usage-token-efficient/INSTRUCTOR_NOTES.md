# Instructor Notes: Workshop 2 - AI Token-Efficient Usage

## 📋 Pre-Workshop Checklist

### 1 Week Before
- [ ] Test demo script on your machine (`python scripts/demo-script.py`)
- [ ] Install tiktoken: `pip install tiktoken`
- [ ] Prepare example repository for live coding
- [ ] Test AI assistant (GitHub Copilot, Cursor, etc.) is working
- [ ] Review current token pricing (prices change!)
- [ ] Set up screen recording for workshop recording

### 1 Day Before
- [ ] Review all workshop materials
- [ ] Prepare live coding examples
- [ ] Test demo script end-to-end
- [ ] Check meeting platform (Zoom, Teams, etc.)
- [ ] Send reminder email to participants
- [ ] Prepare polls/interactive questions

### 1 Hour Before
- [ ] Open all demo files and browsers tabs
- [ ] Test screen sharing and audio
- [ ] Have demo script ready
- [ ] Open token counter tool
- [ ] Have example prompts ready to copy-paste

---

## ⏱️ Detailed Timing Guide

### Introduction (5 minutes)
- Welcome and agenda overview (2 min)
- Learning objectives (1 min)
- Poll: "How much do you spend on AI tools monthly?" (2 min)

### Part 1: Understanding the Foundation (15 minutes)
- **How LLMs Work** (5 min)
  - Keep it high-level, avoid deep technical details
  - Focus on "prediction machine" concept
  - Mention context window limitations
  
- **What Are Tokens** (5 min)
  - Run Demo 1 from demo script (token counting)
  - Show surprising examples (emoji, code, non-English)
  - Poll: "Guess how many tokens in this function?"
  
- **Evolution to Agents** (3 min)
  - Visual timeline: Prompts → Chat → Agents
  - Emphasize the token multiplication
  
- **Understanding Context Windows** (3 min) **⭐ NEW SECTION**
  - Visual explanation using context-windows.png diagram
  - Show how everything counts toward the limit
  - Demonstrate context accumulation over turns
  - Run Demo 8 from demo script (Context Window Visualization)
  
- **Why Usage Exploding** (2 min)
  - Quick examples of agent tool calls
  - Show real token counts from a simple agent request
  - Connect back to context window filling up

**💡 Teaching Tip**: Use the demo script for Part 1. It's interactive and keeps engagement high.

**💡 Visual Aid**: Display `assets/context-windows.png` when explaining how context accumulates. Point to each component (system, history, files, prompt, response) and emphasize they ALL count.

**💡 Interactive Poll**: "How many of you have had AI conversations that got expensive or confused after many messages?" (Most will raise hands - this validates the context window problem)

---

### Part 2: The Myth of the "Magic Prompt" (10 minutes)
- **Common Misconceptions** (3 min)
  - Share examples of overly complex prompts you've seen
  - Discuss role-playing prompts ("Act as a senior engineer...")
  
- **What Actually Works** (4 min)
  - Live demo: Compare vague vs. specific prompt
  - Show side-by-side token counts
  - Run Demo 3 from demo script
  
- **Iteration Over Perfection** (3 min)
  - Live example: Ask AI something vague, then refine
  - Show how refining is cheaper than trying to be perfect

**💡 Teaching Tip**: Have 2-3 "bad prompt" examples from real world ready to share anonymously.

**⚠️ Common Pitfall**: Students might feel their elaborate prompts are being criticized. Frame it as "optimization" not "wrong."

---

### Part 3: Software Engineering Principles for AI (15 minutes)
- **Single Responsibility Principle** (3 min)
  - Live demo: Ask AI to "build everything" vs. one thing
  - Show the difference in complexity
  
- **Divide and Conquer** (5 min)
  - Walk through TODO API example from slides
  - Break down a complex task on the fly
  - Show the step-by-step approach
  
- **Senior-Junior Delegation** (4 min)
  - Use analogy: How would you delegate to a junior dev?
  - Show specific vs. vague instructions
  - Live example: Delegate a task to AI
  
- **Code Review Mindset** (3 min)
  - Show example of AI-generated code with issues
  - Walk through the checklist
  - Emphasize: Always review!

**💡 Teaching Tip**: Use the senior-junior analogy heavily. It resonates well.

**Interactive Element**: Ask participants to share in chat: "What would you tell a junior dev about this task?"

---

### Part 4: Token-Efficient Workflows (15 minutes)
- **Minimize Context** (3 min)
  - Run Demo 5 (context loading efficiency)
  - Show file reference vs. pasting content
  
- **Batch Changes** (2 min)
  - Quick example: 3 sequential changes vs. 1 batch
  - Calculate token savings
  
- **Request Diffs** (2 min)
  - Show: Full file regeneration vs. diff
  - Emphasize 80-90% savings potential
  
- **Real Workflow Example** (5 min)
  - Run Demo 4 (incremental vs. monolithic)
  - Walk through both approaches
  - Show cumulative token counts
  
- **Cost Calculator** (3 min)
  - Run Demo 7 (cost calculator)
  - Show monthly/annual costs
  - Calculate team savings

**💡 Teaching Tip**: The cost calculator is a great "aha!" moment. Let the numbers speak.

**Interactive Element**: "Let's calculate YOUR potential savings" - have students estimate their usage.

---

### Part 5: Agentic AI (10 minutes)
- **What Makes AI Agentic** (3 min)
  - Show the planning → tool use → execution cycle
  - Explain tool calling overhead
  
- **Token Cost of Autonomy** (3 min)
  - Run Demo 6 (agentic costs)
  - Show the step-by-step token accumulation
  
- **Using Skills Effectively** (2 min)
  - Explain framework-specific skills
  - Show examples: Django, React, Docker skills
  - When to use vs. general prompting
  
- **Controlling Agents** (2 min)
  - Show boundary-setting examples
  - Approval workflows
  - Limited scope prompts

**💡 Teaching Tip**: If possible, do a live agentic demo with token tracking enabled.

---

### Part 6: Real-World Techniques (10 minutes)
- Quick fire: Cover 3-4 techniques from the list
  - Explain Then Generate (2 min)
  - Reference Implementation (2 min)
  - Incremental Refinement (2 min)
  - Error-Driven Refinement (2 min)
  - Test-Driven (2 min if time allows)

**💡 Teaching Tip**: Pick the techniques most relevant to your audience. Skip others if time is tight.

---

### Hands-On Time (Optional, 15 minutes if workshop is 1.5h)
- Guide students through Exercise 2 or 3 from hands-on lab
- Have them share results in chat
- Discuss findings as a group

---

### Summary & Q&A (10 minutes)
- Key takeaways recap (3 min)
- Q&A (7 min)
- Share resources
- Point to hands-on lab for practice
- Mention next workshop

**💡 Teaching Tip**: Keep common Q&A answers ready (see below).

---

## 🎯 Learning Objectives Alignment

Make sure to explicitly connect content to objectives:

| Objective | Covered In | Assessment |
|-----------|------------|------------|
| Explain how tokens work | Part 1, Demo 1 & 2 | Token counting exercise |
| Design efficient prompts | Part 2 & 4 | Prompt comparison exercise |
| Divide-and-conquer | Part 3 | TODO API breakdown |
| Use agentic features | Part 5 | Agent supervision examples |
| Measure token usage | Throughout | Cost calculator demo |

---

## 💡 Teaching Tips

### For Engagement
1. **Use Polls**: Every 10-15 minutes, run a quick poll
   - "How many of you have hit token limits?"
   - "Which technique will you try first?"
   
2. **Chat Interactions**: Ask participants to share in chat
   - "What's your biggest AI token waste?"
   - "Share a time AI generated too much code"
   
3. **Live Demos**: Do at least 3-4 live AI interactions
   - Show real token counts
   - Make deliberate mistakes to show refinement
   
4. **Real Numbers**: Use actual cost calculations
   - "For your team of X, that's $Y per year"
   - Make it tangible

### For Comprehension
1. **Analogies Work**: Use the senior-junior delegation analogy heavily
2. **Show, Don't Just Tell**: Every technique should have a live demo
3. **Repeat Key Concepts**: 
   - "Tokens = Money" (say it 5+ times)
   - "Small tasks > Big tasks"
   - "Specific > Vague"

### For Retention
1. **Actionable Takeaways**: End each section with "What you can do tomorrow"
2. **Cheat Sheet**: Share a one-page summary (create from key takeaways)
3. **Practice Exercises**: Strongly encourage the hands-on lab

---

## ⚠️ Common Issues & Solutions

### Issue 1: "This seems like premature optimization"
**Response**: 
> "It's not about micro-optimizing every prompt. It's about building good habits that:
> - Save money at scale (show team cost calculation)
> - Lead to better results (focused prompts = clearer outcomes)
> - Reduce debugging time (smaller tasks = easier to fix)
> Think of it like writing clean code — not premature optimization, just good practice."

### Issue 2: "My company pays for AI, why should I care?"
**Response**:
> "Great question! Even if cost isn't your concern:
> - Efficient prompts get better results faster
> - Less token waste = faster responses (less processing)
> - Skills transfer when you do pay for API usage
> - Team-wide efficiency compounds
> Plus, companies track usage. Being efficient looks good!"

### Issue 3: "Won't future models make this unnecessary?"
**Response**:
> "Models will improve, but:
> - Better models often cost proportionally more
> - Context windows are growing, but so is our use of them
> - Efficiency principles apply regardless of model
> - These skills help you work smarter with ANY AI tool
> It's like saying 'faster computers mean we don't need efficient algorithms.' Efficiency always matters."

### Issue 4: "Agentic AI is supposed to be autonomous. Isn't supervising it defeating the purpose?"
**Response**:
> "Think of it like code review:
> - You could auto-merge all pull requests (autonomous)
> - But reviewing catches issues and maintains quality (supervised)
> Supervision doesn't defeat autonomy — it guides it.
> For complex tasks, let agents explore. For focused tasks, provide boundaries.
> The key is choosing the right level of autonomy for each situation."

### Issue 5: "I tried being specific, but AI still generated too much"
**Response**:
> "Add explicit constraints:
> - 'Max 20 lines of code'
> - 'Show only the function, not the whole file'
> - 'No tests yet, just the implementation'
> - 'Stop after generating X, wait for my feedback'
> Also, consider: Is your task actually small enough? Break it down further."

### Issue 6: "Why not just make context windows infinite?" ⭐ NEW
**Response**:
> "Great question! Several reasons:
> - **Processing Time**: More tokens = slower responses. Even with 200k windows, processing takes time
> - **Cost**: Providers charge per token. Infinite windows would cost infinite money
> - **Memory**: GPUs have physical memory limits. Each token requires computation and storage
> - **Diminishing Returns**: Research shows attention degrades over very long contexts
> - **Technical Architecture**: Current transformer models have quadratic complexity with context length
> 
> Think of it like RAM — we could theoretically have terabytes, but it's expensive and often unnecessary.
> The real skill is working efficiently within limits."

### Issue 7: "Can't the AI just remember what we discussed earlier?" ⭐ NEW
**Response**:
> "This is a common misconception! LLMs are **stateless** — each request is completely independent.
> 
> [Show the llm-no-memory.png diagram]
> 
> To create the illusion of memory:
> - The conversation history is re-sent with EVERY request
> - Turn 1: You send 100 tokens
> - Turn 2: You send 100 new + 100 old = 200 tokens
> - Turn 10: You send 100 new + 900 old = 1,000 tokens
> 
> This is why:
> - Long conversations get expensive (compounding history)
> - Starting fresh saves tokens (no history to resend)
> - Context windows fill up over time
> - Summaries help (condense 1,000 tokens into 100)
> 
> It's not a bug, it's the architecture. Understanding this is key to efficiency!"

### Issue 8: "My AI conversation got confused after many messages. Why?" ⭐ NEW
**Response**:
> "This is **context window overflow** in action!
> 
> [Show the context-windows.png diagram and point to the overflow scenario]
> 
> What happened:
> 1. Your conversation reached the context window limit (e.g., 128k tokens)
> 2. To fit new messages, the oldest ones were dropped
> 3. The AI literally 'forgot' earlier parts of the conversation
> 4. It started contradicting itself or missing important context
> 
> **Solutions**:
> - Start a fresh conversation when switching topics
> - Ask AI to summarize progress, then start new chat with summary
> - Be more concise (every message counts toward the limit)
> - Avoid loading unnecessary files into context
> - Use stateless requests for independent tasks
> 
> Think of it like a whiteboard — once it's full, you have to erase old stuff to add new stuff!"

### Issue 9: "How do I know when I'm close to the context limit?" ⭐ NEW
**Response**:
> "Great awareness question! Here are signs:
> 
> **Warning Signs**:
> - Responses getting slower
> - AI 'forgetting' things you discussed earlier
> - Contradicting previous statements
> - API errors about token limits
> - Unexpected behavior in agentic workflows
> 
> **Proactive Monitoring**:
> - Many AI tools show token usage (GitHub Copilot, Cursor)
> - Use tiktoken library to count tokens yourself
> - Track conversation length (>20 turns = likely high usage)
> - Notice when files are loaded (each file = thousands of tokens)
> 
> **Best Practice**:
> Don't wait for the limit! Start fresh when:
> - Switching topics or tasks
> - You've had >15-20 turns of conversation
> - The AI loaded many files
> - You notice degraded performance
> 
> [Run Demo 8 to show visual token accumulation]"

---

## 🎭 Interactive Elements

### Poll Questions (prepare in advance)

1. **Opening Poll**: "How much do you estimate you spend on AI tools monthly?"
   - Free (using company subscription)
   - $0-20
   - $20-50
   - $50-100
   - $100+

2. **Mid-Workshop**: "Which wastes more tokens?"
   - Loading entire codebase for small change
   - Asking AI to explain everything it does
   - Over-engineered prompts
   - All of the above

3. **Technique Poll**: "Which technique will you try this week?"
   - Divide and conquer
   - Senior-junior delegation
   - Incremental refinement
   - Batching changes

### Chat Prompts

1. "Share in chat: What's one AI interaction that used way more tokens than needed?"
2. "Type in chat: How many AI requests do you make per day?"
3. "Share: What's your biggest 'AI generated too much' story?"

### Live Coding Suggestions

**Example 1: Vague vs. Specific**
```
Vague: "Create a login function"
[Show what AI generates - probably too much]

Specific: "Create a function check_credentials(username, password) 
that returns True if username=='admin' and password=='pass123', False otherwise. 
Max 5 lines."
[Show the difference]
```

**Example 2: Incremental Building**
```
Step 1: "Create a User class with name and email attributes"
[Test it]

Step 2: "Add a validate_email method to User class"
[Test it]

Step 3: "Add __str__ method that returns 'User: {name} ({email})'"
[Show how each step builds on previous]
```

---

## 📊 Metrics to Track

During the workshop, track:
- Number of attendees
- Poll response rates
- Questions asked (log for future)
- Chat engagement level
- Lab completion rate (if doing hands-on)

After the workshop:
- Attendance vs. registration
- Post-workshop survey results
- Lab submissions (if applicable)
- Follow-up questions

---

## 🎥 Recording Tips

If recording for YouTube:

1. **Chapters**: Mark these timestamps for chapters
   - Introduction
   - Understanding Tokens
   - Prompt Efficiency
   - Software Engineering Principles
   - Token-Efficient Workflows
   - Agentic AI
   - Real-World Techniques
   - Q&A

2. **Highlight Moments**: Flag these for short clips
   - Token cost calculator demo
   - Vague vs. specific prompt comparison
   - Incremental vs. monolithic demo
   - Real cost savings calculation

3. **Editing Notes**:
   - Speed up long terminal outputs (1.5x)
   - Add captions for key takeaways
   - Include overlay text for statistics
   - Add visual markers for section transitions

---

## 📚 Additional Resources to Share

**During Workshop**:
- OpenAI Tokenizer link
- tiktoken GitHub repo
- Prompt Engineering Guide

**Post-Workshop Email**:
- Link to recording
- Hands-on lab exercises
- Cheat sheet (one-pager of techniques)
- Recommended reading list
- Next workshop announcement

---

## 🔄 Continuous Improvement

**After Each Workshop**:
1. Review questions asked → Update FAQs
2. Check poll results → Adjust content emphasis
3. Note timing issues → Adjust schedule
4. Collect feedback → Iterate materials
5. Update pricing examples → Keep current

**Keep in your notes**:
- Great audience questions (add to slides)
- Timing that ran over/under
- Demos that worked particularly well
- Explanations that clicked
- New examples from participants

---

## 🎯 Success Metrics

Workshop is successful if participants can:

1. ✅ Explain what tokens are and why they cost money
2. ✅ Identify 3+ ways to reduce token usage in their work
3. ✅ Write a token-efficient prompt vs. their old approach
4. ✅ Break down a complex task into incremental steps
5. ✅ Calculate approximate token costs for their usage

**Post-Workshop Goals**:
- 80%+ would recommend to colleagues
- 70%+ complete at least one hands-on exercise
- 50%+ implement at least one technique within a week

---

## 📝 Instructor Self-Checklist

During delivery, am I:
- [ ] Speaking at a good pace (not too fast)
- [ ] Checking chat regularly for questions
- [ ] Showing enthusiasm for the topic
- [ ] Using real examples, not just theory
- [ ] Staying on schedule
- [ ] Engaging with polls and questions
- [ ] Making it practical and actionable
- [ ] Showing empathy for common frustrations

---

## 🚨 Backup Plans

**If Demo Script Fails**:
- Use screenshot examples prepared in advance
- Show pre-calculated token counts
- Reference the content markdown for examples

**If Running Out of Time**:
- Skip Part 6 (Real-World Techniques) - least critical
- Shorten Q&A to 5 minutes
- Point to materials for self-study

**If Running Under Time**:
- Extend hands-on practice
- Do more live coding examples
- Deep dive into Q&A
- Show bonus examples

**If Technical Issues**:
- Have slides as backup (no live demos)
- Share screen of static examples
- Use chat more heavily for engagement

---

## 💬 Sample Transitions

**Between sections**:
> "So we've seen how tokens work. Now let's talk about why 'magic prompts' are a myth..."

> "Great! We've covered the principles. Now let's see how to apply them in real workflows..."

> "You might be wondering how agentic AI fits into all this. Let's dive in..."

**To demos**:
> "Rather than just talking about this, let me show you the actual difference..."

> "Let's see this in action with a live demo..."

> "I've prepared a demo that shows exactly what I mean..."

**To interactive elements**:
> "I'm curious — let's do a quick poll on this..."

> "Share in the chat: Have you experienced this?"

> "Let's make this real with your numbers..."

---

## 📧 Pre-Workshop Email Template

Subject: Workshop Tomorrow: AI Token-Efficient Usage 🚀

Hi everyone!

Looking forward to seeing you at tomorrow's workshop on AI token efficiency!

**Quick Prep** (Optional but recommended):
- Install tiktoken: `pip install tiktoken`
- Have your favorite AI assistant ready (GitHub Copilot, Cursor, etc.)
- Think about one project where you've used AI recently

**What to Expect**:
- Live demos of token-efficient techniques
- Real cost calculations
- Practical workflows you can use immediately
- Hands-on examples

**Workshop Details**:
- Date/Time: [INSERT]
- Link: [INSERT]
- Duration: 1 hour
- Recording: Will be shared afterward

See you tomorrow!

[Your Name]

---

## 📧 Post-Workshop Email Template

Subject: Workshop Resources: AI Token-Efficient Usage ✅

Hi everyone!

Thanks for attending today's workshop! Here are the promised resources:

**📹 Recording**: [Link]

**📚 Materials**:
- Workshop Content: [Link to materials/workshop-2-content.md]
- Hands-On Lab: [Link to exercises/hands-on-lab.md]
- Demo Script: [Link to scripts/demo-script.py]

**🎯 Quick Wins for This Week**:
1. Try one "specific prompt" instead of a vague one
2. Break down your next complex task into 3-5 smaller prompts
3. Use the token counter to measure your usage

**📊 Poll Results**:
[Share interesting findings from workshop polls]

**❓ Q&A Summary**:
[Top 3 questions with answers]

**🔜 Next Workshop**: [Announce if available]

**📝 Feedback**: [Survey link]

Questions? Reply to this email or join our [Discord/Slack/Forum].

Keep those tokens efficient! 💰

[Your Name]

---

**Good luck with the workshop! You've got this! 🎉**
