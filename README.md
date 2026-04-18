<div align="center">

```
██████╗   █████╗  ███████╗ ██████╗
██╔══██╗ ██╔══██╗ ╚══███╔╝ ╚═██╔═╝
██║  ██║ ███████║   ███╔╝    ██║  
██║  ██║ ██╔══██║  ███╔╝     ██║  
██████╔╝ ██║  ██║ ███████╗ ██████╗
╚═════╝  ╚═╝  ╚═╝ ╚══════╝ ╚═════╝
```

### **Agent = Model + Harness.**

DAZI is a harness engineering framework for AI coding agents.
System prompts, guardrails, feedback loops, verification, persistent corrections —
all in one terminal REPL. Works with any OpenAI-compatible model.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-managed-6E9F18?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Version](https://img.shields.io/badge/version-0.3.1-00ADD8)](https://github.com/YiSays/DAZI)

</div>

---

![DAZI Demo](docs/demo.gif)

```bash
git clone https://github.com/YiSays/DAZI.git
cd dazi
uv sync
uv run dazi            # Interactive setup wizard on first launch
```

*Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Works with OpenAI, DeepSeek, OpenRouter, Ollama, or any OpenAI-compatible API.*

---

## The 90-Second Fix

*Stop pasting stack traces into browser tabs.*

```
It's 11 PM. Your build is failing. You don't know why.

❯ /plan
❯ The CI pipeline is failing on the auth tests. Explore the test suite
  and the auth module, then write me a plan to fix it.

  DAZI reads 23 files across your codebase...
  DAZI writes a plan with 4 steps.

❯ /go

  Step 1: Fix the expired token mock in test_auth.py
  Step 2: Update the middleware to handle None refresh tokens
  Step 3: Add edge case test for concurrent session expiry
  Step 4: Run the test suite — all 47 tests pass

❯ /commit

  fix(auth): handle expired refresh tokens in middleware

  2 files changed, 18 insertions(+), 3 deletions(-)

Done. In 90 seconds. Without opening a browser.
```

Plan mode explores, read-only. Execute mode implements. Skills automate the repetitive parts.

---

## Why DAZI?

*Because your terminal is where the code lives. Your AI should live there too.*

| | Browser Chat | IDE Copilot | **DAZI** |
|---|---|---|---|
| Reads your full codebase | Copy-paste fragments | Open files only | Auto-discovers project |
| Plans before coding | No | No | Plan Mode (read-only) |
| Runs commands | No | Separate terminal | Built-in, parallel |
| Multi-agent teams | No | No | Coordinate 3, 5, 10 agents |
| Works offline | No | Varies | Yes, with local models |
| Model choice | Locked in | Locked in | Any OpenAI-compatible API |
| Knows your spend | No | No | Per-session cost tracking |
| Remembers across sessions | No | Partial | Persistent memory store |
| Built on harness engineering | No | Partial | Full control stack |

*DAZI isn't another chatbot. It's a harness engineering framework — the control layer that makes AI agents reliable.*

---

## Built on Harness Engineering

*Prompt engineering tells the model what to do. Harness engineering makes sure it actually does it.*

The model is just one component. The harness is everything else: the guardrails that prevent destructive actions, the verification loops that catch mistakes, the persistent memory that carries corrections forward, the instruction files that encode team conventions.

| Harness Layer | What It Does | DAZI Feature |
|---|---|---|
| **Feedforward guides** | Steer the agent before it acts | DAZI.md, skills, system prompts |
| **Computational controls** | Fast, deterministic checks | Permissions, hooks, plan mode |
| **Feedback sensors** | Observe after the agent acts | POST_TOOL_USE hooks, `/review` |
| **Steering corrections** | Permanently eliminate failure modes | Memory (`/remember`), DAZI.md updates |
| **Verification-first** | Explore read-only, then execute | `/plan` → `/go` workflow |
| **Generator/Evaluator split** | Separate creating from reviewing | Multi-agent teams, `/review` skill |

The mindset shift: when the agent makes a mistake, don't just fix the output. Improve the harness so that specific failure can't happen the same way again. Add a rule to DAZI.md. Create a hook. Save a memory. **The harness gets stronger every session.**

Read the full [Harness Engineering with Dazi](docs/harness-engineering.md) deep-dive — a module-by-module breakdown of how every Dazi component maps to the formal harness engineering taxonomy.

---

## How It Works

Every user input follows the same pipeline — from keystroke to response, through the full harness stack:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              DAZI REPL LOOP                               │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──▶ User Input ──▶ /command? ──yes──▶ handle_command() ──▶ continue     │
│  │                       │                                                │
│  │                       no                                               │
│  │                       ▼                                                │
│  │                    HumanMessage                                        │
│  │                       ▼                                                │
│  │   ┌────────────── run_graph_turn() ────────────────────────────────┐   │
│  │ ┌─▶                                                                │ │ │
│  │ │ │  ┌──────────────── System Prompt Builder ───────────────────┐  │ │ │
│  │ │ │  │ STATIC (cached):             DYNAMIC (per-turn):         │  │ │ │
│  │ │ │  │ ├─ Agent identity            ├─ Session guidance (mode)  │  │ │ │
│  │ │ │  │ ├─ Task instructions         ├─ Skills                   │  │ │ │
│  │ │ │  │ ├─ Tool descriptions         ├─ Relevant memories        │  │ │ │
│  │ │ │  │ ├─ Tone & style              ├─ Environment info         │  │ │ │
│  │ │ │  │ └─ DAZI.md rules             └─ Permission rules         │  │ │ │
│  │ │ │  └────────────────┬─────────────────────────────────────────┘  │ │ │
│  │ │ │                   ▼                                            │ │ │
│  │ │ │  ┌─────────── LangGraph Pipeline ───────────────────────────┐  │ │ │
│  │ │ │  │                                                          │  │ │ │
│  │ │ │  │  check_compact ──▶ call_llm ◀─────────────────────────┐  │  │ │ │
│  │ │ │  │                       │                               │  │  │ │ │
│  │ │ │  │         ┌─────────────┴──────────────┐                │  │  │ │ │
│  │ │ │  │         ▼ (no tools)                 ▼ (tools)        │  │  │ │ │
│  │ │ │  │  Streamed Response           check_permissions        │  │  │ │ │
│  │ │ │  │  (Rich Live display)               │                  │  │  │ │ │
│  │ │ │  │         │                          ├─ PRE_TOOL_USE    │  │  │ │ │
│  │ │ │  │         │                          ▼                  │  │  │ │ │
│  │ │ │  │         │                    execute_tools            │  │  │ │ │
│  │ │ │  │         │                          ├─ SAFE  ──▶ parallel │  │ │ │
│  │ │ │  │         │                          ├─ WRITE ──▶ serial   │  │ │ │
│  │ │ │  │         │                          └─ POST_TOOL_USE      │  │ │ │
│  │ │ │  │         │                          ▼                  │  │  │ │ │
│  │ │ │  │         │                    ToolMessages ────────────┘  │  │ │ │
│  │ │ │  │         │                    (loop if tool calls)        │  │ │ │
│  │ │ │  │         └─────────────┬──────────────┘                   │  │ │ │
│  │ │ │  │                       ▼                                  │  │ │ │
│  │ │ │  │                 Auto-compact                             │  │ │ │
│  │ │ │  │                 (if context > threshold)                 │  │ │ │
│  │ │ │  └───────────────────────┬──────────────────────────────────┘  │ │ │
│  │ │ └──────────────────────────┼─────────────────────────────────────┘ │ │
│  │ │                            ▼                                       │ │
│  │ │              State Update (messages, mode)                         │ │
│  │ │                            │                                       │ │
│  │ │                            ▼                                       │ │
│  │ └── inject tick ◀── yes ── Proactive Tick?                           │ │
│  │                              │                                       │ │
│  │                              no                                      │ │
│  │                              ▼                                       │ │
│  │    ┌────────────────────── Prompt ──────────────────────────┐        │ │
│  │    │ Status Bar: mode · tokens · team · bg                  │◀───────┘ │
│  │    │ Prompt:     team-name ❯                                │          │
│  │    └─────────────────────────┬──────────────────────────────┘          │
│  │                              │                                         │
│  └────── User types next input ─┘                                         │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│ Side Channels:                                                            │
│ ├─ Background Tasks ── completion_event ──▶ interrupt prompt ──▶ show     │
│ └─ Multi-Agent Teams ── autonomous scan/claim/execute cycle               │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Think Before You Code

*No more "oops, I edited the wrong file."*

```
❯ /plan
  PLAN MODE — read-only tools only

❯ Explore the payment module and write a plan for adding Stripe integration

  DAZI reads 14 files... writes plan.md

❯ /show

  ## Plan: Stripe Payment Integration
  1. Add Stripe SDK dependency
  2. Create PaymentService class in services/payment.py
  3. Add /api/payments endpoint with validation
  4. Write integration tests

❯ /go
  EXECUTE MODE — implementing the plan
```

In plan mode, write tools are physically stripped from the tool list. The LLM literally cannot modify your files. Read, explore, plan — then switch to execute when you're ready. This is **verification-first**: the agent explores before it modifies, and you review the plan before any changes land.

## Build with a Team of Agents

*One problem. Multiple minds. Zero coordination overhead.*

```
❯ Create a team "refactor" with 3 agents:
    - "researcher" explores the codebase
    - "writer" makes the changes
    - "tester" runs tests

  Team "refactor" created. 3 agents ready.

❯ Create tasks:
    1. Map all uses of the deprecated API          (pending)
    2. Refactor to the new API surface              (pending, blocked by #1)
    3. Run full test suite and fix failures          (pending, blocked by #2)

  researcher claimed task #1...
  writer waiting (blocked by #1)...
  researcher finished. writer claimed task #2...
```

Agents share a task board with dependency tracking. They claim work, execute, and report back. You're the team lead — assign, review, coordinate. Or let them self-organize.

## One Command, Zero Repetition

*Turn your best prompts into slash commands.*

**Built-in skills:**

| Command | What it does |
|---------|--------------|
| `/commit` | Analyze staged changes → conventional commit message |
| `/review` | Find bugs, security issues, performance problems |
| `/explain` | Deep-dive into code, architecture, or concepts |
| `/summarize` | TL;DR of anything in under 200 words |

**Create your own:**

```markdown
<!-- .dazi/skills/deploy.md -->
---
name: deploy
description: Deploy to staging
arguments:
  - environment
---

Deploy the current branch to $environment:
1. Run tests with `npm test`
2. Build with `npm run build`
3. Deploy using `deploy --env $environment`
```

Then just type `/deploy staging`. Skills are discovered from bundled, user-level (`~/.dazi/skills/`), and project-level (`.dazi/skills/`) locations.

## Don't Wait. Keep Coding.

*Start a build. Keep talking. Check when you're ready.*

```
❯ Run "npm run build" in the background

  Task bg-001 started — you can keep working

❯ Now add rate limiting to the API endpoints while that builds

  ...DAZI edits files, runs tests, you keep working...

❯ /bg bg-001

  bg-001: completed (exit code 0, 12s)
  Output: ✓ 142 modules bundled in 8.4s
```

## It Remembers So You Don't Have To

*"Use uv, not pip." Say it once. DAZI remembers forever.*

```
❯ /remember This project uses Python 3.12+ with uv for package management

  Memory saved.

  (In a future session, weeks later:)

❯ How should I add a dependency?

  Use `uv add <package>` — you prefer uv over pip (from memory).
```

Four memory types cover everything: your role and preferences (`user`), guidance on how to work (`feedback`), project decisions and deadlines (`project`), and pointers to external resources (`reference`).

This is the **steering loop**: when the agent makes a mistake, save the correction as feedback memory. Next session, it loads automatically. The harness gets stronger every time you use it.

## Know What You Spend

*No surprise API bills.*

```
❯ /cost

Session Cost
┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Model     ┃ Requests ┃ In Tokens ┃ Out Tokens┃ Cost (USD)  ┃
┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ gpt-4o    │ 12       │ 45,230    │ 8,421     │ $0.1827     │
│ gpt-4o-m… │ 3        │ 12,100    │ 2,300     │ $0.0085     │
├───────────┼──────────┴──────────┴───────────┴─────────────┤
│ Total     │ 15       │ 57,330    │ 10,721    │ $0.1912     │
└───────────┴──────────┴───────────┴──────────┴─────────────┘
```

Per-session cost breakdown by model. Check current spend with `/cost`, previous session with `/cost last`.

## Connect Everything

*Filesystems, databases, GitHub, browsers — if there's an MCP server, DAZI talks to it.*

```json
{
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["@anthropic/mcp-fs"] },
    "github": { "command": "npx", "args": ["@anthropic/mcp-github"] },
    "database": { "command": "python", "args": ["mcp_db.py"] }
  }
}
```

MCP tools appear alongside built-in tools, namespaced as `mcp__<server>__<tool>`. Connect, disconnect, and inspect with `/mcp`.

## Conversations Without Limits

*Talk all day. DAZI handles the context window so you don't have to.*

When the conversation approaches the context limit, DAZI automatically compresses old messages — keeping recent context intact. Two strategies: **micro-compact** (fast, replaces old tool results with markers) and **full compact** (LLM-powered summary that preserves intent).

```
❯ /tokens
  Tokens: 98,421 / 128,000 (77%) — 29,579 remaining

❯ /compact
  Compressed 45 messages → 12-message summary
```

## You're in Control

*Approve, auto-approve, or deny — your rules, your codebase.*

Every tool call goes through a permission check. You decide what's allowed:

```bash
❯ /allow git push:*       # Allow all git push variants
❯ /deny rm -rf *          # Never allow recursive delete
❯ /allow npm:*            # Allow all npm commands
```

Four permission modes: `default` (ask before destructive), `plan` (read-only), `acceptEdits` (auto-approve file edits), `bypass` (allow everything). Deny always wins over allow. These aren't just preferences — they're **computational controls** in your harness. Fast, deterministic, no LLM needed.

---

## Works With Your Model

*Bring your own API key. We don't pick sides.*

| Provider | Example Models |
|----------|---------------|
| OpenAI | GPT-4o, GPT-4o-mini, o1, o3-mini |
| DeepSeek | DeepSeek-V3, DeepSeek-R1 |
| OpenRouter | Any model through one endpoint |
| Ollama | Local models, fully offline |
| Custom | Any OpenAI-compatible endpoint |

Set your API key, model, and base URL during the interactive setup wizard. Change anytime with `/onboard`.

---

## Configuration

Three layers, higher priority wins:

```
DEFAULT (hardcoded) → USER (~/.dazi/settings.json) → PROJECT (.dazi/settings.json)
```

**User-level** (`~/.dazi/settings.json`) — applies to all projects:
```json
{
  "model": "gpt-4o",
  "api_base_url": "https://api.openai.com/v1",
  "max_concurrent_tools": 5,
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["@anthropic/mcp-fs"] }
  }
}
```

**Project-level** (`.dazi/settings.json`) — shared with your team:
```json
{
  "allow_rules": ["git status", "npm test"],
  "deny_rules": ["rm -rf *"],
  "env": { "NODE_ENV": "test" }
}
```

Primitives: higher layer wins. Lists: concatenate. Dicts: shallow merge. Reload without restart with `/reload`.

---

## Command Reference

| Command | Description |
|---------|-------------|
| `/plan` | Enter plan mode (read-only) |
| `/go` | Exit plan mode, resume execution |
| `/show` | Display the plan file |
| `/tools` | List available tools |
| `/tasks` | Show task board |
| `/task <id>` | Get task details |
| `/skills` | List skills |
| `/skill <name>` | Show skill details |
| `/<skill>` | Invoke a user-invocable skill |
| `/mcp` | Show MCP servers |
| `/cost` | Show session cost |
| `/cost last` | Show previous session cost |
| `/settings` | Show settings with sources |
| `/rules` | List permission rules |
| `/allow <pattern>` | Add allow rule |
| `/deny <pattern>` | Add deny rule |
| `/hooks` | List registered hooks |
| `/compact` | Compress conversation context |
| `/tokens` | Show token usage |
| `/remember <text>` | Save a memory |
| `/forget <id>` | Delete a memory |
| `/memories` | List memories |
| `/reindex` | Rebuild memory index |
| `/dazimd` | Show loaded DAZI.md files |
| `/bg` | List background tasks |
| `/bg <id>` | Check background task |
| `/teams` | List teams |
| `/team <name>` | Switch to team |
| `/team create <name>` | Create team |
| `/team delete <name>` | Delete team |
| `/inbox` | Check messages |
| `/send <agent> <msg>` | Send message |
| `/broadcast <msg>` | Message all |
| `/proactive` | Show proactive status |
| `/proactive on` | Enable proactive mode |
| `/proactive off` | Disable proactive mode |
| `/worktree` | List worktrees |
| `/worktree create <name>` | Create worktree |
| `/worktree finish <name>` | Finish worktree |
| `/reload` | Reload settings, skills, and MCP servers |
| `/clear` | Reset conversation |
| `/quit` | Exit DAZI |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for the LLM provider |
| `OPENAI_MODEL` | Default model name (overridden by settings) |
| `OPENAI_BASE_URL` | Custom API endpoint (overridden by settings) |
| `DAZI_PROACTIVE` | Set to `1` to enable proactive mode at startup |

---

## Roadmap: Towards a Full Harness Engineering Infrastructure

DAZI already covers the core harness layers — feedforward guides, computational controls, feedback sensors, and the steering loop. Here's what's coming next to make it a complete harness engineering infrastructure:

### Automated Steering Loop
**The gap:** Today you manually notice recurring failures and save corrections. The agent doesn't track which mistakes repeat.
**The vision:** DAZI detects when the same failure pattern recurs across sessions and automatically suggests harness improvements — a new DAZI.md rule, a permission update, or a hook addition. *From "you fix it" to "the harness fixes itself."*

### Built-in Verification Hooks
**The gap:** Hooks exist but there are no pre-built verification hooks for common workflows (lint after edit, test after commit, type-check after refactor).
**The vision:** A library of ready-made verification hooks: `post-edit-lint`, `post-write-test`, `pre-commit-typecheck`. One line in settings.json to activate. Computational controls that work out of the box.

### AI-as-Judge Evaluator Pattern
**The gap:** The `/review` skill is a single agent reviewing its own work. Research shows models can't reliably evaluate their own output.
**The vision:** A dedicated evaluator agent — separate from the generator — that runs semantic code review after every edit cycle. Not just lint (computational), but judgment on correctness, security, and intent (inferential). The generator/evaluator split from GAN research, applied to code.

### Harness Observability
**The gap:** You can't see which controls catch which failures, or how your harness improves over time.
**The vision:** A dashboard showing which permission rules fire most, which hooks prevent the most issues, which memory corrections get loaded most often. Measure the harness, not just the agent.

### Technology-Aware Feedforward
**The gap:** DAZI.md is static — you write it once and it applies the same guidance regardless of which file or language you're editing.
**The vision:** Auto-detect project technology stack (React, Django, Go microservices) and inject relevant conventions, patterns, and anti-patterns automatically. The feedforward layer adapts to what you're actually building.

### Entropy Management
**The gap:** Long autonomous sessions accumulate cruft — unused imports, commented-out code, inconsistent naming.
**The vision:** A background agent that periodically refactors for code hygiene. Not waiting for you to notice — the harness keeps the codebase clean autonomously.

---

## Contributing

DAZI is open source and built in the open. PRs welcome. File issues. Star the repo if it saves your evening.

License: MIT
