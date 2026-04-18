# Harness Engineering with Dazi

## Agent = Model + Harness

The model is powerful but undisciplined. It can write code, explain architecture, debug failures — but left to its own devices, it will also edit the wrong files, import the wrong libraries, forget your conventions, and silently accumulate technical debt.

**Harness engineering** is the discipline of designing the control systems that wrap around an AI agent to make it reliable. The model is the engine. The harness is everything else: the guardrails, the feedback loops, the verification checks, the persistent memory of past corrections.

The term was coined by [Mitchell Hashimoto](https://mitchellh.com/writing/my-ai-adoption-journey) in February 2026, and formalized into a taxonomy by [Martin Fowler and Birgitta Boeckeler at Thoughtworks](https://martinfowler.com/articles/harness-engineering.html) in April 2026. OpenAI validated the concept at production scale by shipping a product with over one million lines of code — zero manually written — using Codex within a rigorous harness ([OpenAI, "Harness engineering: leveraging Codex in an agent-first world"](https://openai.com/index/harness-engineering/)).

The equestrian metaphor is deliberate: the horse has power, but without reins, saddle, and bridle, it goes wherever it pleases. The harness channels that power productively.

**DAZI** is a harness engineering framework for AI coding agents. It implements the full taxonomy — guides, sensors, computational controls, inferential controls, verification, and the steering loop — in a single terminal REPL. This document explains each concept and shows how Dazi's components bring it to life.

---

## The Harness Engineering Taxonomy

Martin Fowler's formalization identifies four core functions that a harness must provide:

### Guides (Feedforward Controls)

**Steer the agent before it acts.**

Guides are instructions, conventions, and context that shape the agent's behavior before it generates any output. They are the most common form of harness engineering — and the most neglected. Most teams rely on the model's training data and hope for the best. A well-engineered harness encodes team conventions, architecture decisions, and domain knowledge into persistent instruction files that load into every session.

Examples: system prompts, AGENTS.md / CLAUDE.md / DAZI.md files, skill templates, coding standards.

### Sensors (Feedback Controls)

**Observe after the agent acts, and help it self-correct.**

Sensors run after the agent takes an action — writing a file, running a command, editing code — and evaluate whether the action was correct. They provide the data that feeds the steering loop. Without sensors, you have no visibility into what the agent is doing wrong.

Examples: post-edit hooks that run linters, test runners, AI code review, cost trackers.

### Verification

**Validate before execution.**

Verification gates sit between the agent's intent and the actual execution. They check whether a proposed action is safe, authorized, and correct before it happens. This is different from sensors (which observe after) — verification prevents damage.

Examples: permission systems, plan-only modes, pre-commit checks, dependency validation.

### Correction (The Steering Loop)

**Permanently eliminate recurring failure modes.**

The steering loop is the most important concept in harness engineering. When the agent makes a mistake — and it will — you don't just fix the output. You improve the harness so that specific failure *cannot happen the same way again*. As Hashimoto put it:

> "It is the idea that anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again."

This is what separates harness engineering from prompt engineering. Prompt engineering tries to get the prompt right once. Harness engineering iteratively strengthens the system every time it fails. The harness gets stronger every session.

### Two Execution Types

Every guide, sensor, and verification check falls into one of two categories:

**Computational** — Deterministic, fast, run by the CPU. Linters, type checkers, permission rules, file existence checks. Milliseconds. Results are reliable and reproducible.

**Inferential** — Semantic, probabilistic, run by the LLM. AI code review, quality assessment, "LLM as judge." Slower and non-deterministic, but can evaluate things that pure computation cannot — intent, correctness, maintainability.

A well-engineered harness uses both: fast computational checks catch obvious errors instantly, while inferential checks catch subtle problems that require understanding.

---

## Dazi as a Harness Engineering Framework

DAZI maps directly onto the harness engineering taxonomy. Here is how each component corresponds to a harness function.

### Guides / Feedforward Controls

These modules steer the agent before it takes any action.

| Dazi Module | Harness Function | How It Works |
|---|---|---|
| `prompt_builder.py` | System prompt assembly | Builds structured system prompts with static/dynamic caching. Different guidance for plan mode vs execute mode. API-level prompt caching optimization with boundary markers. |
| `dazimd.py` | Project-level instructions | Loads `DAZI.md` files hierarchically: project root → user global. Supports `@include` directives for composing from multiple files. Encodes team conventions and architecture decisions. |
| `skills.py` | Reusable prompt templates | Markdown files with YAML frontmatter. Bundled skills (`/commit`, `/review`, `/explain`, `/summarize`) provide best-practice workflows. Custom skills at user or project level. Argument substitution for parameterized workflows. |
| `settings.py` | Three-layer configuration | `DEFAULT → USER (~/.dazi/settings.json) → PROJECT (.dazi/settings.json)`. Primitives: higher wins. Lists: concatenate. Dicts: shallow merge. Consistent behavior across sessions. |

**What this means in practice:** When you write `Always use httpx, not requests` in `DAZI.md`, that instruction loads into every session. The agent sees it before it writes a single line of code. That's a feedforward guide — steering before acting.

### Sensors / Feedback Controls

These modules observe the agent's actions and provide correction signals.

| Dazi Module | Harness Function | How It Works |
|---|---|---|
| `hooks.py` | Event-driven sensors | Fires on `PRE_TOOL_USE`, `POST_TOOL_USE`, `POST_TOOL_USE_FAILURE`, `USER_PROMPT_SUBMIT`, `SESSION_START`. Can modify tool inputs/outputs, override permission decisions, or block execution entirely. Priority-ordered, chainable. |
| `memory.py` | Persistent feedback store | Four memory types: `user` (role, preferences), `feedback` (behavioral guidance), `project` (decisions, deadlines), `reference` (external resources). Stored as individual `.md` files with YAML frontmatter. Relevance scoring with keyword search. Auto-injected into system prompt when relevant. |
| `cost_tracker.py` | Resource monitoring | Tracks token usage and estimated cost per model. Per-session and cross-session persistence. Observable via `/cost` command. |

**What this means in practice:** When you add a `POST_TOOL_USE` hook that runs `ruff check` after every file edit, that's a sensor — it observes the agent's output and provides immediate feedback. When you save `/remember never hardcode Path.home() in constructors`, that's a persistent correction that loads automatically next session.

### Computational Controls

These modules provide fast, deterministic checks — no LLM calls, no ambiguity.

| Dazi Module | Harness Function | How It Works |
|---|---|---|
| `permissions.py` | Pre-execution guardrails | Rule-based access control with exact, prefix, wildcard, and glob pattern matching. Four permission modes: `default`, `plan`, `acceptEdits`, `bypass`. Deny always wins over allow. Deterministic — no LLM in the decision path. |
| `compact.py` | Context window management | Two strategies: micro-compact (clear old tool results, no LLM call) and full compact (LLM summarizes old messages). Auto-triggers on threshold. Circuit breaker after 3 consecutive failures. Prevents token overflow — a computational guardrail. |
| `registry.py` | Mode-filtered tool lists | Maintains separate tool lists for plan mode and execute mode. Plan mode physically strips write and destructive tools. The LLM cannot call tools that aren't in the list — not a suggestion, not a preference, a structural constraint. |
| `background.py` | Non-blocking execution | Asyncio subprocess management for long-running commands. Prevents agent blocking on builds, tests, or deployments. Graceful shutdown (SIGTERM → SIGKILL). Output capture to files. |
| `lifecycle.py` | Clean startup/shutdown | Initializes all subsystems in correct order. Registers cleanup handlers for graceful exit. Prevents resource leaks and inconsistent state. |
| `task_store.py` | Dependency tracking | File-based task storage with bidirectional dependency tracking (`blocks` / `blocked_by`). Tasks stored as individual JSON files — corruption isolation. Ensures correct execution order for multi-step workflows. |
| `resilience.py` | Failure handling | Retry logic, circuit breakers, and error recovery patterns for tool execution failures. |

**What this means in practice:** When you run `/plan`, the tool list is filtered at the `registry.py` level. Write tools are gone. Not hidden — gone. The LLM receives a tool list that doesn't include them. That's a computational control: deterministic, structural, impossible to bypass.

### Generator / Evaluator Split

Research shows models cannot reliably evaluate their own output (Anthropic, "Building Effective Agents," Dec 2024). The solution: separate the agent that generates code from the agent that evaluates it.

| Dazi Module | Harness Function | How It Works |
|---|---|---|
| `team.py` + `coordinator.py` | Multi-agent teams | Team creation with shared task boards. Autonomous teammates self-organize: scan → claim → execute → report. File-based inter-agent messaging via `mailbox.py`. Role specialization — different agents for different concerns. |
| `repl_teams.py` | Team REPL commands | `/team create`, `/inbox`, `/send`, `/broadcast` for real-time coordination. |
| `graph.py` | Plan → Execute pipeline | Structured workflow: exploration (generator, read-only) → plan review (human-in-the-loop) → implementation (executor). The plan step is the generator; the review step is the evaluator. |
| `skills.py` (`/review`) | AI-as-judge | Built-in code review skill that evaluates code for bugs, security, performance, and readability. An inferential evaluator that runs after the generator produces output. |

**What this means in practice:** When you use `/plan` to explore, then `/go` to execute, then `/review` to evaluate — you are manually orchestrating the generator/evaluator split. When you create a team with a "researcher" agent and a "reviewer" agent working on the same codebase, you are architecturally separating generation from evaluation.

### The Steering Loop

The steering loop is the meta-pattern that ties everything together. Every other component exists to feed information into this loop.

| Dazi Component | How It Strengthens the Harness |
|---|---|
| `/remember <text>` | Save a correction as feedback memory. It loads automatically in future sessions. |
| `/allow <pattern>` / `/deny <pattern>` | Add a permission rule. It applies to every tool call in every future session. |
| `DAZI.md` edits | Update project instructions. Every agent session loads them. |
| Hook creation | Add a runtime check that fires on every relevant tool call. |
| `/plan` → `/go` review | Catch mistakes in the planning phase before they become code changes. |

---

## A Full Session: Building a To-Do App

*Abstract modules and tables are useful, but nothing teaches like a real session. Here's what it looks like to build a small feature — a to-do app — using Dazi's harness from start to finish. Every phase exercises a different harness layer.*

### Phase 1: Feedforward Setup — Steer Before You Start

It's 2 PM. Your PM just dropped a feature request: a to-do app with categories, due dates, and a REST API. You haven't written a line of code yet. You start by strengthening the harness.

```
❯ /dazimd
  ./DAZI.md (project)

❯ Write project conventions into `DAZI.md` that it is python project, and it use `uv` to ma
nage project and `uv run pytest` for unit test.

  Written to .dazi/DAZI.md:
  "Use FastAPI + aiosqlite. Follow async/await everywhere.
   No sync database calls. All endpoints return JSON."

❯ /deny rm -rf *
  Deny rule added: rm -rf *

❯ /allow pytest:*
  Allow rule added: pytest:*

❯ /allow uv run:*
  Allow rule added: uv run:*
```

**What happened:** Before the agent writes a single file, three harness layers are already working:
- **Feedforward guide** — `DAZI.md` loads into every future session's system prompt. The agent will see "Use FastAPI + aiosqlite" before it generates any code.
- **Computational controls** — `/deny rm -rf *` and `/allow` rules are deterministic. No LLM involved. If the agent tries `rm -rf build/`, it gets blocked. Period.

This is the harness engineering mindset: *invest 60 seconds in setup, save 30 minutes of corrections.*

### Phase 2: Plan Mode — Explore Before You Touch

Now you tell the agent what to build. But first, you flip into plan mode.

```
❯ /plan
  PLAN MODE — read-only tools only
  Write tools physically removed from tool list.

❯ Build a to-do app with TUI: Rich frontend and sqlite3 backend, 
  CRUD on tasks (title, category, due_date, completed).

  DAZI reads 8 files across the project...
  Existing structure: main.py, requirements.txt, tests/
  No existing database code or models found.

  Plan written to .dazi/plan.md:

  ## Plan: To-Do App
  1 uv init + uv add rich + uv add --dev pytest
  2 Create todo.py (~150-180 lines)
  3 Create tests/test_todo.py
  4 Verify with uv run pytest and manual run
```

```
❯ /show
  (reviews the plan — looks good, but...)

❯ Add input validation to the plan: title required, category from
  a fixed enum, due_date must be ISO 8601.

  Plan updated. Step 2 now includes validation rules.
```

**What happened:** In plan mode, write tools don't exist in the LLM's tool list. Not hidden — *removed* at the registry level. The agent can read files, search code, and write to the plan file, but it physically cannot edit your source code. This is **verification-first**: explore, plan, get human approval, then execute.

The `/show` command and the interactive revision loop is the **human-in-the-loop checkpoint** — the generator (agent) produces a plan, you (the evaluator) review it, and only then does execution proceed.

### Phase 3: Execute with Task Tracking

You're happy with the plan. Time to execute.

```
❯ /go
  EXECUTE MODE — all tools enabled
  Plan loaded from .dazi/plan.md
```

The agent now creates tasks with dependencies, so work proceeds in the right order:

```
❯ Break the plan into tasks with dependencies

  │ task_create({'subject': 'Initialize project and install dependencies', 'description':
'Run `uv init` (if not already done), then `uv...)
  └ Task created: #1
  │ task_create({'activeForm': 'Implementing database layer (TodoDB class)', 'description':
'Create `todo.py` with the `TodoDB` class:\n...)
  └ Task created: #2
  │ task_create({'activeForm': 'Implementing TUI layer with Rich', 'description': 'Add TUI
functions to `todo.py` using Rich:\n- `show_m...)
  └ Task created: #3
  │ task_create({'activeForm': 'Implementing main loop and entry point', 'description':
'Add `main()` function to `todo.py`:\n- Infinite...)
  └ Task created: #4
  │ task_create({'activeForm': 'Writing unit tests for TodoDB', 'description': 'Create
`tests/test_todo.py` with pytest unit tests for `...)
  └ Task created: #5
  │ task_create({'activeForm': 'Running end-to-end verification', 'description': 'Final
verification:\n1. `uv sync` — ensure dependencie...)
  └ Task created: #6
  │ task_update({'taskId': '2', 'addBlockedBy': ['1']})
  │ task_update({'taskId': '3', 'addBlockedBy': ['2']})
  │ task_update({'taskId': '4', 'addBlockedBy': ['3']})
  │ task_update({'taskId': '5', 'addBlockedBy': ['2']})
  │ task_update({'taskId': '6', 'addBlockedBy': ['4', '5']})
```

The agent claims Task #1 and starts working. Task #2 through #5 sit idle — blocked by upstream dependencies. The task store (`task_store.py`) enforces ordering. No skipping ahead, no working on the API before the data model exists. This is a **computational control**: deterministic dependency resolution, zero LLM involvement.

```
❯ /tasks

┏━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ ID ┃ Status  ┃ Subject                                     ┃ Owner ┃ Blocked By ┃
┡━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│  1 │ pending │ Initialize project and install dependencies │ -     │ -          │
│  2 │ pending │ Implement database layer (TodoDB class)     │ -     │ 1          │
│  3 │ pending │ Implement TUI layer with Rich               │ -     │ 2          │
│  4 │ pending │ Implement main loop and entry point         │ -     │ 3          │
│  5 │ pending │ Write unit tests for TodoDB                 │ -     │ 2          │
│  6 │ pending │ Run end-to-end verification                 │ -     │ 4, 5       │
└────┴─────────┴─────────────────────────────────────────────┴───────┴────────────┘
```

### Phase 4: Background Execution — Don't Wait, Keep Coding

Task #1 is done. The agent moves to #2, then #3. After finishing the database layer, you want to run the existing test suite to make sure nothing broke — but you don't want to sit idle while tests run.

```
❯ Run the test suite in the background while you work on Task #4

  Background task bg-001 started: uv run pytest tests/ -v
  You can keep working.

  (Agent continues implementing Task #4: REST endpoints...)

❯ /bg bg-001
  bg-001: completed (exit code 0, 8s)
  12 passed, 0 failed
```

The agent never stopped. Tests ran in a subprocess managed by `background.py` — async, non-blocking, with output captured to a file. When the test finished, the result was waiting. This is a **computational control**: the harness prevents the agent from blocking on long-running operations. You (and the agent) keep moving.

### Phase 5: Error Handling — When Things Go Wrong

The agent finishes Task #4 and starts on Task #5 (tests). Then it makes a mistake.

```
  Agent: I'll create a test that directly calls the database...

  ❯ /deny rm -rf dist/*
  ⛔ DENIED — matches rule: rm -rf *
```

Wait — that wasn't the agent's mistake. That was the permission system catching a different problem. Let's see what the agent actually did wrong:

```
  Agent: Writing test_database.py...
  (Uses sqlite3.connect() — synchronous!)

  ❯ /review
  Reviewing 3 files...

  ⚠ test_database.py:14 — uses sqlite3.connect() (sync). This blocks
  the event loop. Use aiosqlite for async test fixtures.
  ⚠ main.py:22 — missing input validation on category field
```

Two problems caught. The first violates the `DAZI.md` rule ("Use aiosqlite. No sync database calls"). The second is a missing validation that you specified in the plan revision.

You fix the harness:

```
❯ /remember Always use aiosqlite for database operations, even in tests.
  Never use sqlite3.connect() — it blocks the event loop.

  Memory saved. (feedback type)
```

**What happened:** Three harness layers caught problems:
- **Feedforward guide** — `DAZI.md` said "no sync database calls" but the agent missed it. This is why sensors exist — guides aren't foolproof.
- **Inferential sensor** — `/review` (AI-as-judge) caught both the sync call and the missing validation. It understood *semantics*, not just syntax.
- **Steering correction** — `/remember` saved the lesson permanently. Next session, this memory auto-injects into the system prompt. The agent won't make this mistake again.

The harness caught the error, you corrected it, and the harness got stronger.

### Phase 6: Multi-Agent Team — Generator/Evaluator Split

Task #5 is done, but the PM just asked for two more features: user authentication and task search. Rather than doing it sequentially, you spin up a team.

```
❯ /team create todo-dev
  Team "todo-dev" created.

  Spawn 3 agents:
    - "architect" — plans the implementation (read-only, generator)
    - "builder"   — writes the code (executor)
    - "reviewer"  — reviews every change (evaluator)
```

```
❯ /tasks
  #6  Design auth flow (JWT tokens)               [pending]
  #7  Implement auth middleware                    [pending, blocked by #6]
  #8  Add /tasks/search endpoint with filters     [pending]
  #9  Write auth + search tests                   [pending, blocked by #7, #8]

❯ Architect claims #6, builder waits...
❯ Architect finishes. Builder claims #7...
❯ Builder working on #7. Reviewer runs /review on each file...

  /review main.py:
  ✓ JWT validation looks correct
  ⚠ Token expiry should be configurable, not hardcoded

  /review search.py:
  ✓ Query parameterization prevents SQL injection
  ✓ Pagination implemented correctly
```

This is the **generator/evaluator split** in action. The architect generates the plan (generator). The builder implements it (generator). The reviewer evaluates the output (evaluator). Three separate agents, three separate inference passes. Research shows models can't reliably evaluate their own output — so Dazi doesn't make them try.

The task store coordinates the dependency chain: #9 waits for both #7 and #8. Builder can work on #7 and #8 in parallel (no dependency between them). When both finish, #9 unblocks automatically.

```
❯ /tasks
  #6  Design auth flow                            [completed]  architect
  #7  Implement auth middleware                   [completed]  builder
  #8  Add search endpoint                         [completed]  builder
  #9  Write tests                                 [in_progress] builder
```

### Phase 7: The Steering Loop Closes

All tasks are done. Time to wrap up.

```
❯ /cost

  Session Cost
  ┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┓
  ┃ Model     ┃ Requests ┃ In Tokens ┃ Out Tokens┃ Cost (USD)  ┃
  ┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━┩
  │ gpt-4o    │ 23       │ 89,412    │ 14,203    │ $0.3641     │
  │ gpt-4o-m… │ 4        │ 18,300    │ 3,100     │ $0.0143     │
  └───────────┴──────────┴───────────┴───────────┴─────────────┘

❯ /review
  Reviewing all changes in this session...

  ✓ No security issues found
  ✓ No performance concerns
  ⚠ main.py:45 — consider extracting magic number 3600 to config
  ✓ Test coverage: 94%

❯ /commit

  feat(todo): add to-do app with auth, search, and full test coverage

  8 files changed, 347 insertions(+), 12 deletions(-)
```

Done. In under 30 minutes. With a team of three agents, background tests, and the harness catching two bugs before they hit production.

### Session 2: The Harness Remembers

*A week later, you start a new session for a different feature.*

```
❯ uv run dazi
  Loading memories...
  1 memory loaded: "Always use aiosqlite for database operations"

❯ Add a new endpoint that exports tasks to CSV

  Agent: Creating export endpoint... using aiosqlite to fetch tasks...
  (Memory auto-injected — agent uses async database calls without being told)
```

The agent didn't make the same mistake. Not because you prompted it better this time — but because you improved the **harness** last time. The correction persisted across sessions. The harness got stronger.

> **This is the fundamental difference.** Prompt engineering: "hope the agent does better next time." Harness engineering: "make sure it *can't* make that specific mistake again."

### Harness Layers at a Glance

| Phase | What Happened | Harness Layer | Dazi Module |
|---|---|---|---|
| 1. Setup | DAZI.md + permissions before coding | Feedforward guide + Computational control | `dazimd.py`, `permissions.py` |
| 2. Plan | Read-only exploration, plan review | Verification | `registry.py`, `graph.py` |
| 3. Execute | Tasks with dependency chains | Computational control | `task_store.py` |
| 4. Background | Tests run while coding continues | Computational control | `background.py` |
| 5. Errors | `/review` catches bugs, `/remember` saves corrections | Inferential sensor + Steering loop | `hooks.py`, `memory.py` |
| 6. Team | Architect plans, builder codes, reviewer evaluates | Generator/evaluator split | `team.py`, `coordinator.py` |
| 7. Wrap-up | Cost check, final review, commit | Feedback sensor | `cost_tracker.py`, `skills.py` |
| Next session | Memory auto-loads, mistake prevented | Steering loop | `memory.py`, `prompt_builder.py` |

---

## The Harness Architecture

```
DAZI Harness Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────────┐
  │                 FEEDFORWARD GUIDES                   │
  │                                                      │
  │   DAZI.md          Skills          System Prompt      │
  │   (dazimd.py)      (skills.py)     (prompt_builder)  │
  │                                                      │
  │   "Steer the agent before it acts"                   │
  └────────────────────────┬────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │              AGENT EXECUTION LOOP                     │
  │            (graph.py — LangGraph)                     │
  │                                                      │
  │   Input → [Compact Check] → LLM → [Permission       │
  │          Check] → Tools → Output                     │
  └────────────────────────┬────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │             COMPUTATIONAL CONTROLS                    │
  │                                                      │
  │   Permissions      Tool Registry    Compact           │
  │   (permissions.py) (registry.py)    (compact.py)      │
  │   Background       Lifecycle        Task Store       │
  │   (background.py)  (lifecycle.py)   (task_store.py)  │
  │                                                      │
  │   "Fast, deterministic, no LLM needed"               │
  └────────────────────────┬────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │              FEEDBACK SENSORS                        │
  │                                                      │
  │   Hooks            Memory           Cost Tracker     │
  │   (hooks.py)       (memory.py)      (cost_tracker)   │
  │                                                      │
  │   "Observe after the agent acts"                     │
  └────────────────────────┬────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │              STEERING CORRECTIONS                    │
  │                                                      │
  │   /remember        /allow /deny      DAZI.md        │
  │   Hooks            /plan → /go       /review         │
  │                                                      │
  │   "The harness gets stronger every session"          │
  └─────────────────────────────────────────────────────┘
```

---

## What's Next: The Roadmap to Full Harness Maturity

DAZI covers the core harness engineering taxonomy today — guides, sensors, computational controls, verification, and the steering loop. But harness engineering is an evolving discipline, and there are capabilities that push it further.

| Harness Capability | Current Status | What's Missing |
|---|---|---|
| **Feedforward guides** | Solid — DAZI.md, skills, system prompts | Technology-aware auto-injection (detect project stack, inject relevant conventions) |
| **Computational controls** | Strong — permissions, registry, compact, lifecycle | Pre-built verification hook library (lint-after-edit, test-after-write, typecheck-after-refactor) |
| **Feedback sensors** | Good — hooks, memory, cost tracker | Automated failure pattern detection (track which mistakes recur across sessions) |
| **Steering loop** | Manual — user notices and saves corrections | Auto-suggest harness improvements (detect recurring failure → propose DAZI.md rule, permission, or hook) |
| **Inferential controls** | Basic — `/review` skill uses same model | Dedicated evaluator agent with separate inference (LLM-as-judge with different model/config) |
| **Generator/evaluator split** | Partial — plan→execute, multi-agent teams | Architectural separation of generator and evaluator with independent context windows |
| **Harness observability** | Minimal — cost tracking only | Dashboard showing which controls fire, which catch failures, harness health over time |
| **Entropy management** | None | Background agent that periodically refactors for code hygiene (unused imports, inconsistent naming, stale patterns) |

The trajectory is clear: from **manual** harness engineering (you notice, you fix, you save) to **automated** harness engineering (the system detects, suggests, and improves itself). DAZI provides the infrastructure today to make that evolution possible.

---

## Why This Matters

The progression of AI-assisted development has followed three phases:

1. **Prompt engineering** (2022–2024) — *What do I say to the model?* Single-turn instructions. Fragile — reordering examples can shift accuracy by 40%+. No persistence.

2. **Context engineering** (2025) — *What does the model see?* RAG, memory, conversation history, tool definitions. Better, but still focuses on the input. A bigger context window doesn't make a flaky agent reliable.

3. **Harness engineering** (2026+) — *How does the whole system run?* The full environment: tools, guardrails, feedback loops, verification, lifecycle. The burden shifts from "wait for a better model" to "build a better harness."

DAZI is built for phase 3. It doesn't try to be a better model — it provides a better harness. The model is pluggable (any OpenAI-compatible API). The harness is the product.

> "Every time the agent makes a mistake, don't just hope it does better next time. Engineer the environment so it can't make that specific mistake the same way again."
>
> — Mitchell Hashimoto

---

## References

- Mitchell Hashimoto, ["My AI Adoption Journey"](https://mitchellh.com/writing/my-ai-adoption-journey), February 5, 2026 — Step 5: "Engineer the Harness." Coined the term and defined the mindset shift.
- Martin Fowler, Birgitta Boeckeler, ["Harness engineering for coding agent users"](https://martinfowler.com/articles/harness-engineering.html), April 2, 2026 — Formal taxonomy: guides, sensors, computational/inferential execution types.
- OpenAI, ["Harness engineering: leveraging Codex in an agent-first world"](https://openai.com/index/harness-engineering/), 2026 — Production validation: 1M+ lines of code, zero manually written, using Codex within a rigorous harness.
- Anthropic, ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents), December 2024 — Early groundwork: simple composable patterns, externalized memory, verification-first workflows.
