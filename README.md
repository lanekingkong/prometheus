# Prometheus — Universal AI Context Operating System

> **Solve AI Context Debt. Once and for all.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange)](#)

**Prometheus** is the first universal AI Context Operating System (ContextOS) — a production-grade platform that brings **structured context governance, composable skill ecosystems, persistent multi-modal memory, knowledge graph intelligence, and multi-agent orchestration** to any AI system.

---

## Why Prometheus?

### The Problem: AI Context Debt

In 2026, 57% of enterprises run AI agents in production, yet **95% of GenAI pilots deliver zero measurable P&L impact** (MIT NANDA Research, 300+ deployments). The root cause? **Context debt** — the gap between what agents infer and what the business actually means.

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| Inconsistent Answers | Same question, different results | No governed context layer |
| Authoritative Hallucination | Wrong answers delivered confidently | Smarter models amplify bad context |
| Dev-Pass / Prod-Break | Tests pass, production fails | Context differs between environments |
| Cannot Scale | Works for 1 use case, fails for 2 | Context not composable |
| Adoption Stalls | Nobody trusts the outputs | No traceability or governance |

> **"Context Engineering has displaced Prompt Engineering as the critical discipline for teams working with coding agents."** — Mike Mason, ThoughtWorks (Jan 2026)

### The Solution: ContextOS

Prometheus provides a complete **Context Operating System** that:

1. **Governs** context creation, validation, versioning, and injection
2. **Orchestrates** skills as composable, shareable Markdown-driven modules
3. **Persists** knowledge across sessions via graph + vector dual-engine memory
4. **Compresses** tokens intelligently (60-95% reduction) while preserving semantics
5. **Coordinates** multi-agent workflows with shared context visibility

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Prometheus ContextOS                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  Skill   │  │ Context  │  │  Memory  │  │  Knowledge  │ │
│  │ Ecosystem│  │ Governor │  │  Layer   │  │   Graph     │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘ │
│       │              │             │               │         │
│  ┌────┴──────────────┴─────────────┴───────────────┴──────┐  │
│  │              Multi-Agent Orchestrator                   │  │
│  └────────────────────────┬───────────────────────────────┘  │
│  ┌────────────────────────┴───────────────────────────────┐  │
│  │              Token Compression Engine                   │  │
│  └────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Adapters: OpenAI | Anthropic | Ollama | DeepSeek | Custom   │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Install
pip install prometheus-contextos

# Initialize a new context workspace
ptm init my-project

# Create a skill
ptm skill create "code-reviewer" --description "Expert code review with best practices"

# Register knowledge
ptm context add ./docs/ --recursive

# Query with context-aware intelligence
ptm ask "What are the security vulnerabilities in this codebase?"

# Run multi-agent workflow
ptm orchestrate --plan "audit_and_fix.yaml"
```

---

## Core Capabilities

### 1. Skill Ecosystem
Markdown-driven composable skills. Create, share, and compose AI capabilities like LEGO blocks.

```markdown
---
name: security-auditor
version: 1.0.0
requires: ["code-reader", "vulnerability-db"]
---

# Security Auditor Skill
You are an expert security auditor. When reviewing code:
1. Check OWASP Top 10 vulnerabilities
2. Validate input sanitization
3. Review authentication/authorization flows
```

### 2. Context Governance
Structured context with version control, validation, and traceable provenance.

### 3. Memory Layer
Dual-engine (vector + graph) persistent memory. Remember across sessions, reason across concepts.

### 4. Knowledge Graph
Entity-relationship extraction and reasoning. Connect dots across your entire knowledge base.

### 5. Multi-Agent Orchestration
Coordinate specialized agents with shared context. No more isolated agent silos.

### 6. Token Compression
60-95% token reduction while preserving semantic meaning. Save costs, maintain quality.

---

## Inspired By

Prometheus synthesizes the best ideas from the most innovative open-source projects of 2026:

| Project | Stars | What We Learned |
|---|---|---|
| OpenClaw | 302K+ | Skill system architecture, AgentSkills protocol |
| n8n | 191K+ | Visual workflow automation patterns |
| Haystack | 20K+ | Context-engineered pipeline design |
| Cognee | 16K+ | Dual-engine memory (vector + graph) |
| Dify | 132K+ | Production AI app platform UX |
| CrewAI | 12K+ | Minimalist multi-agent coordination |
| codegraph | 48K+ | Code knowledge graph construction |
| headroom | 25K+ | Token compression techniques |
| mattpocock/skills | 126K+ | Composable skill modules |

---

## Documentation

- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [Skill Creation Guide](docs/SKILLS.md)
- [Context Governance](docs/CONTEXT.md)
- [Multi-Agent Patterns](docs/AGENTS.md)
- [API Reference](docs/API.md)

---

## Community

- GitHub Issues: Bug reports & feature requests
- Discussions: Ideas, Q&A, show & tell
- Contributing: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## License

Apache 2.0 — Free for personal and commercial use.

---

**Prometheus stole fire from the gods. We steal context from chaos.**
