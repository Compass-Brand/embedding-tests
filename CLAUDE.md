# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project: Embedding Tests

**Description:** Embedding model evaluation and testing - benchmarking different embedding models for quality, performance, and cost.

**Project Type:** testing

---

## Tech Stack

| Layer    | Technology   |
| -------- | ------------ |
| Language | Python 3.11+ |
| Testing  | pytest       |

---

## Standards & Guidelines

This project follows Compass Brand standards:

- **Rules:** Inherited from parent [compass-brand/.claude/rules/](https://github.com/Compass-Brand/compass-brand/tree/main/.claude/rules) - coding style, security, testing, git workflow, performance, and agent delegation rules
- **Tech Stack:** See [Universal Tech Stack](https://github.com/Compass-Brand/compass-brand/blob/main/docs/technical-information/tech_stack.md)

---

## Development Methodology: TDD

All functional code MUST follow Test-Driven Development.

```text
RED -> GREEN -> REFACTOR
```

---

## Git Discipline (MANDATORY)

**Commit early, commit often.**

- Commit after completing any file creation or modification
- Maximum 15-20 minutes between commits
- Use conventional commit format: `type: description`

Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`
