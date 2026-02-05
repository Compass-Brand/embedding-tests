# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project: Embedding Tests

**Description:** RAG-focused embedding and reranker testing framework - benchmarking 10 models (6 text embeddings, 2 multimodal embeddings, 2 multimodal rerankers) for quality, performance, and precision impact on a Tesla P40.

**Project Type:** testing

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Testing | pytest |
| ML | torch, transformers (>=4.57.0), sentence-transformers |
| Quantization | bitsandbytes |
| RAG | chromadb, langchain-text-splitters |
| Evaluation | mteb, ragas |
| CLI | typer, rich |
| Config | pydantic, pyyaml |

---

## Key Architecture Decisions

| Decision | Rationale |
|---|---|
| Protocols (not ABC) for model interfaces | Pythonic duck typing, easy mocking |
| YAML configs for models + experiments | Human-readable, composable |
| One model loaded at a time | P40 can't hold two 8B models |
| FP16 storage + FP32 compute on P40 | P40's FP16 compute is 64x slower |
| `eager` attention on P40 | FA2 needs SM 80+, SDPA may have issues on SM 6.1 |
| ChromaDB in RAM | 47GB system RAM available, keep VRAM for models |
| bitsandbytes for quantization | Works dynamically, no pre-quantized variants needed |

## P40 Loading Patterns

```python
# sentence-transformers (text embeddings)
model = SentenceTransformer(model_id, trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "eager"})

# transformers (VL models)
model = AutoModel.from_pretrained(model_id,
    torch_dtype=torch.float16, attn_implementation="eager",
    trust_remote_code=True)

# INT8 quantized
config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
model = AutoModel.from_pretrained(model_id, quantization_config=config, device_map="auto")
```

---

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run unit tests (no GPU)
pytest -m "not gpu"

# Run GPU integration tests
pytest -m gpu

# Run with coverage
pytest tests/unit/ --cov=embedding_tests --cov-report=term-missing

# CLI
emb-test list                                    # List models
emb-test run configs/experiments/quick_sanity.yaml  # Run experiment
emb-test report results/                          # Generate reports
```

---

## Standards & Guidelines

This project follows Compass Brand standards:

- **Rules:** Inherited from parent [compass-brand/.claude/rules/](https://github.com/Compass-Brand/compass-brand/tree/main/.claude/rules)
- **Coverage:** 80%+ overall, 100% on config/hardware/metrics

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
