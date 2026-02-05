# embedding-tests

RAG-focused embedding and reranker testing framework for evaluating 10 models across multiple precision levels on a Tesla P40 (24GB VRAM).

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run unit tests (no GPU needed)
pytest -m "not gpu"

# List available models
emb-test list

# Run quick sanity check (requires GPU)
emb-test run configs/experiments/quick_sanity.yaml
```

## Model Inventory

### Text Embeddings (sentence-transformers)

| Model | Params | Dim | FP16 Fits P40? |
|---|---|---|---|
| Qwen3-Embedding-0.6B | 0.6B | 1024 | YES |
| Qwen3-Embedding-4B | 4B | 2560 | YES |
| Qwen3-Embedding-8B | 8B | 4096 | YES |
| llama-embed-nemotron-8b | 7.5B | 4096 | YES |
| Octen-Embedding-8B | 7.6B | 4096 | YES |
| KaLM-Embedding-Gemma3-12B | 11.76B | 3840 | NO (INT8/INT4) |

### Multimodal Embeddings (transformers)

| Model | Params | Dim |
|---|---|---|
| Qwen3-VL-Embedding-2B | 2B | 2048 |
| Qwen3-VL-Embedding-8B | 8B | 4096 |

### Multimodal Rerankers (transformers)

| Model | Params |
|---|---|
| Qwen3-VL-Reranker-2B | 2B |
| Qwen3-VL-Reranker-8B | 8B |

## Experiment Presets

| Preset | Description |
|---|---|
| `quick_sanity.yaml` | Qwen3-Embedding-0.6B only, fast check |
| `text_embeddings.yaml` | All 6 text models at FP16 + INT8 |
| `multimodal.yaml` | VL embedding models |
| `rerankers.yaml` | Reranker comparison |
| `precision_comparison.yaml` | FP16 vs INT8 vs INT4 for 8B models |
| `rag_chunking.yaml` | Chunking strategy comparison |
| `full_benchmark.yaml` | All models, all precisions |

## Architecture

```
src/embedding_tests/
  config/       - Model configs, hardware detection, experiment YAML
  models/       - ST wrapper, VL embedding/reranker wrappers, loader
  hardware/     - Precision strategy, VRAM estimation
  pipeline/     - Chunking, embedding, retrieval, reranking, RAG
  evaluation/   - Metrics, MTEB integration, RAG evaluator
  runner/       - Experiment orchestrator, checkpoints, CLI
  reporting/    - Results collection, comparison, export
```

## P40 Constraints

- **No BF16** (CC 6.1): Forces `torch_dtype=torch.float16`
- **No Flash Attention 2**: Uses `attn_implementation="eager"`
- **FP16 compute is 1:64 vs FP32**: Store FP16, compute FP32
- **24GB VRAM**: 12B model requires INT8/INT4 quantization
