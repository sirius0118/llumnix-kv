# Llumnix-KV

Llumnix-KV is a general, flexible, and high-performance KV cache transfer and storage framework for distributed LLM inference consisting of two core components: **Hybrid Connector** and **Blade-KVT**.

See also the [llumnix](https://github.com/llumnix-project/llumnix) repository for how to use Llumnix-KV in an end-to-end distributed serving deployment.

## vLLM 0.21 migration adapter

This branch includes `patches/vllm-v0.21.0.patch` for the standalone
`HybridConnector` migration backend. Apply it to the vLLM source before building:

```bash
git apply --check /path/to/llumnix-kv/patches/vllm-v0.21.0.patch
git apply /path/to/llumnix-kv/patches/vllm-v0.21.0.patch
```

The validation target is `0.21.0+zeta.all`, source commit
`f65c123b10ccb3db216a8dab02742f5ba0a67b0e`; this is not a claim of compatibility
with every 0.21 build. Validation uses Qwen3-30B-A3B, matching TP8 layouts,
GDR, synchronous decoding and disabled prefix caching, with eager execution
and CUDA Graph decode. Seeded sampling transfers each worker's request RNG;
unseeded sampling, speculative decoding, PP/DP/EP migration and transactional
rollback are outside this adapter's validated scope. Set
`LLX_KVT_NIC_AFFINITY` to a comma-separated NIC name per local GPU to override
automatic NIC selection.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              vLLM Engine (Python)                   │
│                                                     │
│  ┌─────────────┐    ┌──────────────────────────┐    │
│  │  Scheduler  │    │   HybridConnector        │    │
│  └─────────────┘    │  ┌────────────────────┐  │    │
│                     │  │   KVT ( C++ )      │  │    │
│                     │  │  ┌──────────────┐  │  │    │
│                     │  │  │ ParseBlock   │  │  │    │
│                     │  │  └──────────────┘  │  │    │
│                     │  └────────────────────┘  │    │
│                     └──────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## Components

### HybridConnector

A unified KV cache control plane that acts as the "driver layer" between LLM engines and KV Cache transfer mechanisms.
The **"hybrid"**, the key of its name, suggests that Hybrid Connector **unifies multiple transfer paths in one KV connector**.

**Design Principles:**
- **Zero Intrusion**: Engine remains unaware of KV Cache transfer details
- **Zero Overhead**: No dummy steps or polling mechanisms
- **Minimal Interface**: Only `start_load_kv` and `save_kv_layer`
- **Fully Asynchronous**: All transfer logic runs in independent threads

**Key Features:**
- Request lifecycle management with reference counting decoupling
- Support for multiple backends (PD separation, KVStore, migration)
- Complete fault tolerance (abort, retry, timeout handling)

### Blade-KVT (KV Transfer)

A high-performance, zero-overhead KV Cache transfer module that handles the actual data transmission between nodes.

**Key Features:**
- Bypass design for sidecar-style integration
- Full CUDA Graph compatibility via CUDA Event notifications
- General and flexible support for multiple cache layouts / attention backends
  - FlashAttention, FlashInfer
  - Hybrid attention: full / linear (GDN) / sparse (DSA)
- Support for multiple transport backends
  - GPU Direct RDMA (GDR) - lowest latency
  - TCP - isolated from RDMA traffic
  - Shared Memory - single-node multi-GPU

## Supported Scenarios

| Scenario | Description |
|----------|-------------|
| **PD Separation** | P node handles Prefill, D node handles Decode with KV Cache P→D transfer |
| **KVStore Persistence** | Async save/load between GPU memory and shared storage |
| **Request Migration** | Online migration with minimal service interruption |
| **Multi-Backend** | Run multiple backends simultaneously |

## Documentation

- [HybridConnector Details](doc/HybridConnector-README.md)
- [KVT Details](doc/KVT-README.md)
