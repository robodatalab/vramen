# roost

A local inference server that keeps several models resident side by side under a fixed memory quota.

Roost is a library, not a daemon. You import it, declare a memory budget, and
name the models you want to call. It holds each one in its own process for as
long as the budget allows, hands it to callers a lease at a time, and evicts the
least recently used idle model when something else needs the room.

```python
from roost import CausalModel, EncoderModel, InferenceModelResourceManager
from roost import qwen_chat_prompt

manager = InferenceModelResourceManager(quota_gb=24.0)

writer = CausalModel(
    "Qwen/Qwen3-8B", qwen_chat_prompt, manager, mem_required_gb=17.0
)
embedder = EncoderModel(
    "Qwen/Qwen3-Embedding-0.6B", manager, mem_required_gb=2.0
)

answer = writer.complete("You are terse.", "Name three seabirds.", max_new_tokens=64)
vectors = embedder.encode(["a passage of a story", "another passage"])
```

Both models load on first use and stay loaded. Nothing above starts a server,
opens a port, or writes a config file.

## Install

```bash
uv add roost
```

```bash
pip install roost
```

Requires Python 3.12 or newer. Pulls in `torch`, `transformers`, `accelerate`
and `tqdm`. On macOS the `torch` wheel from PyPI is the one you want; on Linux
with CUDA, install `torch` yourself from the index that matches your driver
before adding roost, and see [Platforms](#platforms).

To try it without adding it to a project:

```bash
uv run --with roost python
```

## What it does

**One process per model.** A model is loaded inside a child process and answers
requests over a pair of queues. When it is evicted the process exits, which is
the only way to be certain the weights and the allocator's arenas are actually
gone. A model that crashes takes its own process down and raises
`ModelNotAvailable` to the caller, not the host.

**A quota spent in declarations.** You tell each model how much room it needs
via `mem_required_gb`, and the manager admits models while the declared sizes
fit inside `quota_gb`. It does not measure a model and then decide. Measurement
happens after the fact and is reported back for you to look at, never counted
against the budget. This is what makes admission predictable: the same set of
models always fits or always does not, regardless of what the allocator happened
to do last time.

**Leases, not timers.** A call holds a lease on its model for its duration. A
model under lease cannot be evicted, so a swap waits for the generation in
flight rather than killing it. Idle models are the only eviction candidates, and
they go oldest first.

**Any `transformers` checkpoint.** Causal, encoder and seq2seq models sit under
the same quota. `CausalModel.complete` generates, `EncoderModel.encode` returns
unit length pooled vectors, `Seq2SeqModel.complete` runs an encoder-decoder edit.
They are ordinary `AutoModel*` loads, so anything on the Hub that `transformers`
can open works, including embedding models whose pooled hidden states no chat
API would give you.

## What it is not

- **Not a router.** It does not choose a model for you or fall back between
  providers. The caller names the model. If you want request routing across
  providers, that is LiteLLM or OpenRouter, and they compose with this fine.
- **Not an inference engine.** No custom kernels, no paged attention, no
  continuous batching, no quantization. It calls `model.generate`. vLLM, SGLang,
  llama.cpp and MLX are engines; roost is the thing that decides which of them
  gets to be in memory, and today it only drives `transformers`.
- **Not an HTTP server.** No port, no OpenAI-compatible endpoint. If you need
  one, put your own FastAPI in front of it. It is called a server because it
  serves models to a process, not because it serves requests to a network.
- **Not multi-tenant.** Replies carry no request ids, so one caller is in flight
  per model at a time. It suits a desktop application or a batch job with a
  handful of models, not a fleet fielding concurrent traffic.
- **Not a cluster scheduler.** One machine, one quota, no replicas, no
  autoscaling, no Kubernetes.

## How it compares

Almost everything in this space is a daemon you talk to over a socket. Roost is
the same idea shrunk to an import.

| Project | What it is | How it differs |
| --- | --- | --- |
| [Ollama](https://ollama.com) | GGUF daemon with an HTTP API | Residency is `OLLAMA_MAX_LOADED_MODELS` plus a per-model `keep_alive` idle timer, so you cap a model *count* and a staleness window rather than a memory budget. Chat and embedding endpoints only. |
| [LM Studio](https://lmstudio.ai/docs/app/api/ttl-and-auto-evict) | Desktop app and server | JIT loading with an idle TTL and auto-evict, default 60 minutes. GUI first, closed source, timer driven. |
| [Lemonade Server](https://lemonade-server.ai/docs/guide/configuration/multi-model/) | Multi-model server | The closest eviction policy of the servers: keeps several models loaded and evicts LRU at the limit. Still a server, still model-count shaped. |
| [LocalAI](https://localai.io/advanced/vram-management/) | OpenAI-compatible multi-backend server | Has VRAM management and idle unload across backends. A whole platform where this is a module. |
| [mlx-serve](https://github.com/raspoli/mlx-serve) | Apple silicon MLX server | Hot-swaps MLX models with auto-unload on inactivity. MLX rather than torch, and over HTTP. |
| [vLLM](https://docs.vllm.ai) / [TGI](https://huggingface.co/docs/text-generation-inference) | Throughput engines | One model per server instance, tuned for concurrent traffic. The opposite problem: many requests against one model, not many models against one machine. |
| [Triton](https://github.com/triton-inference-server/server) | Model repository daemon | `EXPLICIT` model control mode makes load and unload your job through an API. Roost decides for you, from the budget. |
| [Ray Serve](https://docs.ray.io/en/latest/serve/model-multiplexing.html) | Distributed serving framework | `@serve.multiplexed` is the nearest relative: LRU eviction of models within a replica. Bounded by `max_num_models_per_replica`, a count again, and it wants a Ray cluster. |

The recurring difference is the budget. These tools bound residency by how many
models may be loaded, or by how long an idle one may linger. Roost bounds it by
gigabytes, which is the thing that actually runs out, and it declines a model
that cannot fit instead of discovering the problem during a load.

The second difference is the boundary. If your application is already Python and
already holds the manuscript, the index and the request, a daemon on localhost
means serializing your data out and back for every call. An import does not.

## Using it

### Sizing the quota

`mem_required_gb` is your estimate of a model's resident size, and the quota is
what you are willing to spend in total. Leave the machine some room.

```python
from roost import machine_memory

manager = InferenceModelResourceManager(quota_gb=machine_memory() * 0.6)
```

A model whose declaration exceeds the whole quota is refused immediately with
`ModelNotAvailable` rather than being loaded and killed.

### Generating

```python
from roost import CausalModel, ModelNotAvailable, qwen_chat_prompt

model = CausalModel("Qwen/Qwen3-8B", qwen_chat_prompt, manager, mem_required_gb=17.0)

try:
    text = model.complete("You are an editor.", "Tighten this line.", max_new_tokens=256)
except ModelNotAvailable as failure:
    ...
```

The third argument to `CausalModel` is a prompt formatter, `(system, user) -> str`.
`qwen_chat_prompt` renders Qwen's chat template with reasoning turned off;
`coedit_prompt` renders the instruction form CoEdIT models expect. Any callable
of that shape works.

### Embedding

```python
from roost import EncoderModel

encoder = EncoderModel("Qwen/Qwen3-Embedding-0.6B", manager, mem_required_gb=2.0)
vectors = encoder.encode(["first passage", "second passage"])
```

Passages are encoded as they stand, in batches, pooled from the last real token
of each row and normalized to unit length, so a dot product is a cosine.

### Editing

```python
from roost import Seq2SeqModel, coedit_prompt

editor = Seq2SeqModel("grammarly/coedit-large", coedit_prompt, manager, mem_required_gb=3.0)
fixed = editor.complete("Fix grammar", "she dont know", max_new_tokens=64)
```

### Watching it

```python
manager.residents          # the models loaded right now
manager.memory()           # gpu_used, gpu_limit, process, as the children last reported
manager.shutdown()         # stop every serving process, hand the quota back
```

`memory()` reads what the serving processes volunteered on their own queue, so
asking never queues behind a generation that runs for minutes.

### Logging

```python
from roost import log

log.setup()
```

Loads, evictions, tokens per second and memory readings go to the root logger at
`INFO`.

### Adding a model kind

Subclass `ModelKind` and implement `load`. Whatever you return is handed to the
requests you send through `residency`.

```python
from roost.resource_manager import ModelKind

class VisionModel(ModelKind):
    def load(self):
        return AutoModelForVision2Seq.from_pretrained(self.model_id), AutoProcessor.from_pretrained(self.model_id)
```

The instance is pickled into the child process, so keep its attributes
picklable. The manager reference is dropped on the way across.

## Platforms

**macOS on Apple silicon** is what this is built for and tested on. Models load
onto MPS, and the memory readings come from `torch.mps`.

**Linux** is not supported out of the box today. The three model classes pass
`device_map="mps"` to `from_pretrained`, so CUDA needs that changed. Everything
else is portable: the process, queue and quota machinery is plain
`multiprocessing`, and `process_memory` already handles the Linux units for
`ru_maxrss`. Off MPS the GPU figures report `0.0`, which does not affect
admission, since the quota is spent in declarations rather than measurements.

**Windows** does not work. `roost.utils` imports `resource`, which is POSIX
only, so the package fails at import. WSL is the path there.

## Developing

```bash
git clone https://github.com/robodatalab/roost
cd roost
uv sync
uv run python -m unittest discover -s tests -t . -v
```

`uv sync` installs roost into `.venv` as an editable install, so `import roost`
resolves to `src/roost`.

## Releasing

Publishing runs on [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API token is stored in the repository. One-time setup on PyPI, under the
project's *Publishing* settings:

| Field | Value |
| --- | --- |
| Owner | `robodatalab` |
| Repository | `roost` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

After that, cutting a GitHub release runs
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which builds
the sdist and wheel with `uv build` and uploads them with `uv publish`. The job
requests a short-lived OIDC token from GitHub, PyPI verifies the claims against
the configuration above and mints a scoped upload credential for that run only.

To release, bump `version` in `pyproject.toml`, commit, then tag and publish a
release on GitHub. To check the artifacts first without uploading:

```bash
uv build
```

## License

[Apache 2.0](LICENSE). Use it freely, including commercially. Keep the copyright
notice and the [NOTICE](NOTICE) file, and say so if you modify a file.
