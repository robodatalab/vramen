# vramen

A local inference server that keeps several models resident side by side under a fixed memory quota.

Vramen is a library, not a daemon. You import it, declare a memory budget, and
name the models you want to call. It holds each one in its own process for as
long as the budget allows, hands it to callers a lease at a time, and evicts the
least recently used idle model when something else needs the room.

```python
from vramen import CausalModel, EncoderModel, InferenceModelResourceManager
from vramen import qwen_chat_prompt

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
uv add vramen
```

```bash
pip install vramen
```

Requires Python 3.12 or newer. Pulls in `torch`, `transformers`, `accelerate`
and `tqdm`. On macOS the `torch` wheel from PyPI is the one you want; on Linux
with CUDA, install `torch` yourself from the index that matches your driver
before adding vramen, and see [Platforms](#platforms).

Drawing is an extra, `uv add 'vramen[image]'`, which adds `diffusers` and
Pillow, and the `sentencepiece` and `protobuf` that the T5 encoders in these
pipelines are still shipped as. Nothing else needs it, and without it the rest
of the package works as it did.

To try it without adding it to a project:

```bash
uv run --with vramen python
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

**Any `transformers` checkpoint, and any `diffusers` pipeline.** Causal,
encoder, seq2seq and text to image models sit under the same quota.
`CausalModel.complete` generates, `EncoderModel.encode` returns unit length
pooled vectors, `Seq2SeqModel.complete` runs an encoder-decoder edit, and
`Text2ImageModel.draw` returns a PNG. The first three are ordinary `AutoModel*`
loads and the fourth is a `DiffusionPipeline`, so anything on the Hub that
either library can open works, including embedding models whose pooled hidden
states no chat API would give you, and an 8.9B denoiser that no chat API has at
all.

## What it is not

- **Not a router.** It does not choose a model for you or fall back between
  providers. The caller names the model. If you want request routing across
  providers, that is LiteLLM or OpenRouter, and they compose with this fine.
- **Not an inference engine.** No custom kernels, no paged attention, no
  continuous batching, no quantization. It calls `model.generate`. vLLM, SGLang,
  llama.cpp and MLX are engines; vramen is the thing that decides which of them
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

Almost everything in this space is a daemon you talk to over a socket. Vramen is
the same idea shrunk to an import.

| Project | What it is | How it differs |
| --- | --- | --- |
| [Ollama](https://ollama.com) | GGUF daemon with an HTTP API | Residency is `OLLAMA_MAX_LOADED_MODELS` plus a per-model `keep_alive` idle timer, so you cap a model *count* and a staleness window rather than a memory budget. Chat and embedding endpoints only. |
| [LM Studio](https://lmstudio.ai/docs/app/api/ttl-and-auto-evict) | Desktop app and server | JIT loading with an idle TTL and auto-evict, default 60 minutes. GUI first, closed source, timer driven. |
| [Lemonade Server](https://lemonade-server.ai/docs/guide/configuration/multi-model/) | Multi-model server | The closest eviction policy of the servers: keeps several models loaded and evicts LRU at the limit. Still a server, still model-count shaped. |
| [LocalAI](https://localai.io/advanced/vram-management/) | OpenAI-compatible multi-backend server | Has VRAM management and idle unload across backends. A whole platform where this is a module. |
| [mlx-serve](https://github.com/raspoli/mlx-serve) | Apple silicon MLX server | Hot-swaps MLX models with auto-unload on inactivity. MLX rather than torch, and over HTTP. |
| [vLLM](https://docs.vllm.ai) / [TGI](https://huggingface.co/docs/text-generation-inference) | Throughput engines | One model per server instance, tuned for concurrent traffic. The opposite problem: many requests against one model, not many models against one machine. |
| [Triton](https://github.com/triton-inference-server/server) | Model repository daemon | `EXPLICIT` model control mode makes load and unload your job through an API. Vramen decides for you, from the budget. |
| [Ray Serve](https://docs.ray.io/en/latest/serve/model-multiplexing.html) | Distributed serving framework | `@serve.multiplexed` is the nearest relative: LRU eviction of models within a replica. Bounded by `max_num_models_per_replica`, a count again, and it wants a Ray cluster. |

The recurring difference is the budget. These tools bound residency by how many
models may be loaded, or by how long an idle one may linger. Vramen bounds it by
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
from vramen import machine_memory

manager = InferenceModelResourceManager(quota_gb=machine_memory() * 0.6)
```

A model whose declaration exceeds the whole quota is refused immediately with
`ModelNotAvailable` rather than being loaded and killed.

### Generating

```python
from vramen import CausalModel, ModelNotAvailable, qwen_chat_prompt

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
from vramen import EncoderModel

encoder = EncoderModel("Qwen/Qwen3-Embedding-0.6B", manager, mem_required_gb=2.0)
vectors = encoder.encode(["first passage", "second passage"])
```

Passages are encoded as they stand, in batches, pooled from the last real token
of each row and normalized to unit length, so a dot product is a cosine.

### Editing

```python
from vramen import Seq2SeqModel, coedit_prompt

editor = Seq2SeqModel("grammarly/coedit-large", coedit_prompt, manager, mem_required_gb=3.0)
fixed = editor.complete("Fix grammar", "she dont know", max_new_tokens=64)
```

### Drawing

```python
from vramen import Text2ImageModel

painter = Text2ImageModel(
    "lodestones/Chroma1-HD",
    manager,
    mem_required_gb=28.0,
    negative_prompt="low quality, blurry, deformed",
)
png = painter.draw("a heron on a jetty at dawn", seed=433)
open("heron.png", "wb").write(png)
```

The picture comes back as the bytes of a PNG rather than as a PIL image. The
reply crosses a process boundary, and bytes are what a file, a socket and a
browser all take as they stand. A seed draws the same picture again from the
same prompt; without one every call draws something new. `draw` takes `width`,
`height`, `steps` and `guidance` as well, and defaults to 1024 square in 40
steps at guidance 3.0, which is what the Chroma card asks for.

`model_index.json` names the pipeline a checkpoint wants, so Chroma, Flux and
SDXL all load through this one class. Weights missing from the hub cache are
downloaded on the first load, which for Chroma is some 28GB, and that happens
inside the serving process, so whatever else is resident goes on answering
while it runs.

Pass `offload=True` to have `diffusers` keep one component on the accelerator at
a time and the rest in RAM. On a discrete card that is the difference between
fitting and not; on Apple silicon, where both are the same memory, it mostly
costs time. `mem_required_gb` is your declaration either way — the quota is
spent on what you say a model needs, not on what it turns out to hold.

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
from vramen import log

log.setup()
```

Loads, evictions, tokens per second and memory readings go to the root logger at
`INFO`.

### Adding a model kind

Subclass `ModelKind` and implement `load`. Whatever you return is handed to the
requests you send through `residency`.

```python
from vramen import ModelKind

class VisionModel(ModelKind):
    def load(self):
        return AutoModelForVision2Seq.from_pretrained(self.model_id), AutoProcessor.from_pretrained(self.model_id)
```

The instance is pickled into the child process, so keep its attributes
picklable. The manager reference is dropped on the way across.

## Platforms

**macOS on Apple silicon** is what this is built for and tested on. Models load
onto MPS, and the memory readings come from `torch.mps`.

**Linux** is not supported out of the box today. Three of the four model
classes pass `device_map="mps"` to `from_pretrained` and `Text2ImageModel` calls
`pipe.to("mps")`, so CUDA needs those changed — though a `Text2ImageModel` built
with `offload=True` asks `diffusers` for the accelerator and lands on CUDA as it
stands. Everything
else is portable: the process, queue and quota machinery is plain
`multiprocessing`, and `process_memory` already handles the Linux units for
`ru_maxrss`. Off MPS the GPU figures report `0.0`, which does not affect
admission, since the quota is spent in declarations rather than measurements.

**Windows** does not work. `vramen.utils` imports `resource`, which is POSIX
only, so the package fails at import. WSL is the path there.

## Developing

```bash
git clone https://github.com/robodatalab/vramen
cd vramen
uv sync
uv run python -m unittest discover -s tests -t . -v
```

`uv sync` installs vramen into `.venv` as an editable install, so `import vramen`
resolves to `src/vramen`.

## Releasing

Publishing runs on [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API token is stored in the repository. One-time setup on PyPI, under the
project's *Publishing* settings:

| Field | Value |
| --- | --- |
| Owner | `robodatalab` |
| Repository | `vramen` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

After that, releasing is automatic. Every push to main runs
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which runs the
tests, works out the version, commits and tags it, builds the sdist and wheel
with `uv build` and uploads them with `uv publish`. A merged pull request lands
on main as a push, so merges and direct pushes travel the same path. The job
requests a short-lived OIDC token from GitHub, PyPI verifies the claims against
the configuration above and mints a scoped upload credential for that run only.

Nothing has to be run by hand to cut a release, including the version:

| You push | You get |
| --- | --- |
| anything to main | the next patch — `0.2.0` → `0.2.1` |
| `version` in `pyproject.toml` raised to `0.3.0` | `0.3.0`, released as it stands |

So a patch costs nothing to think about, and a minor or a major is a one-line
edit made in the pull request that earns it. The next version is counted from
the highest `v*` tag rather than from the file, so reverting a release commit
cannot walk the version back into a number PyPI has already handed out.

To see what the next push would release, or to check the artifacts without
uploading:

```bash
make next-version
uv build
```

`make publish-local` is the fallback for when GitHub is unavailable. It uploads
from your machine with an API token, forfeiting trusted publishing and PEP 740
attestations, and it ships without tagging — the next push to main asks the
index and steps over whatever it already holds.

## License

[Apache 2.0](LICENSE). Use it freely, including commercially. Keep the copyright
notice and the [NOTICE](NOTICE) file, and say so if you modify a file.
