```
  ___ ____  ___ ____  _____
 |_ _|  _ \|_ _|  _ \| ____|
  | || |_) || || | | |  _|
  | ||  _ < | || |_| | |___
 |___|_| \_\___|____/|_____|

  the eye that sees through your weights.
```

---

Iride is a single-file CLI that gives AI agents full
introspection over PyTorch `.pt` checkpoints.

All output is machine-readable JSON.  
No tensor ever enters the context window.  
You give it a checkpoint. It tells you what's inside.

---

## requirements

```
pip install torch
pip install transformers   # optional, for tokenized attention plots
```

One file. No framework. No config.

---

## quick start

```bash
python iride.py scan model.pt              # bulk anomaly scan
python iride.py tree model.pt              # show architecture
python iride.py scalars model.pt           # find gates, scales, skip weights
python iride.py block-profile model.pt     # per-block depth trends

python iride.py attention \                # attention head analysis
  --script model.py --model-class GPT \
  --weights model.pt --input-shape 1,8,128

python iride.py residual-contrib \         # what each block contributes
  --script model.py --model-class GPT \
  --weights model.pt --input-shape 1,8,128

python iride.py mlp-usage \                # per-neuron MLP usage stats
  --script model.py --model-class GPT \
  --weights model.pt --input-shape 1,8,128

python iride.py mlp-usage-plot \           # HTML heatmap of MLP usage
  --script model.py --model-class GPT \
  --weights model.pt --input-shape 1,8,128
```

`python iride.py --help` for everything.  
`python iride.py <command> --help` for any command.

---

## 25 commands, 4 tiers

```
WEIGHT INSPECTION                 no forward pass
  tree            layers, shapes, dtypes, param counts
  scan            bulk anomaly detection, all layers at once
  stats           single layer statistics
  histogram       distribution shape, percentiles, skewness, kurtosis
  sparsity        dead neurons, structured sparsity, weakest indices
  compare-init    drift from kaiming / xavier / lecun initialization
  svd             rank, condition number, singular values
  stable-rank     per-layer stable rank, effective rank, rank collapse detection
  qk-spectral     per-head Q@K^T spectral norm, attention divergence risk
  super-weights   outlier FFN parameters (critical for quantization)
  diff            compare a layer between two checkpoints

TRANSFORMER ANALYSIS              requires forward pass
  residual-stream norm growth, cosine drift, dead layers
  attention       per-head entropy, diagonality, verticality, patterns
  attention-plot  html heatmap with optional hf tokenizer labels
  mlp-usage       per-neuron usage: dead / super / gini / contribution
  mlp-usage-plot  html heatmap, one row per mlp layer, one cell per neuron
  run-forward     per-layer activation stats, nan/inf tracing
  massive-activations  outlier hidden-state scalars (Sun et al. 2024)
  dormant-heads   per-head output-norm dormancy (Sanyal et al. 2025)

INTERPRETIVE ANALYSIS             what the model is trying to do
  scalars         find and interpret all gates, scales, skip weights
  block-profile   per-block comparison across depth
  residual-contrib measure each block's contribution to the stream

COMPOSABLE PRIMITIVES             low-level, stackable
  slice           sub-tensor by numpy-style index
  topk            largest/smallest values with coordinates
  cosine          cosine similarity between any two layers
  reduce          aggregate along a dimension
  matmul          multiply two matrices, report product rank
```

---

## what the agent sees

success:
```json
{
  "status": "success",
  "data": { "rank": 512, "condition_number": 42.3, "..." : "..." }
}
```

failure:
```json
{
  "status": "error",
  "error_type": "LayerNotFound",
  "message": "Layer 'fc3.weight' not found.",
  "suggested_fix": "Run 'python iride.py tree <file>' to list layers."
}
```

the agent reads `suggested_fix` and corrects itself.

---

## wiring it up to an agent

Iride was built for agents, not for you.  
Your job is to point the agent at it and step back.

**1.** drop `iride.py` and `agent_instructions.md` next to your checkpoints.

**2.** add this to your agent's system prompt, `CLAUDE.md`, or project context:

```
Before analyzing any .pt file, read agent_instructions.md and follow its rules.
Never write raw Python to inspect tensors. Use Iride.
```

**3.** for transformer analysis commands (`attention`, `residual-stream`,
`residual-contrib`, `attention-plot`), provide a python file with your model class.
the model's `forward()` must return attention weights for attention commands to work.

```python
# model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self): ...
    def forward(self, x):
        ...
        return output, attn_weights
```

causal masking is auto-detected for decoder models.  
if auto-detection fails, pass `--causal true`. it is not optional.

---

`agent_instructions.md` contains the full protocol:  
commands, decision trees, causal masking rules, error handling,  
and an interpretive analysis guide that teaches the agent  
how to read scalar parameters, residual contributions,  
and block-level trends -- not just numbers, but meaning.

---

## mlp usage pack

Transformer FFN layers behave as key-value memories
(Geva et al. 2020, Lample et al. 2019). Each column `j` of `W_up`
is a "key" matched against `X`; each column `j` of `W_down` is the
"value" it writes to the residual stream.

Iride's `mlp-usage` pack measures, on a forward pass, how often
each key fires, how strongly it fires, and how much the matching
value actually moves the output. Dead columns = wasted capacity.
Superneurons that fire for every token = candidates for a shared
prefix cache across layers. The long tail = layer-specific
specialists.

### what is measured, per column

- `activation_fraction`: P(|h_j| > threshold). Zero ≈ dead, one ≈ superneuron.
- `post_mean`: E[h_j] after activation.
- `contribution`: E[|h_j|] · ||W_down[:, j]||₂. The honest metric: a
  neuron that fires a lot but has a near-zero downstream column moves
  nothing. Contribution captures the actual effect on the output.
- `gini`: how unequal the usage is across neurons, in [0, 1].
- `normalized_usage_entropy`: 1.0 = flat usage, < 0.5 = peaked on few neurons.
- `top_neurons` / `bottom_neurons`: most / least useful columns by contribution.

### json output (`mlp-usage`)

```bash
python iride.py mlp-usage \
  --script model.py --model-class GPT \
  --weights model.pt --input-shape 4,512 \
  --activation relu2
```

Returns per-layer stats plus a global summary (total dead neurons,
super neurons, average gini). Use `--text "..." --tokenizer gpt2`
for token-aware measurement on realistic inputs.

### html heatmap (`mlp-usage-plot`)

```bash
python iride.py mlp-usage-plot \
  --script model.py --model-class GPT \
  --weights model.pt --text "The cat sat on" --tokenizer gpt2 \
  --metric contribution --sort-by index
```

Produces `mlp_usage_<metric>_<sort-by>.html` next to the checkpoint.
One row per MLP layer, one cell per memory neuron. Dark blue = dead,
cyan = moderate, red = heavily used. Hover any cell to see the raw value.

Two reading modes:

- `--sort-by index` keeps the original column order. Reveals spatial
  structure: shared-prefix clustering near column 0, dead tails,
  periodic patterns, or per-block usage bands in parameter-banked models.
- `--sort-by metric` sorts each row descending. Reveals the head/tail
  distribution and lets you compare how peaked each layer is.

Three metric modes:

- `--metric fraction`     how often the neuron fires
- `--metric magnitude`    how strongly it fires
- `--metric contribution` how much it moves the output (default, most honest)

### flags

- `--activation relu2 | relu | gelu | silu | leakyrelu2 | identity`
  applied to the captured up-projection output. Pass `identity` if
  your hook captures already-activated values (fused MLP modules).
- `--mlp-pattern '<regex>'` overrides auto-detection. Use when your
  up-projections don't match the standard keyword set (e.g. banked
  parameters, custom names). Run `tree` first to find the right names.
- `--expansion-threshold` (default 1.5): minimum `out/in` ratio to
  auto-classify a Linear as an MLP up-projection. Lower it if your
  MLP uses no expansion or a small one.
- `--no-w-down` reports raw post-activation magnitudes instead of
  output contributions. Use when W_down is tied, banked, or simply
  absent from the checkpoint.
- `--max-cols N` (plot only, default 1024): cells per row before
  stride-sampling. Raise it to see every neuron in wide MLPs.

### when to use what

| Question | Command |
|---|---|
| Any dead neurons? how many? | `mlp-usage` → `dead_pct_global` |
| Is a layer using its capacity well? | `mlp-usage` → `gini`, `normalized_usage_entropy` |
| Which neurons actually matter? | `mlp-usage` → `top_neurons` per layer |
| Visual structure across all layers at once | `mlp-usage-plot --sort-by index` |
| How peaked is each layer? | `mlp-usage-plot --sort-by metric` |
| Any super-generic shared neurons? | `mlp-usage` → `super_count`, then plot with `--metric fraction` |

---

## license

do whatever you want with it.
