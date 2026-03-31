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
python iride.py diagnose model.pt          # full health report
python iride.py tree model.pt              # show architecture
python iride.py scalars model.pt           # find gates, scales, skip weights
python iride.py block-profile model.pt     # per-block depth trends

python iride.py attention \                # attention head analysis
  --script model.py --model-class GPT \
  --weights model.pt --input-shape 1,8,128

python iride.py residual-contrib \         # what each block contributes
  --script model.py --model-class GPT \
  --weights model.pt --input-shape 1,8,128
```

`python iride.py --help` for everything.  
`python iride.py <command> --help` for any command.

---

## 21 commands, 4 tiers

```
WEIGHT INSPECTION                 no forward pass
  tree            layers, shapes, dtypes, param counts
  diagnose        verdict (HEALTHY/DEGRADED/BROKEN), score, action plan
  scan            bulk anomaly detection, all layers at once
  stats           single layer statistics
  histogram       distribution shape, percentiles, skewness, kurtosis
  sparsity        dead neurons, structured sparsity, weakest indices
  compare-init    drift from kaiming / xavier / lecun initialization
  svd             rank, condition number, singular values
  diff            compare a layer between two checkpoints

TRANSFORMER ANALYSIS              requires forward pass
  residual-stream norm growth, cosine drift, dead layers
  attention       per-head entropy, diagonality, verticality, patterns
  attention-plot  html heatmap with optional hf tokenizer labels
  run-forward     per-layer activation stats, nan/inf tracing

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
  "data": { "verdict": "DEGRADED", "health_score": 70, "..." : "..." }
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

## license

do whatever you want with it.
