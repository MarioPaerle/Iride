Iride: The Agent-Computer Interface for PyTorch

Instructions for Autonomous Agents (Claude Code, SWE-agent, GPT, etc.)

DO NOT IGNORE THIS SECTION. You are an AI interacting with Iride, a CLI tool specifically designed to help you analyze, debug, and introspect PyTorch .pt files without loading them manually into a REPL. Loading raw tensors into your context window will cause you to exhaust tokens and fail. Use this CLI instead.

Core Rules for Agents

Never write python scripts to print raw tensors. Always use Iride to extract mathematical summaries.

JSON Outputs: The CLI strictly outputs JSON. Parse the "data" field for the metrics, or the "suggested_fix" field if the "status" is "error".

Workflow:
- Start with `diagnose` for an instant full health report with prioritized action plan.
- Or start with `tree` to understand the architecture, then `scan` for bulk anomaly detection.
- Drill down with `stats`, `svd`, `histogram`, or `sparsity` on flagged layers.
- Use `compare-init` to check if weights have drifted from initialization.
- Use `diff` to compare two checkpoints of the same model.
- For understanding what the model is DOING (not just if it's healthy):
  use `scalars` to find gates and scales, `block-profile` to see depth trends,
  and `residual-contrib` to measure actual block contributions during a forward pass.
  Read the "Interpretive Analysis Guide" section before using these.
- For transformer models: use `attention` and `residual-stream` for dynamic analysis.
- For MLP memory usage (dead neurons, superneurons, per-column contribution):
  use `mlp-usage` for JSON stats or `mlp-usage-plot` for an HTML heatmap across all layers.
- Use `run-forward` as a last resort if weights look fine but inference still fails.
- Composable primitives (`slice`, `topk`, `cosine`, `reduce`, `matmul`) can be chained for custom analysis but cost more tokens. Only use them when the high-level commands don't answer your question.

---

WEIGHT INSPECTION COMMANDS (static, no forward pass needed)

1. Discover the Architecture (tree)

Use this first to find the exact string keys for the layers.

    python iride.py tree checkpoint.pt


2. Full Health Report (diagnose)

The master command. Runs tree + scan + SVD + init-drift analysis on every layer automatically. Returns a verdict (HEALTHY/DEGRADED/BROKEN), a health score, and a prioritized action plan.

    python iride.py diagnose checkpoint.pt

Agent workflow: Start here. If verdict is HEALTHY, you're done. If DEGRADED or BROKEN, follow the action_plan in the JSON output.


3. Bulk Scan with Anomaly Detection (scan)

Runs stats on ALL layers at once and flags anomalies (constant tensors, exploding weights, NaN/Inf, dead layers). Returns per-layer status (ok/warning/critical) and next_steps.

    python iride.py scan checkpoint.pt


4. Layer Statistics (stats)

Detailed numerical statistics for a single layer.

    python iride.py stats checkpoint.pt --layer "transformer.h.0.attn.c_proj.weight"


5. Weight Distribution Histogram (histogram)

Bucket values into bins with percentiles, skewness, kurtosis, and distribution type detection (normal/degenerate/heavy-tailed/bimodal).

    python iride.py histogram checkpoint.pt --layer "fc1.weight"
    python iride.py histogram checkpoint.pt --layer "fc1.weight" --bins 30


6. Dead Neuron & Structured Sparsity Analysis (sparsity)

Per-neuron analysis: dead rows (output neurons), dead columns (input features), weakest neurons by L2 norm. Works on biases and conv filters too.

    python iride.py sparsity checkpoint.pt --layer "fc1.weight"
    python iride.py sparsity checkpoint.pt --layer "conv1.weight" --threshold 1e-4


7. Compare Weights vs Initialization (compare-init)

Compares every weight matrix against Kaiming/Xavier/LeCun expected distributions. Reports drift_ratio and status: dead, shrunk, near_init, trained, high_drift, exploded.

    python iride.py compare-init checkpoint.pt
    python iride.py compare-init checkpoint.pt --init kaiming


8. Singular Value Decomposition (svd)

Compute rank, condition number, singular values. If the layer is Conv2D (4D tensor), you MUST use --flatten.

    python iride.py svd checkpoint.pt --layer "fc1.weight"
    python iride.py svd checkpoint.pt --layer "conv1.weight" --flatten


9. Compare Checkpoints (diff)

Check if a layer changed between two checkpoints. l2_distance=0 means frozen gradients.

    python iride.py diff epoch_1.pt epoch_2.pt --layer "fc1.bias"


---

TRANSFORMER ANALYSIS COMMANDS (require forward pass)

These commands dynamically load a model, run a forward pass, and capture internal activations. They all share the same input arguments:

    --script <file.py>      Python file containing the model class
    --model-class <Name>    Name of the nn.Module class
    --weights <file.pt>     Checkpoint to load
    --input-shape <dims>    Comma-separated input shape, e.g. 1,8,128
    --tokenizer <name>      HuggingFace tokenizer (optional, enables --text)
    --text <string>         Real text input (requires --tokenizer)
    --causal <auto|true|false>  Causal masking mode (default: auto-detect)

IMPORTANT ON CAUSAL MASKING: If the model is a decoder (GPT, LLaMA, Mistral, etc.), causal masking is NOT optional. It MUST be applied or attention analysis will be wrong. The tool auto-detects causal models from class name and config. If auto-detection fails for a known causal model, you MUST pass --causal true. Never analyze a causal model's attention without masking.


10. Residual Stream Analysis (residual-stream)

Traces how hidden states evolve through the model. Captures norm growth, cosine similarity between consecutive layers, update ratios, and per-position variance. Detects norm explosion, collapse, and dead layers.

    python iride.py residual-stream \
      --script model.py --model-class MyTransformer \
      --weights checkpoint.pt --input-shape 1,8,128

    # With real text input
    python iride.py residual-stream \
      --script model.py --model-class MyTransformer \
      --weights checkpoint.pt --tokenizer gpt2 --text "The cat sat on"

Key metrics to watch:
- l2_norm across positions: should grow smoothly, not explode or collapse
- cosine_sim_to_previous: ~1.0 means a dead layer that doesn't transform the representation
- update_ratio: how much each layer changes the stream (healthy range: 0.01-0.5)
- per_position_variance: decreasing = representation collapse


11. Attention Head Analysis (attention)

Captures post-softmax attention weights from every head. Computes per-head entropy, diagonality, verticality, pattern classification, and top attention pairs.

    python iride.py attention \
      --script model.py --model-class MyTransformer \
      --weights checkpoint.pt --input-shape 1,8,128

    # Causal model with tokenized input
    python iride.py attention \
      --script model.py --model-class CausalLM \
      --weights checkpoint.pt --tokenizer gpt2 --text "The cat sat on"

REQUIREMENT: The model's attention module must return post-softmax attention weights as part of its forward() output (as a tuple element). Models that only return the projected output won't work — modify the model to also return attn_weights.

Per-head metrics:
- normalized_entropy: 0=perfectly peaked, 1=uniform (dead head)
- diagonality: high = self/local attention pattern
- local_diagonality: attention within +-1 of diagonal
- verticality: one key position attracts most attention (BOS/CLS sink)
- pattern: classified as "uniform", "diagonal", "local", "vertical", "sparse", or "diffuse"
- top_pairs: strongest attention connections (with token labels if tokenizer used)


12. Attention Heatmap Visualization (attention-plot)

Generates a self-contained HTML heatmap for a specific layer and head. Returns both the HTML file path (for the user to open in a browser) and machine-readable metrics in JSON.

    python iride.py attention-plot \
      --script model.py --model-class MyTransformer \
      --weights checkpoint.pt --input-shape 1,8,128 \
      --layer-idx 0 --head-idx 0

    # With token labels on axes
    python iride.py attention-plot \
      --script model.py --model-class CausalLM \
      --weights checkpoint.pt --tokenizer gpt2 --text "The cat sat on the mat" \
      --layer-idx 2 --head-idx 3

Use `attention` first to identify interesting heads, then `attention-plot` to visualize them.


13. Dynamic Forward Pass Inspection (run-forward)

Captures activation statistics from every layer during a forward pass. Use when weights look fine but inference produces NaN/Inf.

    python iride.py run-forward \
      --script model.py --model-class SimpleMLP \
      --weights checkpoint.pt --input-shape 1,128


13a. MLP Column Usage (mlp-usage)

Treats every column of W_up as a "memory neuron" (Geva 2020: FFN layers as key-value memories). For each MLP, runs a forward pass, hooks the up-projection, applies the activation, and reports per-column usage stats.

    python iride.py mlp-usage \
      --script model.py --model-class GPT \
      --weights model.pt --input-shape 4,512

    python iride.py mlp-usage \
      --script model.py --model-class GPT \
      --weights model.pt --text "The cat sat on" --tokenizer gpt2 \
      --activation relu2

Per-column metrics returned (as summary stats + top/bottom lists, never raw arrays):
- activation_fraction: P(|h_j| > threshold). Near zero = dead neuron. Near one = superneuron.
- post_mean: E[h_j] after activation. Magnitude of typical firing.
- contribution: E[|h_j|] * ||W_down[:, j]||. The honest "impact on output" metric. A neuron
  that fires loudly but whose W_down column is near-zero moves nothing.
- dead_count / super_count: based on activation_fraction thresholds (0.01 and 0.95).
- gini in [0, 1]: 0 = every neuron carries equal load, 1 = all load on one neuron.
- normalized_usage_entropy: 1.0 = flat use, <0.5 = sharply peaked.
- top_neurons / bottom_neurons: indices sorted by contribution.

Auto-detects MLP up-projections by: (a) module name keywords (fc1, c_fc, w_up, up_proj,
gate_proj, intermediate, ...) AND/OR (b) shape (out >= in * expansion_threshold).
Attention-like names (q_proj, kv, c_attn, ...) are excluded.

Override auto-detection with --mlp-pattern '<regex>' (run `tree` first to find module names).

Activation choices: relu2 (default), relu, gelu, silu, leakyrelu2, identity.
Use --activation identity when the hook captures already-activated values (fused MLP blocks).

--no-w-down disables the W_down weighting; contribution becomes just E[|h_j|].

Agent workflow: use mlp-usage to detect (a) wasted capacity from dead columns, (b) over-shared
generic neurons via super_count (candidates for a shared-prefix cache across layers), and
(c) which specific columns drive the output via top_neurons. High gini with low dead_count
means the layer has found specialists; low gini across the board means undifferentiated usage.


13b. MLP Usage Heatmap (mlp-usage-plot)

Same analysis as mlp-usage but renders an HTML heatmap (one row per MLP layer, one cell
per memory neuron). Dark blue = dead, cyan = moderate, red = heavily used. The visual analog
of `attention-plot` but for MLP memory.

    python iride.py mlp-usage-plot \
      --script model.py --model-class GPT \
      --weights model.pt --text "The cat sat on" --tokenizer gpt2 \
      --metric contribution --sort-by index

Metric choices:
- contribution (default): E[|h_j|] * ||W_down[:, j]||
- magnitude: E[|h_j|] post-activation
- fraction: P(|h_j| > threshold)

Sort choices:
- index (default): keep original column order; reveals spatial structure (shared-prefix
  clustering near column 0, dead tails, periodic patterns, per-bank bands).
- metric: sort each row descending; reveals the head/tail distribution.

--max-cols N (default 1024) caps cells per row; larger rows are stride-sampled.

Output: mlp_usage_<metric>_<sort-by>.html written next to the checkpoint. The JSON response
includes per_layer_summary (dead_count, super_count, gini, entropy, top_5_neurons) so the
agent can reason without opening the HTML.


---

COMPOSABLE PRIMITIVES (low-level, stackable, weight-only)

These are minimal single-purpose tools. Each does one thing and returns JSON. An agent can chain them to build custom analyses (e.g., manually computing OV circuits, comparing head subspaces, isolating specific neuron groups). They cost more tokens than high-level commands because multiple calls are needed. Only use them when the high-level commands don't answer your specific question.


14. Slice — Extract a Sub-Tensor

    python iride.py slice checkpoint.pt --layer "fc1.weight" --index "0:10,5:15"
    python iride.py slice checkpoint.pt --layer "fc1.weight" --index ":,3"
    python iride.py slice checkpoint.pt --layer "fc1.weight" --index "0"

Numpy-style indexing. Returns stats on the extracted sub-tensor.


15. Top-K Values — Find Outliers

    python iride.py topk checkpoint.pt --layer "fc1.weight" --k 10
    python iride.py topk checkpoint.pt --layer "fc1.weight" --k 5 --smallest

Returns values with multi-dimensional index coordinates.


16. Cosine Similarity — Compare Two Layers

    # Same file, different layers
    python iride.py cosine checkpoint.pt checkpoint.pt --layer1 "head.0.weight" --layer2 "head.1.weight"

    # Different files, same layer (track training progress)
    python iride.py cosine epoch1.pt epoch5.pt --layer1 "fc1.weight" --layer2 "fc1.weight"

Requires same total number of elements. Use `slice` first if shapes differ.


17. Reduce — Aggregate Along a Dimension

    python iride.py reduce checkpoint.pt --layer "fc1.weight" --dim 0 --op norm
    python iride.py reduce checkpoint.pt --layer "fc1.weight" --dim 1 --op mean

Operations: mean, sum, max, min, norm, var, absmax.
dim=0 on weights = per-input-feature summary. dim=1 = per-output-neuron summary.


18. Matrix Multiply — Compose Weight Matrices

    # Compute Q @ K^T (need transpose on K)
    python iride.py matmul checkpoint.pt --layer1 "attn.q.weight" --layer2 "attn.k.weight" --transpose2

    # Compute OV circuit: W_out @ W_value
    python iride.py matmul checkpoint.pt --layer1 "attn.out.weight" --layer2 "attn.v.weight"

Returns stats + rank of the product matrix. The rank reveals effective dimensionality of the composed transformation.


---

INTERPRETIVE ANALYSIS COMMANDS (what the model is trying to do)

These commands go beyond health checking. They tell you what the model has learned
about its own architecture -- which layers it trusts, which it bypasses, how it
distributes computation across depth.


19. Scalar Parameter Analysis (scalars)

Finds ALL small learned parameters (gates, scales, temperatures, skip weights).
Groups them by naming pattern, detects trends across depth, interprets sign patterns.

    python iride.py scalars checkpoint.pt
    python iride.py scalars checkpoint.pt --max-numel 32

What to look for:
- Sign changes across depth = model selectively uses (+) or suppresses (-) blocks
- Valley pattern (pos-neg-pos) = model relies on early/late blocks, suppresses middle
- All negative = model is actively subtracting those blocks' contributions
- Near zero = gates are inactive, those controls haven't been utilized yet
- Near one = identity-initialized gates, barely moved from init


20. Block Profile (block-profile)

Auto-detects repeating block structure and shows per-block comparison:
weight stds, norm means, scalars, and how they trend from first to last block.

    python iride.py block-profile checkpoint.pt

What to look for:
- Weight std increasing with depth = healthy gradient flow (deeper layers closer to loss)
- Weight std decreasing with depth = vanishing gradients, later layers not learning
- Norm weight mean != 1.0 = model is learning to rescale the residual stream at that point
- Norm mean changing across depth = model learned different scales for different layers


21. Residual Stream Contributions (residual-contrib)

Runs a forward pass and measures what each block ACTUALLY contributes to the
residual stream. For each block: how much it changes the stream, and whether
the change is aligned (amplifying), opposing (correcting), or orthogonal (adding new info).

    python iride.py residual-contrib \
      --script model.py --model-class GPT \
      --weights model.pt --input-shape 1,8,128

    # Specify block prefix if auto-detection fails
    python iride.py residual-contrib \
      --script model.py --model-class GPT \
      --weights model.pt --input-shape 1,8,128 --block-prefix "transformer.h"

Per-block behaviors:
- pass-through: block barely modifies the stream (contribution_ratio < 0.01). Dead or untrained.
- amplifying: update aligned with input (cos > 0.3). Reinforcing existing features.
- correcting: update opposes input (cos < -0.3). Subtracting noise or undoing earlier blocks.
- transforming: large orthogonal update (ratio > 0.5). Adding substantially new information.
- refining: moderate orthogonal update. Adding new features without reinforcing or opposing.


---

INTERPRETIVE ANALYSIS GUIDE

This section teaches you how to INTERPRET the numbers, not just report them.
The goal is to understand what the model is trying to do with its architecture.


Reading Scalar Parameters

Every learnable scalar is the model voting on something. They are never random noise.

- Gate/skip scalars control residual mixing. A value of 0.18 means "use 18% of this
  block's output." A value of -0.03 means "subtract 3% of this block's output from
  the stream" -- the model is actively trying to UNDO what that block does.

- Temperature scalars control distribution sharpness. Values > 1 make distributions
  flatter (more exploration), values < 1 make them peakier (more decisive).

- The PATTERN across depth matters more than individual values. A valley pattern
  (positive -> negative -> positive) means: "I need the early layers (embedding
  processing) and the late layers (output preparation), but the middle layers
  aren't useful yet." This is NORMAL in early training. With more epochs, the
  middle layers will learn useful features and their gates will rise.

- Sign changes are the most important signal. A model that flips from + to - at
  block 3 is telling you: "everything before block 3 helps, everything after hurts."
  That's a strong signal about where the model's useful representations live.


Reading Residual Contributions

The residual stream is the information highway through the model. Each block reads
from it and writes back to it. What a block writes back tells you its role:

- Amplifying blocks reinforce what's already in the stream. They're saying "yes,
  this signal is correct, more of it." Common in early layers that clean up embeddings.

- Correcting blocks subtract from the stream. They're saying "no, this signal is
  wrong, remove it." This is how the model learns to suppress noise or undo the
  damage from undertrained blocks. If a block has a negative gate AND correcting
  behavior, the model is DOUBLY trying to remove its influence.

- Transforming blocks add entirely new information orthogonal to the input. These
  are the blocks doing the "real work" -- extracting new features that weren't
  in the stream before.

- Pass-through blocks do nothing. In early training, middle blocks are often
  pass-through because they haven't learned useful features yet. The model routes
  around them via the residual connection.

The contribution_ratio tells you how much each block matters. A model where
block 0 has ratio 0.4 and block 4 has ratio 0.02 is telling you: "block 0 is
doing 20x more work than block 4."


Reading Block Profiles

Cross-block trends reveal gradient flow and training dynamics:

- If ALL weight stds increase with depth, gradients are flowing well. The last
  block gets the strongest signal (closest to loss) and has drifted most from init.

- If weight stds are flat, the model may be using techniques like gradient
  normalization or careful LR scheduling that equalize gradient magnitudes.

- Norm weight means tell you about learned scaling. If block 0's norm2 has
  mean 0.71 but block 7's has mean 1.02, the model learned to DOWNSCALE the
  residual stream before the first MLP (0.71x) but leave it unchanged before
  the last MLP (1.02x ~= 1.0). This is the model compensating for its own
  architecture -- it found that the first MLP works better with smaller inputs.

- QK-norm variance increasing with depth means later attention heads are
  developing more specialized query/key scaling. The heads are differentiating
  -- learning to weight different dimensions differently for attention.


Connecting the Signals

The real insight comes from combining multiple signals:

1. Run `scalars` -> find that middle blocks have negative gates
2. Run `block-profile` -> find that middle block weight stds are lowest
3. Run `residual-contrib` -> find that middle blocks are pass-through or correcting

These three signals tell the SAME story: the middle blocks haven't learned useful
features yet. The model is routing around them. This is expected in early training.

Alternatively:
1. `scalars` -> all gates positive and growing
2. `block-profile` -> weight stds uniformly increasing
3. `residual-contrib` -> blocks are mostly refining/transforming

This model is well-trained. Every block contributes, gradients flow well, and the
model trusts its full depth.


---

RECOMMENDED AGENT DECISION TREES

For general checkpoint debugging:

    1. diagnose -> read verdict
    2. If BROKEN/DEGRADED: follow action_plan.critical
    3. histogram on flagged layers -> understand distribution shape
    4. sparsity on flagged weight matrices -> find dead neurons
    5. compare-init -> check if issue is training-related or corruption

For understanding what the model is doing (the deep analysis):

    1. tree -> understand architecture, note any scalar params or special layers
    2. scalars -> find gates/scales, read the trend and sign pattern
    3. block-profile -> see how weight stds and norm means change across depth
    4. residual-contrib -> measure actual block contributions during forward pass
    5. Combine the three signals: do gate values explain the contribution patterns?
       Do weight stds correlate with contribution ratios? Write the narrative.

For transformer attention analysis:

    1. diagnose -> check weight health first
    2. attention -> get per-head entropy and pattern classification
    3. For interesting heads: attention-plot for visualization
    4. residual-stream -> check information flow
    5. If heads look redundant: cosine between head weight matrices

For manual mechanistic interpretability:

    1. tree -> find layer names
    2. slice -> extract Q, K, V weight sub-matrices for specific heads
    3. matmul -> compose QK^T or OV circuits
    4. svd on the product -> check effective rank
    5. topk -> find strongest connections in composed matrices

---

ERROR HANDLING

If you receive "status": "error", read the "suggested_fix" field. It contains specific instructions on how to fix your command.
