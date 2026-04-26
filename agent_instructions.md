Iride: The Agent-Computer Interface for PyTorch

Instructions for Autonomous Agents (Claude Code, SWE-agent, GPT, etc.)

DO NOT IGNORE THIS SECTION. You are an AI interacting with Iride, a CLI tool specifically designed to help you analyze, debug, and introspect PyTorch .pt files without loading them manually into a REPL. Loading raw tensors into your context window will cause you to exhaust tokens and fail. Use this CLI instead.

Core Rules for Agents

Never write python scripts to print raw tensors. Always use Iride to extract mathematical summaries.

JSON Outputs: The CLI strictly outputs JSON. Parse the "data" field for the metrics, or the "suggested_fix" field if the "status" is "error".

MUST — Plot Transparency Rule: whenever you generate or refer to a plot (attention-plot, mlp-usage-plot, or any future visualization command), you MUST state clearly, in the same chat turn that shows the plot to the human:

  1. WHAT the plot shows, in one sentence (the axes and what the colour encodes).

  2. The exact METRIC used, as pseudocode or a short formula. For example, for mlp-usage-plot with --metric contribution:
         contribution[j] = mean_{b, l}( | activation(X W_up)[b, l, j] | ) * || W_down[:, j] ||_2
     Do not hand-wave. If the metric has a threshold, state the threshold. If it depends on the activation function, state the activation.

  3. The INPUT data used for the forward pass, and how representative it is of the real training distribution. Flag honestly:
       - "random noise from --input-shape 1,128" -> say explicitly it is synthetic and may not match real usage patterns.
       - "a single short --text prompt" -> say the batch is small and statistics are noisy.
       - "real tokenised batch of size N, sequence L" -> say N and L and where the tokens came from.
     If the data is NOT the training distribution, say so and flag that dead / super / contribution statistics may differ under real workload.

  4. Any CAVEAT introduced by flags that affect the meaning. Examples:
       - --no-w-down -> "contribution does not account for the W_down column norm, so a neuron that fires a lot but whose value column is near zero will still look important here."
       - --activation identity on a non-fused MLP -> "the captured tensor is the raw W_up output (pre-activation), so 'firing' is defined on the linear score, not the post-activation value."
       - --sort-by metric -> "columns are re-sorted per row; the x-axis is rank-by-usage, NOT the original neuron index."
       - --max-cols stride-sampling -> "only every k-th neuron is drawn; small clusters may be aliased out."

If the plot's underlying input was a random / synthetic batch, the human MUST be told in the chat, not just hidden in the JSON response. Silently using random data and presenting the plot as if it measured real model usage is a reporting bug.

Prefer to embed the metric pseudocode and input description directly inside the HTML plot when the command supports it. When it does not, put the same information in the chat alongside the "open this in a browser" instruction.

Workflow:
- Start with `tree` to understand the architecture, then `scan` for bulk anomaly detection.
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


2. Bulk Scan with Anomaly Detection (scan)

Runs stats on ALL layers at once and flags anomalies (constant tensors, exploding weights, NaN/Inf, dead layers). Returns per-layer status (ok/warning/critical) and next_steps.

    python iride.py scan checkpoint.pt


3. Layer Statistics (stats)

Detailed numerical statistics for a single layer.

    python iride.py stats checkpoint.pt --layer "transformer.h.0.attn.c_proj.weight"


4. Weight Distribution Histogram (histogram)

Bucket values into bins with percentiles, skewness, kurtosis, and distribution type detection (normal/degenerate/heavy-tailed/bimodal).

    python iride.py histogram checkpoint.pt --layer "fc1.weight"
    python iride.py histogram checkpoint.pt --layer "fc1.weight" --bins 30


5. Dead Neuron & Structured Sparsity Analysis (sparsity)

Per-neuron analysis: dead rows (output neurons), dead columns (input features), weakest neurons by L2 norm. Works on biases and conv filters too.

    python iride.py sparsity checkpoint.pt --layer "fc1.weight"
    python iride.py sparsity checkpoint.pt --layer "conv1.weight" --threshold 1e-4


6. Compare Weights vs Initialization (compare-init)

Compares every weight matrix against Kaiming/Xavier/LeCun expected distributions. Reports drift_ratio and status: dead, shrunk, near_init, trained, high_drift, exploded.

    python iride.py compare-init checkpoint.pt
    python iride.py compare-init checkpoint.pt --init kaiming


7. Singular Value Decomposition (svd)

Compute rank, condition number, singular values. If the layer is Conv2D (4D tensor), you MUST use --flatten.

    python iride.py svd checkpoint.pt --layer "fc1.weight"
    python iride.py svd checkpoint.pt --layer "conv1.weight" --flatten


8. Compare Checkpoints (diff)

Check if a layer changed between two checkpoints. l2_distance=0 means frozen gradients.

    python iride.py diff epoch_1.pt epoch_2.pt --layer "fc1.bias"


9. Stable Rank & Effective Rank (stable-rank)

Weights-only scan of every 2D weight matrix. Computes stable rank (||W||_F^2 / sigma_max^2), effective rank (exp of spectral entropy), and their ratios to min(shape). Flags rank collapse.

    python iride.py stable-rank checkpoint.pt
    python iride.py stable-rank checkpoint.pt --flatten --no-skip-embeddings

Key output fields:
- layers[name].stable_rank: effective number of "loud" singular directions
- layers[name].effective_rank: exp(spectral entropy); counts directions by energy share
- layers[name].srank_ratio / erank_ratio: normalized by min(shape); ratios much less than 1 indicate rank collapse
- layers[name].status: "ok" | "warning" (srank_ratio < 0.1 or erank_ratio < 0.5) | "critical" (stable_rank < 3)
- summary.worst_layer + worst_srank_ratio: inspect this layer first


10. QK Spectral Norm (qk-spectral)

Weights-only per-head spectral norm of W_Q @ W_K^T / sqrt(d_head) for every attention layer. High values predict attention entropy collapse and loss spikes (Zhai 2023, Takase 2025, OLMo 2 2025). Handles separate Q/K projections, fused QKV (GPT-2 c_attn, Qwen qkv_proj, BLOOM Wqkv), and GQA/MQA (num_kv_heads less than num_heads). num_heads is inferred from d_model when possible; pass --num-heads otherwise.

    python iride.py qk-spectral checkpoint.pt
    python iride.py qk-spectral checkpoint.pt --num-heads 32 --flag-threshold 100

Key output fields:
- layers[name].per_head_sigma: list of per-head sigma_max values, scaled by 1/sqrt(d_head)
- layers[name].max_sigma / mean_sigma: aggregate per layer
- layers[name].num_heads / num_kv_heads / d_head: resolved head geometry
- layers[name].fused: true if derived from a fused c_attn / qkv_proj weight
- layers[name].critical_heads / warning_heads: counts above thresholds
- layers[name].status: "ok" | "warning" (sigma > 0.5 * flag_threshold) | "critical" (sigma > flag_threshold)
- summary.global_max_sigma / global_max_layer / global_max_head: where divergence risk is highest
- summary.unresolved_layers: Q/K candidates that could not be resolved (missing pair, bad shape, etc.)


11. Super Weight Detection (super-weights)

Weights-only detector for 'super weights' -- individual scalar parameters in FFN down-projection matrices whose removal catastrophically collapses model quality (Yu et al., "The Super Weight in Large Language Models", ICLR 2025, arXiv 2411.07191). For each row of W, flags entries where both |W[r,c]| / median(|W[r,:]|) exceeds --ratio-threshold AND the MAD-based robust z-score exceeds --mad-threshold. Scans *.mlp.down_proj.weight, *.mlp.c_proj.weight, *.feed_forward.w2.weight and friends; pass --include-o-proj to also scan attention output projections.

    python iride.py super-weights checkpoint.pt
    python iride.py super-weights checkpoint.pt --include-o-proj
    python iride.py super-weights checkpoint.pt --ratio-threshold 200 --mad-threshold 75

Key output fields:
- by_layer[name]: list of outlier entries per primary layer, sorted by ratio_vs_row_median desc
  - each entry: {row, col, value, ratio_vs_row_median, mad_z, status}
  - status: "critical" (ratio >= 1000) | "warning" (otherwise)
- aux_projections[name]: same shape as by_layer, populated only when --include-o-proj
- skipped_layers: target matches skipped for reason "quantized_weight" | "not_2d" | "nan_or_inf"
- summary.max_ratio / top_layer: the single largest ratio observed and where
- summary.earliest_layer_idx: smallest layer index containing a super weight (super weights typically cluster in layers 1-3)
- summary.top_entries_global: flat list of the top --max-report entries across all layers
- super_weights_found: total count across by_layer + aux_projections


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


12. Residual Stream Analysis (residual-stream)

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


13. Attention Head Analysis (attention)

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


14. Attention Heatmap Visualization (attention-plot)

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

IMPORTANT: obey the Plot Transparency Rule from the Core Rules section. When you hand this HTML to the human, also tell them in chat: what the heatmap shows (attention weights, rows = query positions, columns = key positions), the metric as pseudocode (softmax_j(Q_i K_j^T / sqrt(d_k)), with causal mask if is_causal), and the input (real tokens vs synthetic --input-shape, batch/seq size). If you used a random input, flag it.


15. Dynamic Forward Pass Inspection (run-forward)

Captures activation statistics from every layer during a forward pass. Use when weights look fine but inference produces NaN/Inf.

    python iride.py run-forward \
      --script model.py --model-class SimpleMLP \
      --weights checkpoint.pt --input-shape 1,128


16. Massive Activations (massive-activations)

Forward-pass detector of outlier scalars in the residual stream (Sun et al., "Massive Activations in Large Language Models", COLM 2024, arXiv 2402.17762). Hooks every transformer block, captures the 3D hidden state, and flags individual entries whose absolute value is both above an absolute floor AND orders larger than the median of the tensor.

    python iride.py massive-activations \
      --script model.py --model-class LlamaForCausalLM \
      --weights checkpoint.pt --tokenizer gpt2 --text "Hello world"

    # Custom thresholds and more outliers per layer
    python iride.py massive-activations \
      --script model.py --model-class GPT \
      --weights checkpoint.pt --input-shape 1,128 \
      --abs-threshold 50 --ratio-threshold 500 --top-k 20

Key arguments:
- --abs-threshold (default 100.0): min |a| to count as massive.
- --ratio-threshold (default 1000.0): min |a| / median(|a|) to count as massive.
- --top-k (default 10): max outliers to report per layer.

Key output fields:
- layers: per-block dict with max_abs, median_abs, ratio, num_massive, top_outliers, status (ok | critical).
- summary: n_flagged, total_massive_scalars, max_ratio_global, first/last layer with massive activations.
- Each top_outlier gives token index, feature index, raw value, and absolute value.

What to look for:
- Massive activations concentrated on specific tokens/features across mid-to-late layers are EXPECTED in healthy decoder LLMs. They implement an attention-sink-like mechanism and carry information essential for generation.
- Absence of massive activations in mid/late layers of a supposedly trained decoder LLM, or a sudden drop between two checkpoints, indicates training pathology or aggressive quantization damage (outlier clipping).
- The exact tokens/features carrying the outlier are stable across inputs in a healthy model -- inspect `top_outliers` across a couple of different texts to confirm.



17. Dormant Heads (dormant-heads)

Per-head dormancy detector based on the L2 norm of each head's contribution to the residual stream (Sanyal et al., "Identifying and Evaluating Inactive Heads in Pretrained LLMs", arXiv 2504.03889, 2025). Attention-weight-only definitions (e.g. entropy) are inadequate; the right metric is ||A @ V||_2 per head, optionally projected through the corresponding W_O slice.

Default mode runs a forward pass: for each attention layer captures post-softmax A and the V output, reshapes V into per-head chunks (handling GQA/MQA via `repeat_interleave` and fused QKV projections by taking the last third of the last dim), applies causal masking if needed, then computes `head_norm = mean over batch and tokens of ||A @ V @ W_O^(h)||_2`. A head is flagged as dormant when head_norm is below `--dormancy-threshold` times the layer median. `--weights-only` skips the forward pass entirely and uses the proxy `||W_V^(h)||_F * ||W_O^(h)||_F`.

    # Activation mode
    python iride.py dormant-heads \
      --script model.py --model-class LlamaForCausalLM \
      --weights checkpoint.pt --tokenizer gpt2 --text "Hello world"

    # Weights-only mode (no forward pass)
    python iride.py dormant-heads --weights checkpoint.pt --weights-only

    # Custom threshold and explicit num-heads
    python iride.py dormant-heads --weights checkpoint.pt --weights-only \
      --num-heads 12 --dormancy-threshold 0.05

Key arguments:
- --dormancy-threshold (default 0.1): fraction of the layer median below which a head is dormant.
- --weights-only: skip forward pass, use weights-only proxy.
- --num-heads: override head-count inference (required with --weights-only when inference fails).
- --script / --model-class: required in activation mode only.

Key output fields:
- mode: "activation" or "weights_only".
- layers: per-layer dict with n_heads, n_kv_heads (activation mode), projection ("pre_W_O" | "post_W_O"), layer_median_norm or layer_median_score, heads (output_norm or weight_score, relative_to_median, is_dormant), dormant_heads, status (ok | warning | critical | fallback_weights_only | error).
- summary: n_layers, total_heads, dormant_heads, dormant_pct, mean_layer_median_norm (or score), max_dormant_pct_any_layer.

What to look for:
- Healthy pretrained LLMs typically show 8-15% dormant heads overall.
- >30% dormant in a single layer flags under-trained or collapsed attention and marks safe-to-ablate candidates per Sanyal et al. 2025.
- Fused QKV and GQA/MQA architectures are handled transparently; n_kv_heads reports the true value-head count.
- Layers with `status: "fallback_weights_only"` had no matchable V tensor during the forward pass -- rerun with `--weights-only` to get a complete weights-based picture.


---

COMPOSABLE PRIMITIVES (low-level, stackable, weight-only)

These are minimal single-purpose tools. Each does one thing and returns JSON. An agent can chain them to build custom analyses (e.g., manually computing OV circuits, comparing head subspaces, isolating specific neuron groups). They cost more tokens than high-level commands because multiple calls are needed. Only use them when the high-level commands don't answer your specific question.


18. Slice — Extract a Sub-Tensor

    python iride.py slice checkpoint.pt --layer "fc1.weight" --index "0:10,5:15"
    python iride.py slice checkpoint.pt --layer "fc1.weight" --index ":,3"
    python iride.py slice checkpoint.pt --layer "fc1.weight" --index "0"

Numpy-style indexing. Returns stats on the extracted sub-tensor.


19. Top-K Values — Find Outliers

    python iride.py topk checkpoint.pt --layer "fc1.weight" --k 10
    python iride.py topk checkpoint.pt --layer "fc1.weight" --k 5 --smallest

Returns values with multi-dimensional index coordinates.


20. Cosine Similarity — Compare Two Layers

    # Same file, different layers
    python iride.py cosine checkpoint.pt checkpoint.pt --layer1 "head.0.weight" --layer2 "head.1.weight"

    # Different files, same layer (track training progress)
    python iride.py cosine epoch1.pt epoch5.pt --layer1 "fc1.weight" --layer2 "fc1.weight"

Requires same total number of elements. Use `slice` first if shapes differ.


21. Reduce — Aggregate Along a Dimension

    python iride.py reduce checkpoint.pt --layer "fc1.weight" --dim 0 --op norm
    python iride.py reduce checkpoint.pt --layer "fc1.weight" --dim 1 --op mean

Operations: mean, sum, max, min, norm, var, absmax.
dim=0 on weights = per-input-feature summary. dim=1 = per-output-neuron summary.


22. Matrix Multiply — Compose Weight Matrices

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


23. Scalar Parameter Analysis (scalars)

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


24. Block Profile (block-profile)

Auto-detects repeating block structure and shows per-block comparison:
weight stds, norm means, scalars, and how they trend from first to last block.

    python iride.py block-profile checkpoint.pt

What to look for:
- Weight std increasing with depth = healthy gradient flow (deeper layers closer to loss)
- Weight std decreasing with depth = vanishing gradients, later layers not learning
- Norm weight mean != 1.0 = model is learning to rescale the residual stream at that point
- Norm mean changing across depth = model learned different scales for different layers


25. Residual Stream Contributions (residual-contrib)

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
2. Run `residual-contrib` -> find that middle blocks are pass-through or correcting

These three signals tell the SAME story: the middle blocks haven't learned useful
features yet. The model is routing around them. This is expected in early training.

Alternatively:
1. `scalars` -> all gates positive and growing
2. `block-profile` -> weight stds uniformly increasing
2. `residual-contrib` -> blocks are mostly refining/transforming

This model is well-trained. Every block contributes, gradients flow well, and the
model trusts its full depth.


---

RECOMMENDED AGENT DECISION TREES

For general checkpoint debugging:

    1. tree -> discover layer names
    2. scan -> flag layers with anomalies (ok/warning/critical)
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

    1. scan -> check weight health first
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
