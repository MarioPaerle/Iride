import argparse
import json
import sys
import os
import re
import traceback
import importlib.util
import math

# Ensure we don't spam stdout with warnings that break JSON parsing
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
except ImportError:
    print(json.dumps({"status": "error", "message": "PyTorch is not installed. Please install it."}))
    sys.exit(1)


def emit_result(data):
    """Outputs the final result as a clean JSON for the agent to parse."""
    print(json.dumps({"status": "success", "data": data}, indent=2))
    sys.exit(0)

def emit_error(error_type, message, suggested_fix):
    """Outputs errors in a structured format with actionable hints for the LLM."""
    print(json.dumps({
        "status": "error",
        "error_type": error_type,
        "message": message,
        "suggested_fix": suggested_fix
    }, indent=2))
    sys.exit(1)

def _raw_load(path):
    """Load a .pt file, trying safe mode first then full mode."""
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        pass
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        emit_error("LoadError", str(e), "Ensure the file is a valid PyTorch .pt or .pth state_dict file.")


# Common key names used by training frameworks to nest the actual state_dict
_STATE_DICT_KEYS = ["model_state_dict", "state_dict", "model", "module", "net", "params"]


def load_weights(path):
    if not os.path.exists(path):
        emit_error("FileNotFound", f"The file {path} does not exist.", "Check the file path using 'ls' and try again.")
    data = _raw_load(path)
    if isinstance(data, dict):
        # Check for a known nested state_dict key first
        for key in _STATE_DICT_KEYS:
            if key in data and isinstance(data[key], dict):
                nested = data[key]
                # Verify it actually contains tensors
                sample = list(nested.values())[:5]
                if sample and any(isinstance(v, torch.Tensor) for v in sample):
                    return nested
        # Otherwise return as-is (already a flat state_dict or unknown format)
    return data


def load_full_checkpoint(path):
    """Load the raw checkpoint without unwrapping, for metadata inspection."""
    if not os.path.exists(path):
        emit_error("FileNotFound", f"The file {path} does not exist.", "Check the file path using 'ls' and try again.")
    return _raw_load(path)

def get_layer(state_dict, layer_name):
    if layer_name not in state_dict:
        available = ", ".join(list(state_dict.keys())[:10])
        hint = f"Available layers (first 10): {available}" if state_dict else "The state_dict is empty."
        emit_error("LayerNotFound", f"Layer '{layer_name}' not found in state_dict.",
                   f"Run 'python iride.py tree <file>' to list all available layers. {hint}")
    return state_dict[layer_name]


# --- COMMAND IMPLEMENTATIONS ---

def cmd_tree(args):
    raw = load_full_checkpoint(args.weights)
    sd = load_weights(args.weights)

    # Detect if unwrapping happened
    unwrapped_from = None
    if isinstance(raw, dict):
        for key in _STATE_DICT_KEYS:
            if key in raw and isinstance(raw[key], dict):
                sample = list(raw[key].values())[:5]
                if sample and any(isinstance(v, torch.Tensor) for v in sample):
                    unwrapped_from = key
                    break

    # Build layer tree from the unwrapped state_dict
    tree_info = {}
    total_params = 0
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            tree_info[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "numel": v.numel()
            }
            total_params += v.numel()
        else:
            tree_info[k] = {"type": type(v).__name__}

    # Gather top-level checkpoint metadata (non-tensor fields from raw)
    metadata = {}
    if isinstance(raw, dict) and raw is not sd:
        for k, v in raw.items():
            if isinstance(v, dict) and k in _STATE_DICT_KEYS:
                continue  # skip the state_dict itself
            if isinstance(v, torch.Tensor):
                continue
            try:
                # Keep only JSON-serializable scalars and small values
                json.dumps(v)
                metadata[k] = v
            except (TypeError, ValueError, OverflowError):
                metadata[k] = {"type": type(v).__name__, "length": len(v) if hasattr(v, '__len__') else None}

    result = {
        "total_layers": len(tree_info),
        "total_parameters": total_params,
        "layers": tree_info,
    }
    if unwrapped_from:
        result["unwrapped_from"] = unwrapped_from
        result["agent_hint"] = (f"State dict was nested under '{unwrapped_from}'. "
                                f"Iride auto-unwrapped it. All other commands use the unwrapped weights.")
    if metadata:
        result["checkpoint_metadata"] = metadata

    emit_result(result)


def cmd_stats(args):
    sd = load_weights(args.weights)
    tensor = get_layer(sd, args.layer).float() # Cast to float for math
    
    stats = {
        "shape": list(tensor.shape),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "l2_norm": torch.norm(tensor, p=2).item(),
        "l1_norm": torch.norm(tensor, p=1).item(),
        "zeros_percentage": (tensor == 0).float().mean().item() * 100,
        "nans_count": torch.isnan(tensor).sum().item(),
        "infs_count": torch.isinf(tensor).sum().item()
    }
    emit_result(stats)


def cmd_svd(args):
    sd = load_weights(args.weights)
    tensor = get_layer(sd, args.layer).float()
    
    if tensor.dim() != 2:
        if args.flatten:
            tensor = tensor.view(tensor.size(0), -1)
        else:
            emit_error(
                "DimensionMismatch", 
                f"SVD requires a 2D matrix, but tensor has shape {list(tensor.shape)}.", 
                "Add the '--flatten' flag to your command to automatically reshape it to 2D (flattening all dimensions after the first)."
            )
            
    try:
        # Compute SVD
        U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
        s_min = S[-1].item()
        s_max = S[0].item()
        condition_number = (s_max / s_min) if s_min > 1e-7 else "Infinity"
        
        emit_result({
            "shape_analyzed": list(tensor.shape),
            "condition_number": condition_number,
            "top_5_singular_values": S[:5].tolist(),
            "bottom_5_singular_values": S[-5:].tolist(),
            "rank": torch.linalg.matrix_rank(tensor).item(),
            "is_rank_deficient": torch.linalg.matrix_rank(tensor).item() < min(tensor.shape),
            "agent_hint": "A high condition number (>1e4) or rank deficiency suggests dimensional collapse or redundant features."
        })
    except Exception as e:
        emit_error("SVDComputationError", str(e), "The tensor might contain NaNs or Infs. Run 'pt_agent.py stats' to check for numerical instability.")


def cmd_diff(args):
    sd1 = load_weights(args.weights1)
    sd2 = load_weights(args.weights2)
    
    t1 = get_layer(sd1, args.layer).float()
    t2 = get_layer(sd2, args.layer).float()
    
    if t1.shape != t2.shape:
        emit_error("ShapeMismatch", f"T1 shape {list(t1.shape)} != T2 shape {list(t2.shape)}", "You can only diff layers with identical shapes.")
        
    diff_tensor = t1 - t2
    cosine_sim = torch.nn.functional.cosine_similarity(t1.flatten().unsqueeze(0), t2.flatten().unsqueeze(0)).item()
    
    emit_result({
        "layer": args.layer,
        "l2_distance": torch.norm(diff_tensor, p=2).item(),
        "max_abs_difference": diff_tensor.abs().max().item(),
        "mean_difference": diff_tensor.mean().item(),
        "cosine_similarity": cosine_sim,
        "are_exactly_equal": torch.equal(t1, t2),
        "agent_hint": "If cosine_similarity is 1.0 and l2_distance is 0.0, the layer did not update between these checkpoints (frozen or zero gradient)."
    })


def cmd_run_forward(args):
    # Dynamically load the user's model code
    if not os.path.exists(args.script):
        emit_error("ScriptNotFound", f"Could not find {args.script}.", "Verify the path to the python script containing the model class.")
        
    try:
        spec = importlib.util.spec_from_file_location("dynamic_model", args.script)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        ModelClass = getattr(model_module, args.model_class)
    except AttributeError:
        emit_error("ClassNotFound", f"Class '{args.model_class}' not found in {args.script}.", f"Open {args.script} to verify the exact class name.")
    except Exception as e:
        emit_error("ImportError", str(e), "Failed to load the python script. Ensure it has no syntax errors.")

    # Parse input shape
    try:
        shape = [int(x.strip()) for x in args.input_shape.split(',')]
    except ValueError:
        emit_error("InvalidShape", f"Cannot parse '{args.input_shape}'.", "Provide a comma-separated list of integers, e.g., '1,3,224,224'.")

    # Instantiate and load weights
    try:
        model = ModelClass()
        sd = load_weights(args.weights)
        model.load_state_dict(sd)
        model.eval()
    except Exception as e:
        emit_error("ModelInitError", str(e), "The state_dict does not match the model architecture. Use 'pt_agent.py tree' to compare layers.")

    # Hook mechanism to capture activations
    activation_stats = {}
    
    def get_hook(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                out_float = output.detach().float()
                activation_stats[name] = {
                    "shape": list(out_float.shape),
                    "mean": out_float.mean().item(),
                    "std": out_float.std().item(),
                    "max": out_float.max().item(),
                    "min": out_float.min().item(),
                    "has_nans": torch.isnan(out_float).any().item()
                }
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Module) and not isinstance(layer, type(model)): # Skip root
            layer.register_forward_hook(get_hook(name))

    # Run forward pass
    try:
        dummy_input = torch.randn(*shape)
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
         emit_error("ForwardPassError", str(e), "The forward pass crashed. Check the expected input shape or data type for your model.")

    emit_result({
        "input_shape_used": shape,
        "layers_captured": len(activation_stats),
        "activation_statistics": activation_stats,
        "agent_hint": "Look for layers where 'std' collapses to 0 or 'max' explodes to Infinity. That indicates the point of failure in the forward pass."
    })


# --- NEW COMMANDS ---

def _compute_layer_stats(tensor):
    """Shared helper: compute stats dict for a single tensor (already float)."""
    return {
        "shape": list(tensor.shape),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "l2_norm": torch.norm(tensor, p=2).item(),
        "zeros_percentage": (tensor == 0).float().mean().item() * 100,
        "nans_count": torch.isnan(tensor).sum().item(),
        "infs_count": torch.isinf(tensor).sum().item(),
    }


def _detect_anomalies(name, tensor, stats):
    """Run heuristic checks on a layer and return a list of anomaly dicts."""
    anomalies = []
    sev = lambda level, msg, fix: anomalies.append({"severity": level, "message": msg, "suggested_fix": fix})

    # Critical: NaN / Inf
    if stats["nans_count"] > 0:
        sev("critical", f"{name} contains {stats['nans_count']} NaN values.",
            "Training diverged. Check learning rate, loss function, or input data for this layer.")
    if stats["infs_count"] > 0:
        sev("critical", f"{name} contains {stats['infs_count']} Inf values.",
            "Exploding gradients likely. Reduce learning rate or add gradient clipping.")

    # Critical: constant tensor (zero variance, non-zero values)
    if stats["std"] == 0.0 and tensor.numel() > 1:
        sev("critical", f"{name} is a constant tensor (every value = {stats['mean']:.6g}).",
            "This layer carries zero information. Likely corrupted, overwritten, or never trained. Re-initialize or restore from an earlier checkpoint.")

    # Warning: exploding weights
    if stats["std"] > 10.0:
        sev("warning", f"{name} has very high variance (std={stats['std']:.4f}).",
            "Possible exploding weights. Check gradient norms during training. Consider gradient clipping or lower learning rate.")

    # Warning: near-dead layer (extremely small std relative to magnitude)
    if 0 < stats["std"] < 1e-6 and "bias" not in name:
        sev("warning", f"{name} has near-zero variance (std={stats['std']:.2e}). Layer may be functionally dead.",
            "Vanishing gradients likely. Check activation functions and initialization. This layer is barely contributing.")

    # Warning: high sparsity for weights
    if stats["zeros_percentage"] > 50.0 and "weight" in name:
        sev("warning", f"{name} is {stats['zeros_percentage']:.1f}% zeros.",
            "High sparsity may indicate dead ReLU neurons or aggressive pruning. Run 'sparsity' for per-neuron analysis.")

    # Info: unusually large range
    abs_range = abs(stats["max"] - stats["min"])
    if abs_range > 100.0 and stats["std"] > 0:
        sev("info", f"{name} has a very wide value range [{stats['min']:.4f}, {stats['max']:.4f}].",
            "Check if this layer uses batch normalization or if values are pre/post scaling.")

    return anomalies


def cmd_scan(args):
    sd = load_weights(args.weights)

    results = {}
    all_anomalies = []
    total_params = 0

    for name, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.float()
        total_params += t.numel()
        stats = _compute_layer_stats(t)
        layer_anomalies = _detect_anomalies(name, t, stats)
        results[name] = {
            "stats": stats,
            "anomalies": layer_anomalies,
            "status": "critical" if any(a["severity"] == "critical" for a in layer_anomalies)
                      else "warning" if any(a["severity"] == "warning" for a in layer_anomalies)
                      else "ok"
        }
        all_anomalies.extend(layer_anomalies)

    critical_count = sum(1 for a in all_anomalies if a["severity"] == "critical")
    warning_count = sum(1 for a in all_anomalies if a["severity"] == "warning")

    # Build a prioritized action list for the agent
    next_steps = []
    for name, r in results.items():
        if r["status"] == "critical":
            next_steps.append(f"CRITICAL: Investigate '{name}' immediately. Run: stats --layer \"{name}\" and svd --layer \"{name}\"")
        elif r["status"] == "warning":
            next_steps.append(f"WARNING: Check '{name}'. Run: stats --layer \"{name}\"")

    emit_result({
        "total_layers": len(results),
        "total_parameters": total_params,
        "critical_issues": critical_count,
        "warnings": warning_count,
        "layers": results,
        "next_steps": next_steps if next_steps else ["All layers look healthy. No immediate action needed."],
        "agent_hint": "Focus on layers with status='critical' first. Use 'svd' on weight matrices to check for rank collapse. Use 'sparsity' if zeros_percentage is high."
    })


def cmd_histogram(args):
    sd = load_weights(args.weights)
    tensor = get_layer(sd, args.layer).float()
    flat = tensor.flatten()
    n_bins = args.bins if hasattr(args, 'bins') and args.bins else 20

    # Compute histogram
    hist = torch.histc(flat, bins=n_bins, min=flat.min().item(), max=flat.max().item())
    bin_edges_start = flat.min().item()
    bin_edges_end = flat.max().item()
    bin_width = (bin_edges_end - bin_edges_start) / n_bins if n_bins > 0 else 0

    bins = []
    for i in range(n_bins):
        lo = bin_edges_start + i * bin_width
        hi = lo + bin_width
        bins.append({
            "range": [round(lo, 6), round(hi, 6)],
            "count": int(hist[i].item()),
            "percentage": round(hist[i].item() / flat.numel() * 100, 2)
        })

    # Percentiles
    sorted_vals = flat.sort().values
    n = flat.numel()
    percentiles = {}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = min(int(n * p / 100), n - 1)
        percentiles[f"p{p}"] = round(sorted_vals[idx].item(), 6)

    # Distribution shape analysis
    mean = flat.mean().item()
    std = flat.std().item()
    skewness = 0.0
    kurtosis = 0.0
    if std > 1e-9:
        centered = flat - mean
        skewness = (centered ** 3).mean().item() / (std ** 3)
        kurtosis = (centered ** 4).mean().item() / (std ** 4) - 3.0  # excess kurtosis

    # Detect distribution type
    dist_type = "normal"
    dist_hints = []
    if std < 1e-7:
        dist_type = "degenerate"
        dist_hints.append("All values are nearly identical. This layer is non-functional.")
    elif abs(kurtosis) > 5:
        dist_type = "heavy-tailed"
        dist_hints.append(f"Excess kurtosis={kurtosis:.2f}. Outlier values may dominate this layer's behavior.")
    if abs(skewness) > 1.0:
        dist_hints.append(f"Skewness={skewness:.2f}. Distribution is asymmetric — check for bias in initialization or training.")

    # Check for bimodality: if the two highest bins are separated by a valley
    counts = [b["count"] for b in bins]
    if len(counts) >= 5:
        max_idx = counts.index(max(counts))
        # Look for a second peak separated by a valley
        has_valley = False
        for i in range(1, len(counts) - 1):
            if abs(i - max_idx) > 2 and counts[i] < counts[i-1] and counts[i] < counts[i+1]:
                if counts[i+1] > n * 0.05:  # second peak must be meaningful
                    has_valley = True
                    break
        if has_valley:
            dist_type = "bimodal"
            dist_hints.append("Distribution appears bimodal. Weights may have clustered into two groups — possible sign of mode collapse or quantization artifacts.")

    emit_result({
        "layer": args.layer,
        "total_values": n,
        "bins": bins,
        "percentiles": percentiles,
        "distribution_shape": {
            "type": dist_type,
            "skewness": round(skewness, 4),
            "excess_kurtosis": round(kurtosis, 4),
            "hints": dist_hints
        },
        "agent_hint": "A healthy weight layer typically shows a roughly bell-shaped distribution centered near 0. Degenerate=broken, heavy-tailed=unstable, bimodal=suspicious."
    })


def cmd_sparsity(args):
    sd = load_weights(args.weights)
    tensor = get_layer(sd, args.layer).float()

    total = tensor.numel()
    exact_zeros = (tensor == 0).sum().item()
    near_zeros = (tensor.abs() < args.threshold).sum().item()

    result = {
        "layer": args.layer,
        "shape": list(tensor.shape),
        "total_elements": total,
        "exact_zeros": exact_zeros,
        "exact_zero_pct": round(exact_zeros / total * 100, 2),
        "near_zeros": near_zeros,
        "near_zero_pct": round(near_zeros / total * 100, 2),
        "threshold_used": args.threshold,
    }

    # Structured sparsity: per-row and per-column analysis for 2D weight matrices
    if tensor.dim() == 2:
        rows, cols = tensor.shape

        # Row analysis (output neurons)
        row_norms = torch.norm(tensor, p=2, dim=1)
        dead_rows = (row_norms == 0).sum().item()
        near_dead_rows = (row_norms < args.threshold).sum().item()
        row_norm_list = row_norms.tolist()

        # Column analysis (input features)
        col_norms = torch.norm(tensor, p=2, dim=0)
        dead_cols = (col_norms == 0).sum().item()
        near_dead_cols = (col_norms < args.threshold).sum().item()
        col_norm_list = col_norms.tolist()

        # Identify the specific dead/weakest neurons
        weakest_rows = sorted(range(rows), key=lambda i: row_norm_list[i])[:min(5, rows)]
        weakest_cols = sorted(range(cols), key=lambda i: col_norm_list[i])[:min(5, cols)]

        result["structured_sparsity"] = {
            "output_neurons": {
                "total": rows,
                "dead": dead_rows,
                "near_dead": near_dead_rows,
                "dead_pct": round(dead_rows / rows * 100, 2),
                "weakest_indices": weakest_rows,
                "weakest_norms": [round(row_norm_list[i], 6) for i in weakest_rows],
            },
            "input_features": {
                "total": cols,
                "dead": dead_cols,
                "near_dead": near_dead_cols,
                "dead_pct": round(dead_cols / cols * 100, 2),
                "weakest_indices": weakest_cols,
                "weakest_norms": [round(col_norm_list[i], 6) for i in weakest_cols],
            },
            "is_structured": dead_rows > 0 or dead_cols > 0,
        }

        # Variance across neurons — detects if some neurons dominate
        result["neuron_variance"] = {
            "row_norm_std": round(row_norms.std().item(), 6),
            "row_norm_mean": round(row_norms.mean().item(), 6),
            "col_norm_std": round(col_norms.std().item(), 6),
            "col_norm_mean": round(col_norms.mean().item(), 6),
        }

        anomalies = []
        if dead_rows > 0:
            anomalies.append(f"{dead_rows}/{rows} output neurons are completely dead (zero norm).")
        if dead_cols > 0:
            anomalies.append(f"{dead_cols}/{cols} input features are completely ignored (zero norm).")
        if row_norms.std().item() > row_norms.mean().item() * 2 and row_norms.mean().item() > 0:
            anomalies.append("Extreme variance in output neuron norms — some neurons dominate while others barely contribute.")
        result["anomalies"] = anomalies

    elif tensor.dim() == 1:
        dead_units = (tensor == 0).sum().item()
        near_dead_units = (tensor.abs() < args.threshold).sum().item()
        result["bias_sparsity"] = {
            "dead_units": dead_units,
            "near_dead_units": near_dead_units,
        }
        result["anomalies"] = []
        if dead_units > tensor.numel() * 0.5:
            result["anomalies"].append(f"{dead_units}/{tensor.numel()} bias values are exactly zero.")
    else:
        # For higher-dim tensors (conv), flatten to (out_channels, -1)
        reshaped = tensor.view(tensor.size(0), -1)
        filter_norms = torch.norm(reshaped, p=2, dim=1)
        dead_filters = (filter_norms == 0).sum().item()
        result["filter_sparsity"] = {
            "total_filters": tensor.size(0),
            "dead_filters": dead_filters,
            "dead_pct": round(dead_filters / tensor.size(0) * 100, 2),
        }
        result["anomalies"] = []
        if dead_filters > 0:
            result["anomalies"].append(f"{dead_filters}/{tensor.size(0)} convolutional filters are completely dead.")

    result["agent_hint"] = "Dead neurons (zero norm rows) waste capacity and suggest dying ReLU or broken gradients. Near-dead neurons with threshold check catch vanishing weights. Use 'histogram' for distribution details."
    emit_result(result)


def cmd_compare_init(args):
    sd = load_weights(args.weights)

    init_type = args.init if hasattr(args, 'init') and args.init else "auto"

    results = {}

    for name, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.float()

        # Only analyze weight matrices (skip biases — they're typically zero-init)
        if "bias" in name:
            continue
        if t.dim() < 2:
            continue

        fan_in = t.shape[1]
        fan_out = t.shape[0]
        for d in t.shape[2:]:  # conv kernel dims
            fan_in *= d
            fan_out *= d

        actual_std = t.std().item()
        actual_mean = t.mean().item()

        # Compute expected stds for common inits
        kaiming_std = math.sqrt(2.0 / fan_in)
        xavier_std = math.sqrt(2.0 / (fan_in + fan_out))
        lecun_std = math.sqrt(1.0 / fan_in)

        # Pick the best matching init or use the user-specified one
        if init_type == "auto":
            candidates = {
                "kaiming": kaiming_std,
                "xavier": xavier_std,
                "lecun": lecun_std,
            }
            # Find closest match
            best_init = min(candidates, key=lambda k: abs(actual_std - candidates[k]))
            expected_std = candidates[best_init]
            detected_init = best_init
        else:
            detected_init = init_type
            expected_std = {"kaiming": kaiming_std, "xavier": xavier_std, "lecun": lecun_std}.get(init_type, kaiming_std)

        drift_ratio = actual_std / expected_std if expected_std > 1e-9 else float('inf')

        # Determine status
        if actual_std < 1e-7:
            status = "dead"
            interpretation = "Layer has near-zero variance. Not functioning."
        elif drift_ratio > 5.0:
            status = "exploded"
            interpretation = f"Weights are {drift_ratio:.1f}x larger than expected {detected_init} init. Likely overtrained or diverging."
        elif drift_ratio > 2.0:
            status = "high_drift"
            interpretation = f"Weights are {drift_ratio:.1f}x larger than expected. Significant training has occurred or lr is high."
        elif drift_ratio < 0.2:
            status = "shrunk"
            interpretation = f"Weights are {drift_ratio:.2f}x of expected — much smaller. Possible vanishing gradients or excessive regularization."
        elif drift_ratio < 0.5:
            status = "low_drift"
            interpretation = f"Weights are smaller than expected ({drift_ratio:.2f}x). Mild shrinkage, possibly from weight decay."
        elif 0.8 <= drift_ratio <= 1.2:
            status = "near_init"
            interpretation = "Weights are very close to initialization. Layer may not have trained much."
        else:
            status = "trained"
            interpretation = f"Drift ratio {drift_ratio:.2f}x — reasonable deviation from init. Layer appears to have trained normally."

        results[name] = {
            "shape": list(t.shape),
            "fan_in": fan_in,
            "fan_out": fan_out,
            "actual_std": round(actual_std, 6),
            "actual_mean": round(actual_mean, 6),
            "expected_inits": {
                "kaiming_std": round(kaiming_std, 6),
                "xavier_std": round(xavier_std, 6),
                "lecun_std": round(lecun_std, 6),
            },
            "closest_init": detected_init,
            "drift_ratio": round(drift_ratio, 4),
            "status": status,
            "interpretation": interpretation,
        }

    # Summary
    statuses = [r["status"] for r in results.values()]

    emit_result({
        "init_comparison": results,
        "summary": {
            "layers_analyzed": len(results),
            "dead": statuses.count("dead"),
            "exploded": statuses.count("exploded"),
            "high_drift": statuses.count("high_drift"),
            "trained": statuses.count("trained"),
            "near_init": statuses.count("near_init"),
            "shrunk": statuses.count("shrunk"),
            "low_drift": statuses.count("low_drift"),
        },
        "agent_hint": "Status key: 'near_init'=barely trained, 'trained'=normal, 'high_drift'/'exploded'=overshot, 'shrunk'/'dead'=vanished. Layers stuck at 'near_init' may have frozen gradients. 'exploded' layers need lr reduction or clipping."
    })


# --- LLM WEIGHT DIAGNOSTICS ---

def cmd_stable_rank(args):
    sd = load_weights(args.weights)

    layers = {}
    n_scanned = 0
    n_skipped_1d = 0
    n_skipped_nd = 0
    n_skipped_embed = 0
    n_skipped_too_small = 0
    n_error = 0

    srank_vals = []
    erank_vals = []
    srank_ratios = []
    erank_ratios = []
    low_stable_rank_layers = 0
    low_effective_rank_layers = 0
    critical_layers = 0
    worst_layer = None
    worst_srank_ratio = float("inf")

    def _round_ratio(x):
        return round(x, 6) if x < 0.01 else round(x, 4)

    for name, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue

        # Skip 1D tensors (biases, norms)
        if v.dim() == 1:
            n_skipped_1d += 1
            continue

        # Handle N-D tensors
        if v.dim() != 2:
            if args.flatten:
                tensor = v.float().view(v.size(0), -1)
            else:
                n_skipped_nd += 1
                continue
        else:
            tensor = v.float()

        shape = list(tensor.shape)

        # Skip embeddings when requested
        if args.skip_embeddings and "embed" in name.lower() and max(shape) > args.embed_threshold:
            n_skipped_embed += 1
            continue

        # Skip too-small matrices
        if min(shape) < args.min_dim:
            n_skipped_too_small += 1
            continue

        # NaN/Inf guard
        if torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item():
            layers[name] = {
                "shape": shape,
                "status": "error",
                "reason": "nan_or_inf",
            }
            n_error += 1
            continue

        try:
            sigma = torch.linalg.svdvals(tensor)
        except Exception as e:
            layers[name] = {
                "shape": shape,
                "status": "error",
                "reason": f"svd_failed: {str(e)}",
            }
            n_error += 1
            continue

        s_max = sigma[0].item()
        if s_max <= 0.0 or not math.isfinite(s_max):
            layers[name] = {
                "shape": shape,
                "status": "error",
                "reason": "degenerate_spectrum",
            }
            n_error += 1
            continue

        sigma_sq = sigma * sigma
        stable_rank = (sigma_sq.sum() / (sigma[0] * sigma[0])).item()

        sigma_sum = sigma.sum().item()
        if sigma_sum <= 0.0:
            layers[name] = {
                "shape": shape,
                "status": "error",
                "reason": "zero_spectrum",
            }
            n_error += 1
            continue

        p = sigma / sigma.sum()
        entropy = -(p * torch.log(p + 1e-12)).sum().item()
        effective_rank = math.exp(entropy)

        max_rank = min(shape)
        srank_ratio = stable_rank / max_rank
        erank_ratio = effective_rank / max_rank

        # Status determination
        if stable_rank < 3.0:
            status = "critical"
            critical_layers += 1
        elif srank_ratio < 0.1 or erank_ratio < 0.5:
            status = "warning"
        else:
            status = "ok"

        if srank_ratio < 0.1:
            low_stable_rank_layers += 1
        if erank_ratio < 0.5:
            low_effective_rank_layers += 1

        if srank_ratio < worst_srank_ratio:
            worst_srank_ratio = srank_ratio
            worst_layer = name

        layers[name] = {
            "shape": shape,
            "stable_rank": round(stable_rank, 4),
            "effective_rank": round(effective_rank, 4),
            "max_rank": max_rank,
            "srank_ratio": _round_ratio(srank_ratio),
            "erank_ratio": _round_ratio(erank_ratio),
            "status": status,
        }

        n_scanned += 1
        srank_vals.append(stable_rank)
        erank_vals.append(effective_rank)
        srank_ratios.append(srank_ratio)
        erank_ratios.append(erank_ratio)

    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    summary = {
        "n_layers_scanned": n_scanned,
        "n_skipped_1d": n_skipped_1d,
        "n_skipped_nd": n_skipped_nd,
        "n_skipped_embed": n_skipped_embed,
        "n_skipped_too_small": n_skipped_too_small,
        "n_error": n_error,
        "mean_stable_rank": _mean(srank_vals),
        "mean_effective_rank": _mean(erank_vals),
        "mean_srank_ratio": _mean(srank_ratios),
        "mean_erank_ratio": _mean(erank_ratios),
        "low_stable_rank_layers": low_stable_rank_layers,
        "low_effective_rank_layers": low_effective_rank_layers,
        "critical_layers": critical_layers,
        "worst_layer": worst_layer,
        "worst_srank_ratio": _round_ratio(worst_srank_ratio) if worst_layer is not None else None,
    }

    emit_result({
        "weights": args.weights,
        "layers": layers,
        "summary": summary,
        "agent_hint": "stable_rank is the effective number of 'loud' singular directions (||W||_F^2 / sigma_max^2); effective_rank = exp(spectral entropy). Ratios much less than 1 mean rank collapse: srank_ratio < 0.1 or absolute stable_rank < 3 signals gradient pathology or dead subspace - inspect worst_layer first."
    })


def _infer_num_heads(d_model, preferred_d_heads=(64, 96, 128, 80, 256)):
    """Infer num_heads from d_model by matching a preferred d_head size.

    Returns the first num_heads whose corresponding d_head appears in
    preferred_d_heads (in the order given). If multiple divisors match or
    none do, returns None.
    """
    matches = []
    for d_head in preferred_d_heads:
        if d_head <= 0:
            continue
        if d_model % d_head == 0:
            matches.append((d_head, d_model // d_head))
    if not matches:
        return None
    # Prefer first match in preferred order
    return matches[0][1]


# Suffix patterns for Q/K projection weight naming conventions
_Q_SUFFIXES = (
    ".q_proj.weight",
    ".wq.weight",
    ".q.weight",
    ".query.weight",
)
_K_SUFFIXES = (
    ".k_proj.weight",
    ".wk.weight",
    ".k.weight",
    ".key.weight",
)
_FUSED_SUFFIXES = (
    ".c_attn.weight",
    ".qkv_proj.weight",
    ".Wqkv.weight",
)


def cmd_qk_spectral(args):
    sd = load_weights(args.weights)

    # Group Q/K matrices by parent attention layer.
    # Each entry: {"Q": tensor, "K": tensor, "fused": bool, "source": str}
    layer_groups = {}
    unresolved = []

    for name, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.dim() < 2:
            continue

        matched = False

        # Fused QKV (GPT-2 c_attn, Qwen qkv_proj, BLOOM Wqkv)
        for suf in _FUSED_SUFFIXES:
            if name.endswith(suf):
                parent = name[:-len(suf)]
                t = v.float()
                if t.dim() != 2 or t.shape[0] % 3 != 0:
                    unresolved.append({"layer": parent, "reason": f"fused_bad_shape: {list(t.shape)}"})
                    matched = True
                    break
                chunk = t.shape[0] // 3
                Q = t[0:chunk]
                K = t[chunk:2 * chunk]
                entry = layer_groups.setdefault(parent, {"Q": None, "K": None, "fused": True, "source": name})
                entry["Q"] = Q
                entry["K"] = K
                entry["fused"] = True
                entry["source"] = name
                matched = True
                break
        if matched:
            continue

        # Separate Q
        for suf in _Q_SUFFIXES:
            if name.endswith(suf):
                parent = name[:-len(suf)]
                entry = layer_groups.setdefault(parent, {"Q": None, "K": None, "fused": False, "source": parent})
                entry["Q"] = v.float()
                entry["fused"] = entry.get("fused", False)
                matched = True
                break
        if matched:
            continue

        # Separate K
        for suf in _K_SUFFIXES:
            if name.endswith(suf):
                parent = name[:-len(suf)]
                entry = layer_groups.setdefault(parent, {"Q": None, "K": None, "fused": False, "source": parent})
                entry["K"] = v.float()
                entry["fused"] = entry.get("fused", False)
                matched = True
                break

    if not layer_groups:
        emit_error(
            "NoAttentionWeightsFound",
            "No attention Q/K weight matrices were detected in the state_dict.",
            "Expected state_dict keys matching *.q_proj.weight / *.k_proj.weight or *.c_attn.weight. Use 'tree' to list layers.",
        )

    flag_threshold = float(args.flag_threshold)
    warn_threshold = 0.5 * flag_threshold

    layers_out = {}
    max_sigmas = []
    flagged_layers = 0
    global_max_sigma = -1.0
    global_max_layer = None
    global_max_head = -1

    for parent, entry in layer_groups.items():
        Q = entry.get("Q")
        K = entry.get("K")
        if Q is None or K is None:
            missing = "Q" if Q is None else "K"
            unresolved.append({"layer": parent, "reason": f"missing_{missing}"})
            continue

        if Q.dim() != 2 or K.dim() != 2:
            unresolved.append({"layer": parent, "reason": f"non_2d_shape: Q={list(Q.shape)} K={list(K.shape)}"})
            continue

        d_q = Q.shape[0]
        d_k = K.shape[0]
        d_model = Q.shape[1]
        if K.shape[1] != d_model:
            unresolved.append({"layer": parent, "reason": f"dim_mismatch: Q={list(Q.shape)} K={list(K.shape)}"})
            continue

        # Resolve num_heads for this layer
        if args.num_heads is not None:
            num_heads_q = int(args.num_heads)
        else:
            num_heads_q = _infer_num_heads(d_q)
            if num_heads_q is None:
                emit_error(
                    "UnknownNumHeads",
                    f"Could not infer num_heads for layer '{parent}' (Q shape {list(Q.shape)}, d_model {d_model}).",
                    "pass --num-heads",
                )

        if num_heads_q <= 0 or d_q % num_heads_q != 0:
            unresolved.append({"layer": parent, "reason": f"num_heads_invalid: d_q={d_q} num_heads={num_heads_q}"})
            continue

        d_head = d_q // num_heads_q
        if d_head <= 0 or d_k % d_head != 0:
            unresolved.append({"layer": parent, "reason": f"d_head_invalid: d_head={d_head} d_k={d_k}"})
            continue

        num_kv_heads = d_k // d_head
        scale = 1.0 / math.sqrt(d_head)

        per_head_sigma = []
        critical_heads = 0
        warning_heads = 0
        layer_status = "ok"

        for h in range(num_heads_q):
            kv_h = h * num_kv_heads // num_heads_q
            Wq_h = Q[h * d_head:(h + 1) * d_head]
            Wk_h = K[kv_h * d_head:(kv_h + 1) * d_head]
            if torch.isnan(Wq_h).any().item() or torch.isinf(Wq_h).any().item() \
                    or torch.isnan(Wk_h).any().item() or torch.isinf(Wk_h).any().item():
                per_head_sigma.append(None)
                continue

            M = Wq_h @ Wk_h.t()
            try:
                sv = torch.linalg.svdvals(M)
            except Exception:
                per_head_sigma.append(None)
                continue

            sigma = sv[0].item() * scale
            if not math.isfinite(sigma):
                per_head_sigma.append(None)
                continue

            per_head_sigma.append(round(sigma, 4))

            if sigma > flag_threshold:
                critical_heads += 1
                layer_status = "critical"
            elif sigma > warn_threshold:
                warning_heads += 1
                if layer_status != "critical":
                    layer_status = "warning"

            if sigma > global_max_sigma:
                global_max_sigma = sigma
                global_max_layer = parent
                global_max_head = h

        valid_sigmas = [s for s in per_head_sigma if s is not None]
        if not valid_sigmas:
            unresolved.append({"layer": parent, "reason": "all_heads_failed"})
            continue

        max_sigma = max(valid_sigmas)
        mean_sigma = sum(valid_sigmas) / len(valid_sigmas)

        layers_out[parent] = {
            "num_heads": num_heads_q,
            "num_kv_heads": num_kv_heads,
            "d_head": d_head,
            "fused": bool(entry.get("fused", False)),
            "per_head_sigma": per_head_sigma,
            "max_sigma": round(max_sigma, 4),
            "mean_sigma": round(mean_sigma, 4),
            "critical_heads": critical_heads,
            "warning_heads": warning_heads,
            "status": layer_status,
        }

        max_sigmas.append(max_sigma)
        if layer_status != "ok":
            flagged_layers += 1

    if not layers_out:
        emit_error(
            "NoAttentionWeightsFound",
            "Detected Q/K candidates but none resolved into a valid attention layer.",
            "Expected state_dict keys matching *.q_proj.weight / *.k_proj.weight or *.c_attn.weight. Use 'tree' to list layers.",
        )

    mean_max = sum(max_sigmas) / len(max_sigmas) if max_sigmas else 0.0

    summary = {
        "n_layers": len(layers_out),
        "n_flagged": flagged_layers,
        "mean_max_sigma": round(mean_max, 4),
        "global_max_sigma": round(global_max_sigma, 4) if global_max_layer is not None else None,
        "global_max_layer": global_max_layer,
        "global_max_head": global_max_head if global_max_layer is not None else None,
        "unresolved_layers": unresolved,
    }

    emit_result({
        "weights": args.weights,
        "num_heads": int(args.num_heads) if args.num_heads is not None else None,
        "flag_threshold": flag_threshold,
        "layers": layers_out,
        "summary": summary,
        "agent_hint": "Per-head sigma_max(W_Q W_K^T)/sqrt(d_head) > 100 is a known attention-entropy-collapse precursor (Zhai 2023, Takase 2025). Critical layers warrant QK-RMSNorm, z-loss, or sigmaReparam before resuming training.",
    })


# Suffix patterns for FFN down-projection weight naming conventions
_DOWN_PROJ_SUFFIXES = (
    ".mlp.down_proj.weight",
    ".mlp.c_proj.weight",
    ".mlp.W_down.weight",
    ".ffn.down_proj.weight",
    ".feed_forward.w2.weight",
)

# Suffix patterns for attention output-projection weight naming conventions
_O_PROJ_SUFFIXES = (
    ".self_attn.o_proj.weight",
    ".attn.c_proj.weight",
    ".self_attn.Wo.weight",
    ".attn.out_proj.weight",
)

# Dtype set for quantized tensors that we cannot meaningfully scan
_QUANTIZED_DTYPES = {torch.int8, torch.uint8, torch.qint8}
if hasattr(torch, "int4"):
    _QUANTIZED_DTYPES.add(torch.int4)


def _extract_layer_idx(name):
    """Extract the first integer layer index from a layer name (e.g. 'model.layers.2.mlp.down_proj.weight' -> 2)."""
    m = re.search(r"\.(\d+)\.", name)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def cmd_super_weights(args):
    sd = load_weights(args.weights)

    target_suffixes = list(_DOWN_PROJ_SUFFIXES)
    if args.include_o_proj:
        aux_suffixes = list(_O_PROJ_SUFFIXES)
    else:
        aux_suffixes = []

    ratio_threshold = float(args.ratio_threshold)
    mad_threshold = float(args.mad_threshold)
    max_report = int(args.max_report)

    by_layer = {}
    aux_projections = {}
    skipped_layers = []

    n_down_proj_scanned = 0
    n_o_proj_scanned = 0

    max_ratio = 0.0
    earliest_layer_idx = None
    top_layer = None
    global_entries = []

    def _matches(name, suffixes):
        for suf in suffixes:
            if name.endswith(suf):
                return True
        return False

    for name, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue

        is_primary = _matches(name, target_suffixes)
        is_aux = (not is_primary) and _matches(name, aux_suffixes)
        if not (is_primary or is_aux):
            continue

        # Quantized guard (before any float() cast)
        if v.dtype in _QUANTIZED_DTYPES:
            skipped_layers.append({"layer": name, "reason": "quantized_weight"})
            continue

        if v.dim() != 2:
            skipped_layers.append({"layer": name, "reason": "not_2d"})
            continue

        W = v.float()

        if torch.isnan(W).any().item() or torch.isinf(W).any().item():
            skipped_layers.append({"layer": name, "reason": "nan_or_inf"})
            continue

        # Row-median absolute (robust center), shape (out, 1)
        row_med_abs = W.abs().median(dim=1, keepdim=True).values
        # MAD-based robust z-score from row median
        row_med = W.median(dim=1, keepdim=True).values
        row_mad = (W - row_med).abs().median(dim=1, keepdim=True).values
        mad_scaled = (1.4826 * row_mad).clamp_min(1e-12)
        z = (W - row_med).abs() / mad_scaled
        ratio = W.abs() / row_med_abs.clamp_min(1e-12)
        mask = (ratio >= ratio_threshold) & (z >= mad_threshold)

        if is_primary:
            n_down_proj_scanned += 1
        else:
            n_o_proj_scanned += 1

        indices = mask.nonzero(as_tuple=False)
        if indices.numel() == 0:
            continue

        outliers = []
        for r, c in indices.tolist():
            val = W[r, c].item()
            ratio_val = ratio[r, c].item()
            z_val = z[r, c].item()
            status = "critical" if ratio_val >= 1000.0 else "warning"
            outliers.append({
                "row": r,
                "col": c,
                "value": round(val, 4),
                "ratio_vs_row_median": round(ratio_val, 2),
                "mad_z": round(z_val, 2),
                "status": status,
            })

        # Sort by ratio desc
        outliers.sort(key=lambda o: o["ratio_vs_row_median"], reverse=True)

        if is_primary:
            by_layer[name] = outliers
        else:
            aux_projections[name] = outliers

        # Track global stats
        layer_max_ratio = outliers[0]["ratio_vs_row_median"]
        if layer_max_ratio > max_ratio:
            max_ratio = layer_max_ratio
            top_layer = name

        layer_idx = _extract_layer_idx(name)
        if layer_idx is not None:
            if earliest_layer_idx is None or layer_idx < earliest_layer_idx:
                earliest_layer_idx = layer_idx

        for o in outliers:
            global_entries.append({
                "layer": name,
                "row": o["row"],
                "col": o["col"],
                "value": o["value"],
                "ratio_vs_row_median": o["ratio_vs_row_median"],
            })

    if n_down_proj_scanned == 0 and n_o_proj_scanned == 0:
        emit_error(
            "NoTargetMatricesFound",
            "No FFN down_proj (or attention o_proj with --include-o-proj) matrices were detected in the state_dict.",
            "Expected *.mlp.down_proj.weight or similar. Use 'tree' to inspect available layers.",
        )

    # Global top entries (flat, sorted by ratio desc, capped at max_report)
    global_entries.sort(key=lambda e: e["ratio_vs_row_median"], reverse=True)
    top_entries_global = global_entries[:max_report]

    super_weights_found = sum(len(v) for v in by_layer.values()) \
        + sum(len(v) for v in aux_projections.values())

    scanned_layers = n_down_proj_scanned + n_o_proj_scanned

    summary = {
        "max_ratio": round(max_ratio, 2) if top_layer is not None else 0.0,
        "earliest_layer_idx": earliest_layer_idx,
        "n_down_proj_scanned": n_down_proj_scanned,
        "n_o_proj_scanned": n_o_proj_scanned,
        "top_layer": top_layer,
        "top_entries_global": top_entries_global,
    }

    emit_result({
        "weights": args.weights,
        "ratio_threshold": ratio_threshold,
        "mad_threshold": mad_threshold,
        "include_o_proj": bool(args.include_o_proj),
        "scanned_layers": scanned_layers,
        "super_weights_found": super_weights_found,
        "by_layer": by_layer,
        "aux_projections": aux_projections,
        "skipped_layers": skipped_layers,
        "summary": summary,
        "agent_hint": "Super weights are single scalars whose removal collapses the model. Ratios >= 1000x row-median in early mlp.down_proj layers (typically layers 1-3) are near-certain super weights per Yu et al. (Apple, 2024); pruning or quantizing them destroys generation quality.",
    })


# --- TRANSFORMER ANALYSIS HELPERS ---

def _add_analysis_args(parser):
    """Add shared arguments for transformer analysis commands."""
    parser.add_argument("--script", required=True, help="Python file containing the Model class.")
    parser.add_argument("--model-class", required=True, help="Name of the PyTorch Module class.")
    parser.add_argument("--weights", required=True, help="Path to the .pt weights file.")
    parser.add_argument("--input-shape", default=None, help="Shape of dummy input, e.g. 1,128 (not needed if --text is used).")
    parser.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer name (e.g. 'gpt2'). Required with --text.")
    parser.add_argument("--text", default=None, help="Input text string. Requires --tokenizer.")
    parser.add_argument("--causal", choices=["auto", "true", "false"], default="auto",
                        help="Causal masking: 'auto' detects from model class, 'true'/'false' forces it.")


def _load_model_for_analysis(args):
    """Shared loader: script -> model class -> weights -> input tensor -> causal detection."""
    if not os.path.exists(args.script):
        emit_error("ScriptNotFound", f"Could not find {args.script}.",
                   "Verify the path to the python script containing the model class.")
    try:
        spec = importlib.util.spec_from_file_location("dynamic_model", args.script)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        ModelClass = getattr(model_module, args.model_class)
    except AttributeError:
        emit_error("ClassNotFound", f"Class '{args.model_class}' not found in {args.script}.",
                   f"Open {args.script} to verify the exact class name.")
    except Exception as e:
        emit_error("ImportError", str(e), "Failed to load the python script. Ensure it has no syntax errors.")

    try:
        model = ModelClass()
        sd = load_weights(args.weights)
        model.load_state_dict(sd)
        model.eval()
    except Exception as e:
        emit_error("ModelInitError", str(e),
                   "The state_dict does not match the model architecture. Use 'tree' to compare layers.")

    # Prepare input
    tokens = None
    if getattr(args, 'text', None) and getattr(args, 'tokenizer', None):
        try:
            from transformers import AutoTokenizer
            tokenizer_obj = AutoTokenizer.from_pretrained(args.tokenizer)
            encoded = tokenizer_obj(args.text, return_tensors="pt")
            input_tensor = encoded["input_ids"]
            tokens = tokenizer_obj.convert_ids_to_tokens(input_tensor[0])
        except ImportError:
            emit_error("DependencyError", "transformers is not installed.",
                       "Run: pip install transformers")
        except Exception as e:
            emit_error("TokenizerError", str(e), "Verify the tokenizer name is a valid HuggingFace model ID.")
    elif getattr(args, 'text', None) and not getattr(args, 'tokenizer', None):
        emit_error("MissingArg", "--text requires --tokenizer.",
                   "Add --tokenizer <hf_model_name>, e.g. --tokenizer gpt2")
    elif getattr(args, 'input_shape', None):
        try:
            shape = [int(x.strip()) for x in args.input_shape.split(',')]
        except ValueError:
            emit_error("InvalidShape", f"Cannot parse '{args.input_shape}'.",
                       "Provide comma-separated integers, e.g. '1,128'.")
        input_tensor = torch.randn(*shape)
    else:
        emit_error("MissingInput", "No input specified.",
                   "Provide either --input-shape 1,128 for random input, or --text 'Hello world' --tokenizer gpt2 for real text.")

    # Causal detection
    causal_arg = getattr(args, 'causal', 'auto')
    if causal_arg == 'true':
        is_causal = True
    elif causal_arg == 'false':
        is_causal = False
    else:
        is_causal = _auto_detect_causal(model)

    return model, input_tensor, tokens, is_causal


def _auto_detect_causal(model):
    """Detect if a model is causal/decoder from class name or config."""
    name = type(model).__name__.lower()
    causal_keywords = ['causal', 'gpt', 'decoder', 'llama', 'mistral', 'opt',
                       'bloom', 'falcon', 'phi', 'gemma', 'qwen', 'mamba']
    if any(k in name for k in causal_keywords):
        return True
    if hasattr(model, 'config'):
        if getattr(model.config, 'is_decoder', False):
            return True
        if getattr(model.config, 'is_causal', False):
            return True
    return False


def _apply_causal_mask(attn_weights):
    """Zero the upper triangle and renormalize. Works on (..., seq, seq) tensors."""
    seq_len = attn_weights.shape[-1]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=attn_weights.dtype,
                                  device=attn_weights.device))
    masked = attn_weights * mask
    row_sums = masked.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return masked / row_sums


def _is_attention_shaped(t):
    """Check if tensor looks like post-softmax attention weights."""
    if t.dim() < 3 or t.shape[-1] != t.shape[-2] or t.shape[-1] <= 1:
        return False
    if t.min().item() < -0.01 or t.max().item() > 1.01:
        return False
    row_sums = t.sum(dim=-1)
    if (row_sums - 1.0).abs().max().item() > 0.15:
        return False
    return True


def _run_forward_safe(model, input_tensor):
    """Try multiple calling conventions for the forward pass."""
    try:
        return model(input_tensor)
    except Exception:
        pass
    try:
        return model(input_ids=input_tensor)
    except Exception:
        pass
    try:
        return model(input_tensor, attention_mask=torch.ones_like(input_tensor))
    except Exception as e:
        emit_error("ForwardPassError", str(e),
                   "The forward pass crashed. Check --input-shape or model's expected input format.")


def _head_metrics(attn_2d, tokens=None):
    """Compute all metrics for a single attention head. attn_2d shape: (seq, seq)."""
    seq_len = attn_2d.shape[0]

    # Entropy per row, then average
    eps = 1e-10
    log_attn = torch.log(attn_2d + eps)
    row_entropy = -(attn_2d * log_attn).sum(dim=-1)
    max_entropy = math.log(seq_len) if seq_len > 1 else 1.0
    avg_entropy = row_entropy.mean().item()
    normalized_entropy = avg_entropy / max_entropy

    # Diagonality: mean attention on exact diagonal
    diag_vals = torch.diag(attn_2d)
    diagonality = diag_vals.mean().item()

    # Local diagonality: mean attention within +-1 of diagonal (vectorized)
    indices = torch.arange(seq_len)
    row_idx = indices.unsqueeze(1).expand(seq_len, seq_len)
    col_idx = indices.unsqueeze(0).expand(seq_len, seq_len)
    local_mask = (row_idx - col_idx).abs() <= 1
    local_diagonality = attn_2d[local_mask].mean().item()

    # Verticality: does one key position attract most of the attention?
    col_sums = attn_2d.sum(dim=0)
    col_sums_norm = col_sums / seq_len
    max_col_attn = col_sums_norm.max().item()
    vertical_idx = col_sums_norm.argmax().item()

    # Top attention pairs
    flat = attn_2d.flatten()
    k = min(10, flat.numel())
    topk_vals, topk_idx = torch.topk(flat, k)
    top_pairs = []
    for val, idx in zip(topk_vals, topk_idx):
        row = idx.item() // seq_len
        col = idx.item() % seq_len
        pair = {"from": row, "to": col, "weight": round(val.item(), 4)}
        if tokens:
            if row < len(tokens):
                pair["from_token"] = tokens[row]
            if col < len(tokens):
                pair["to_token"] = tokens[col]
        top_pairs.append(pair)

    # Pattern classification
    pattern = "diffuse"
    if normalized_entropy > 0.95:
        pattern = "uniform"
    elif diagonality > 0.4:
        pattern = "diagonal"
    elif local_diagonality > 0.6:
        pattern = "local"
    elif max_col_attn > 0.5:
        pattern = "vertical"
    elif normalized_entropy < 0.3:
        pattern = "sparse"

    return {
        "entropy": round(avg_entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "diagonality": round(diagonality, 4),
        "local_diagonality": round(local_diagonality, 4),
        "verticality": round(max_col_attn, 4),
        "vertical_target_position": vertical_idx,
        "vertical_target_token": tokens[vertical_idx] if tokens and vertical_idx < len(tokens) else None,
        "pattern": pattern,
        "top_pairs": top_pairs,
    }


def _generate_attention_html(attn_2d, labels, module_name, head_idx, layer_idx, metrics, is_causal):
    """Generate a self-contained HTML heatmap."""
    seq_len = attn_2d.shape[0]

    def color(val):
        v = max(0.0, min(1.0, val))
        r = int(255 * (1 - v * 0.8))
        g = int(255 * (1 - v * 0.9))
        b = 255
        return f"rgb({r},{g},{b})"

    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows_html = "<tr><td style='min-width:80px'></td>"
    for j in range(seq_len):
        lbl = esc(labels[j] if j < len(labels) else f"p{j}")
        rows_html += f"<td class='col-label'>{lbl}</td>"
    rows_html += "</tr>\n"

    for i in range(seq_len):
        lbl = esc(labels[i] if i < len(labels) else f"p{i}")
        row = f"<tr><td class='row-label'>{lbl}</td>"
        for j in range(seq_len):
            val = attn_2d[i, j].item()
            bg = color(val)
            tc = "white" if val > 0.5 else "black"
            row += f"<td style='background:{bg};color:{tc}'>{val:.2f}</td>"
        row += "</tr>\n"
        rows_html += row

    causal_note = " (causal mask applied)" if is_causal else ""
    top_pairs_html = ""
    for p in metrics["top_pairs"][:5]:
        fl = p.get("from_token", f"pos_{p['from']}")
        tl = p.get("to_token", f"pos_{p['to']}")
        top_pairs_html += f"<li>{esc(str(fl))} &rarr; {esc(str(tl))}: {p['weight']:.4f}</li>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Attention: Layer {layer_idx} Head {head_idx}</title>
<style>
  body {{ font-family: monospace; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h2 {{ color: #00d2ff; }}
  .info {{ margin: 10px 0; padding: 10px; background: #16213e; border-radius: 6px; }}
  .info span {{ margin-right: 20px; }}
  .val {{ color: #ff6b6b; font-weight: bold; }}
  table {{ border-collapse: collapse; margin: 20px 0; }}
  td {{ width: 40px; height: 28px; text-align: center; font-size: 9px; border: 1px solid #2a2a4a; }}
  .row-label, .col-label {{ font-weight: bold; background: #16213e; color: #00d2ff; font-size: 10px; min-width: 60px; }}
  .col-label {{ writing-mode: vertical-rl; text-orientation: mixed; height: 60px; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ padding: 3px 0; }}
</style>
</head>
<body>
<h2>Attention Heatmap: Layer {layer_idx} ({esc(module_name)}), Head {head_idx}{causal_note}</h2>
<div class="info">
  <span>Pattern: <span class="val">{metrics['pattern']}</span></span>
  <span>Entropy: <span class="val">{metrics['normalized_entropy']:.4f}</span></span>
  <span>Diagonality: <span class="val">{metrics['diagonality']:.4f}</span></span>
  <span>Verticality: <span class="val">{metrics['verticality']:.4f}</span></span>
</div>
<div class="info">
  <strong>Top attention pairs:</strong>
  <ul>{top_pairs_html}</ul>
</div>
<table>
{rows_html}
</table>
</body>
</html>"""


# --- TRANSFORMER ANALYSIS COMMANDS ---

def cmd_residual_stream(args):
    model, input_tensor, tokens, is_causal = _load_model_for_analysis(args)

    hidden_states = []
    seen_ptrs = set()

    def make_hook(name):
        def hook(module, inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(t, torch.Tensor) or t.dim() != 3:
                return
            ptr = t.data_ptr()
            if ptr in seen_ptrs:
                return
            seen_ptrs.add(ptr)
            hidden_states.append((name, t.detach().cpu().float()))
        return hook

    handles = []
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _run_forward_safe(model, input_tensor)

    for h in handles:
        h.remove()

    if not hidden_states:
        emit_error("NoHiddenStates", "No 3D hidden states captured during forward pass.",
                   "The model may not produce standard (batch, seq, hidden) outputs. Verify the model architecture.")

    stream = []
    first_state = hidden_states[0][1]

    for i, (name, state) in enumerate(hidden_states):
        entry = {
            "position": i,
            "module": name,
            "shape": list(state.shape),
            "l2_norm": round(torch.norm(state).item(), 4),
            "mean_magnitude": round(state.abs().mean().item(), 6),
            "std": round(state.std().item(), 6),
            "per_position_variance": round(state.var(dim=-1).mean().item(), 6),
        }

        if i > 0:
            prev = hidden_states[i - 1][1]
            # Only compute cosine/update when shapes match (same hidden dim)
            if state.shape == prev.shape:
                cos = torch.nn.functional.cosine_similarity(
                    state.flatten().unsqueeze(0), prev.flatten().unsqueeze(0)).item()
                entry["cosine_sim_to_previous"] = round(cos, 6)

                update = state - prev
                update_norm = torch.norm(update).item()
                state_norm = torch.norm(state).item()
                entry["update_norm"] = round(update_norm, 4)
                entry["update_ratio"] = round(update_norm / (state_norm + 1e-9), 6)

            if state.shape == first_state.shape:
                cos_first = torch.nn.functional.cosine_similarity(
                    state.flatten().unsqueeze(0), first_state.flatten().unsqueeze(0)).item()
                entry["cosine_sim_to_input"] = round(cos_first, 6)

        stream.append(entry)

    # Anomaly detection
    anomalies = []
    norms = [s["l2_norm"] for s in stream]
    if len(norms) > 1:
        ratio = norms[-1] / (norms[0] + 1e-9)
        if ratio > 100:
            anomalies.append(f"Norm explosion: {norms[0]:.2f} -> {norms[-1]:.2f} ({ratio:.0f}x growth).")
        elif ratio < 0.01:
            anomalies.append(f"Norm collapse: {norms[0]:.2f} -> {norms[-1]:.2f}.")

    for entry in stream:
        if entry.get("cosine_sim_to_previous", 0) > 0.9999:
            anomalies.append(f"Dead layer: '{entry['module']}' did not modify the residual stream (cosine=1.0).")
        if entry["per_position_variance"] < 1e-8:
            anomalies.append(f"Collapsed variance at '{entry['module']}'. Representation may be a constant vector.")

    emit_result({
        "states_captured": len(stream),
        "causal_model_detected": is_causal,
        "input_tokens": tokens,
        "stream": stream,
        "anomalies": anomalies,
        "agent_hint": "Track l2_norm for explosion/collapse. cosine_sim_to_previous~1.0 = dead layer. "
                      "Decreasing per_position_variance = representation collapse. "
                      "update_ratio shows how much each layer changes the stream (healthy: 0.01-0.5)."
    })


def cmd_attention(args):
    model, input_tensor, tokens, is_causal = _load_model_for_analysis(args)

    captured = {}
    order = [0]

    def make_hook(name):
        def hook(module, inp, output):
            candidates = []
            if isinstance(output, (tuple, list)):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        candidates.append(o)
            elif isinstance(output, torch.Tensor):
                candidates.append(output)
            for t in candidates:
                if _is_attention_shaped(t):
                    captured[f"{order[0]:03d}_{name}"] = t.detach().cpu().float()
                    order[0] += 1
        return hook

    handles = []
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _run_forward_safe(model, input_tensor)

    for h in handles:
        h.remove()

    if not captured:
        emit_error("NoAttentionCaptured",
                   "No post-softmax attention weight tensors found during forward pass.",
                   "The model must expose attention weights (post-softmax, shape [batch, heads, seq, seq]) "
                   "as part of its forward() output. Many custom models don't do this by default. "
                   "Modify the model to return attn_weights from the attention module.")

    layers_analysis = {}
    all_head_summaries = []

    for key in sorted(captured.keys()):
        attn = captured[key]
        name = key.split("_", 1)[1]

        if is_causal:
            attn = _apply_causal_mask(attn)

        if attn.dim() == 3:
            attn = attn.unsqueeze(1)

        batch, n_heads, seq_len, _ = attn.shape
        heads = {}

        for h in range(n_heads):
            head_attn = attn[0, h]
            metrics = _head_metrics(head_attn, tokens)
            heads[f"head_{h}"] = metrics
            all_head_summaries.append({"layer": name, "head": h, **metrics})

        head_entropies = [heads[f"head_{h}"]["normalized_entropy"] for h in range(n_heads)]
        dead = sum(1 for e in head_entropies if e > 0.95)
        patterns = {}
        for h in range(n_heads):
            p = heads[f"head_{h}"]["pattern"]
            patterns[p] = patterns.get(p, 0) + 1

        layers_analysis[name] = {
            "n_heads": n_heads,
            "seq_len": seq_len,
            "heads": heads,
            "dead_heads": dead,
            "mean_normalized_entropy": round(sum(head_entropies) / len(head_entropies), 4),
            "pattern_distribution": patterns,
        }

    total_heads = len(all_head_summaries)
    dead_total = sum(1 for m in all_head_summaries if m["normalized_entropy"] > 0.95)
    global_patterns = {}
    for m in all_head_summaries:
        p = m["pattern"]
        global_patterns[p] = global_patterns.get(p, 0) + 1

    emit_result({
        "causal_masking_applied": is_causal,
        "input_tokens": tokens,
        "attention_layers_found": len(layers_analysis),
        "layers": layers_analysis,
        "summary": {
            "total_heads": total_heads,
            "dead_heads": dead_total,
            "dead_head_pct": round(dead_total / max(total_heads, 1) * 100, 1),
            "pattern_distribution": global_patterns,
        },
        "agent_hint": "Dead heads (normalized_entropy>0.95) waste capacity. "
                      "'vertical' = position sink (BOS/CLS). 'diagonal' = self/local attention. "
                      "'sparse' = selective (usually good). "
                      "Use 'attention-plot --layer-idx N --head-idx M' to visualize specific heads."
    })


def cmd_attention_plot(args):
    model, input_tensor, tokens, is_causal = _load_model_for_analysis(args)

    captured = []

    def make_hook(name):
        def hook(module, inp, output):
            candidates = output if isinstance(output, (tuple, list)) else [output]
            for t in candidates:
                if isinstance(t, torch.Tensor) and _is_attention_shaped(t):
                    captured.append((name, t.detach().cpu().float()))
        return hook

    handles = []
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _run_forward_safe(model, input_tensor)

    for h in handles:
        h.remove()

    if not captured:
        emit_error("NoAttentionCaptured", "No attention weights found.",
                   "Model must expose post-softmax attention weights in forward().")

    layer_idx = args.layer_idx
    if layer_idx >= len(captured):
        emit_error("InvalidLayer",
                   f"Layer index {layer_idx} out of range (found {len(captured)} attention layers).",
                   f"Use --layer-idx between 0 and {len(captured) - 1}. "
                   f"Run 'attention' first to see all available layers.")

    name, attn = captured[layer_idx]

    if is_causal:
        attn = _apply_causal_mask(attn)

    if attn.dim() == 3:
        attn = attn.unsqueeze(1)

    n_heads = attn.shape[1]
    head_idx = args.head_idx

    if head_idx >= n_heads:
        emit_error("InvalidHead",
                   f"Head index {head_idx} out of range ({n_heads} heads available).",
                   f"Use --head-idx between 0 and {n_heads - 1}.")

    head_attn = attn[0, head_idx]
    seq_len = head_attn.shape[0]
    labels = tokens if tokens else [f"pos_{i}" for i in range(seq_len)]
    metrics = _head_metrics(head_attn, tokens)

    html = _generate_attention_html(head_attn, labels, name, head_idx, layer_idx, metrics, is_causal)

    out_dir = os.path.dirname(os.path.abspath(args.weights))
    out_path = os.path.join(out_dir, f"attention_L{layer_idx}_H{head_idx}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    emit_result({
        "html_file": os.path.abspath(out_path),
        "layer_module": name,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "seq_len": seq_len,
        "causal_masking_applied": is_causal,
        "tokens_used": labels,
        "metrics": metrics,
        "agent_hint": f"HTML heatmap saved to '{os.path.abspath(out_path)}'. "
                      f"Tell the user to open it in a browser. "
                      f"The metrics above are the machine-readable equivalent."
    })


# --- LLM ACTIVATION DIAGNOSTICS ---

def cmd_massive_activations(args):
    model, input_tensor, tokens, is_causal = _load_model_for_analysis(args)

    # Determine hidden_size to filter residual-stream-shaped tensors
    hidden_size = None
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            hidden_size = emb.weight.shape[1]
    except Exception:
        pass  # fall through; use hook-time filtering only

    captured = {}  # name -> tensor (B, T, D)

    def make_hook(name):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(t, torch.Tensor):
                return
            if t.dim() != 3:
                return
            D = t.shape[-1]
            if D < 16:
                return
            if hidden_size is not None and D != hidden_size:
                return
            captured[name] = t.detach().float().cpu()
        return hook

    # Hook only block-looking modules first; fall back to all modules if none match.
    BLOCK_SUFFIXES = ("Block", "DecoderLayer", "Layer", "TransformerBlock")
    block_modules = [(n, m) for n, m in model.named_modules()
                     if n and type(m).__name__.endswith(BLOCK_SUFFIXES)]

    handles = []
    if block_modules:
        for n, m in block_modules:
            handles.append(m.register_forward_hook(make_hook(n)))
    else:
        for n, m in model.named_modules():
            if n:
                handles.append(m.register_forward_hook(make_hook(n)))

    with torch.no_grad():
        _run_forward_safe(model, input_tensor)

    for h in handles:
        h.remove()

    if not captured:
        emit_error("NoResidualStreamCaptured",
                   "No 3D activations with dim[-1]==hidden_size were captured during forward pass.",
                   "The model may not expose a standard residual stream. Try running 'residual-stream' first to confirm hook visibility.")

    abs_thr = args.abs_threshold
    ratio_thr = args.ratio_threshold
    top_k = args.top_k

    layers_out = {}
    total_massive = 0
    max_ratio_global = 0.0
    first_layer_with_massive = None
    last_layer_with_massive = None
    n_flagged = 0

    for name in sorted(captured.keys()):
        h = captured[name]
        a = h.abs()
        max_abs = a.max().item()
        med_abs = a.median().item()
        ratio = max_abs / (med_abs + 1e-9)
        mask_count = ((a >= abs_thr) & (a >= ratio_thr * med_abs)).sum().item()
        num_massive = int(mask_count)

        # Top-K outliers by absolute value
        B, T, D = h.shape
        flat_vals = h.flatten()
        flat_abs = a.flatten()
        k = min(top_k, flat_abs.numel())
        vals, idxs = torch.topk(flat_abs, k=k)

        top_outliers = []
        for v_abs, flat_idx in zip(vals.tolist(), idxs.tolist()):
            if v_abs < abs_thr:
                break  # nothing more worth reporting
            rem = flat_idx % (T * D)
            t_pos = rem // D
            d_idx = rem % D
            top_outliers.append({
                "token": int(t_pos),
                "feature": int(d_idx),
                "value": round(float(flat_vals[flat_idx].item()), 4),
                "abs_value": round(float(v_abs), 4),
            })

        status = "ok"
        if num_massive > 0:
            status = "critical"
            if first_layer_with_massive is None:
                first_layer_with_massive = name
            last_layer_with_massive = name
            total_massive += num_massive
            n_flagged += 1

        if ratio > max_ratio_global:
            max_ratio_global = ratio

        layers_out[name] = {
            "max_abs": round(max_abs, 4),
            "median_abs": round(med_abs, 6),
            "ratio": round(ratio, 2),
            "num_massive": num_massive,
            "top_outliers": top_outliers,
            "status": status,
        }

    emit_result({
        "input_shape": list(input_tensor.shape),
        "causal_masking_applied": is_causal,
        "abs_threshold": abs_thr,
        "ratio_threshold": ratio_thr,
        "layers_hooked": len(captured),
        "layers": layers_out,
        "summary": {
            "n_layers": len(layers_out),
            "n_flagged": n_flagged,
            "total_massive_scalars": total_massive,
            "max_ratio_global": round(max_ratio_global, 2),
            "first_layer_with_massive": first_layer_with_massive,
            "last_layer_with_massive": last_layer_with_massive,
        },
        "agent_hint": "Massive activations (|a|>=100 AND |a|>=1000*median) concentrated on specific tokens/features across mid-to-late layers are EXPECTED in healthy decoder LLMs (Sun et al. 2024). Their absence in mid/late layers or sudden disappearance between checkpoints indicates training pathology or aggressive quantization damage."
    })


def _dormant_strip_suffix(name, suffixes):
    for s in suffixes:
        if name.endswith(s):
            return name[: -len(s)]
    return None


def _dormant_attn_parent(attn_name):
    stripped = _dormant_strip_suffix(attn_name, _DORMANT_ATTN_SUFFIXES)
    return stripped if stripped is not None else attn_name


def _dormant_match_v(parent, attn_name, v_store):
    if parent in v_store:
        return v_store[parent]
    if attn_name in v_store:
        return v_store[attn_name]
    for vk, vt in v_store.items():
        if parent and (parent.startswith(vk) or vk.startswith(parent)):
            return vt
    return None


def _dormant_head_split_v(V, H):
    """Reshape (B, T, D_v) -> (B, H, T, d_head), handling GQA/MQA.

    Returns (V_h, n_kv_heads) or (None, None) on failure.
    """
    Bv, Tv, Dv = V.shape
    if H > 0 and Dv % H == 0:
        d_head = Dv // H
        V_h = V.view(Bv, Tv, H, d_head).permute(0, 2, 1, 3).contiguous()
        return V_h, H
    for h_try in (H // 2, H // 4, H // 8, 1):
        if h_try is None or h_try <= 0:
            continue
        if Dv % h_try == 0 and (Dv // h_try) in _DORMANT_DHEAD_CANDIDATES:
            d_head = Dv // h_try
            V_h = V.view(Bv, Tv, h_try, d_head).permute(0, 2, 1, 3).contiguous()
            if H % h_try == 0:
                V_h = V_h.repeat_interleave(H // h_try, dim=1)
                return V_h, h_try
    return None, None


def _dormant_layer_status(dormant_count, n_heads):
    if n_heads <= 0:
        return "ok"
    frac = dormant_count / n_heads
    if frac > 0.3:
        return "critical"
    if dormant_count > 0:
        return "warning"
    return "ok"


def _dormant_weights_only(args):
    sd = load_weights(args.weights)
    threshold = args.dormancy_threshold
    num_heads_override = args.num_heads

    v_entries = {}
    o_entries = {}
    for key, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() != 2:
            continue
        if not key.endswith(".weight"):
            continue
        base = key[: -len(".weight")]
        vp = _dormant_strip_suffix(base, _DORMANT_V_SUFFIXES)
        if vp is not None:
            v_entries[vp] = tensor.detach().float()
            continue
        op = _dormant_strip_suffix(base, _DORMANT_O_SUFFIXES)
        if op is not None:
            o_entries[op] = tensor.detach().float()

    if not v_entries:
        emit_error("NoAttentionCaptured",
                   "No v_proj-like weight tensors found in the state_dict.",
                   "Provide a checkpoint containing attention value/output projections, "
                   "or run without --weights-only to use forward-pass activations.")

    layers_out = {}
    total_heads = 0
    total_dormant = 0
    layer_medians = []
    max_dormant_pct = 0.0

    for parent, W_V in sorted(v_entries.items()):
        W_O = o_entries.get(parent)
        D_out_v, D_in_v = W_V.shape
        H = num_heads_override
        if H is None:
            for h_try in (32, 16, 12, 8, 4, 2, 1):
                if D_out_v % h_try == 0 and (D_out_v // h_try) in _DORMANT_DHEAD_CANDIDATES:
                    H = h_try
                    break
        if H is None or D_out_v % H != 0:
            layers_out[parent] = {
                "status": "error",
                "reason": f"Cannot infer num_heads for W_V shape {list(W_V.shape)}; pass --num-heads.",
            }
            continue

        d_head = D_out_v // H
        W_V_heads = W_V.view(H, d_head, D_in_v)
        v_norms = W_V_heads.reshape(H, -1).norm(dim=-1)

        if W_O is not None and W_O.dim() == 2 and W_O.shape[1] == H * d_head:
            W_O_heads = W_O.view(W_O.shape[0], H, d_head).permute(1, 0, 2).contiguous()
            o_norms = W_O_heads.reshape(H, -1).norm(dim=-1)
            head_scores = v_norms * o_norms
            projection_mode = "post_W_O"
        else:
            head_scores = v_norms
            projection_mode = "pre_W_O"

        layer_median = head_scores.median().item()
        heads = {}
        if layer_median <= 1e-9:
            for h in range(H):
                heads[f"head_{h}"] = {
                    "weight_score": round(head_scores[h].item(), 4),
                    "relative_to_median": None,
                    "is_dormant": False,
                }
            dormant_count = 0
            status_layer = "warning"
        else:
            relative = head_scores / layer_median
            is_dormant = relative < threshold
            for h in range(H):
                heads[f"head_{h}"] = {
                    "weight_score": round(head_scores[h].item(), 4),
                    "relative_to_median": round(relative[h].item(), 4),
                    "is_dormant": bool(is_dormant[h].item()),
                }
            dormant_count = int(is_dormant.sum().item())
            status_layer = _dormant_layer_status(dormant_count, H)

        total_heads += H
        total_dormant += dormant_count
        layer_medians.append(layer_median)
        max_dormant_pct = max(max_dormant_pct, dormant_count / max(H, 1) * 100.0)

        layers_out[parent] = {
            "n_heads": H,
            "projection": projection_mode,
            "layer_median_score": round(layer_median, 4),
            "heads": heads,
            "dormant_heads": dormant_count,
            "status": status_layer,
        }

    mean_median = (sum(layer_medians) / len(layer_medians)) if layer_medians else 0.0
    dormant_pct = (total_dormant / max(total_heads, 1)) * 100.0

    emit_result({
        "mode": "weights_only",
        "dormancy_threshold": threshold,
        "layers": layers_out,
        "summary": {
            "n_layers": len(layers_out),
            "total_heads": total_heads,
            "dormant_heads": total_dormant,
            "dormant_pct": round(dormant_pct, 1),
            "mean_layer_median_score": round(mean_median, 4),
            "max_dormant_pct_any_layer": round(max_dormant_pct, 1),
        },
        "agent_hint": "Weights-only proxy: per-head score is ||W_V^(h)||_F * ||W_O^(h)||_F, flagged dormant below dormancy_threshold x layer median (Sanyal et al. 2025). Healthy pretrained LLMs show 8-15% dormant heads; >30% in a layer suggests under-trained or collapsed attention and safe-to-ablate candidates. For activation-based confirmation, rerun without --weights-only."
    })


def cmd_dormant_heads(args):
    if args.weights_only:
        _dormant_weights_only(args)
        return

    if not args.script or not args.model_class:
        emit_error("MissingArgument",
                   "Activation mode requires --script and --model-class.",
                   "Either provide --script and --model-class for a forward pass, "
                   "or pass --weights-only to use the weights-only proxy instead.")

    model, input_tensor, tokens, is_causal = _load_model_for_analysis(args)

    attn_store = {}
    v_store = {}
    w_o_store = {}

    def make_attn_hook(name):
        def hook(module, inp, out):
            tensors = out if isinstance(out, (tuple, list)) else [out]
            for t in tensors:
                if isinstance(t, torch.Tensor) and _is_attention_shaped(t):
                    attn_store[name] = t.detach().cpu().float()
                    return
        return hook

    def make_v_hook(parent):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor) and t.dim() == 3:
                v_store[parent] = t.detach().cpu().float()
        return hook

    def make_qkv_hook(parent):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor) and t.dim() == 3:
                D = t.shape[-1]
                if D % 3 == 0:
                    v_store[parent] = t[..., -(D // 3):].detach().cpu().float()
        return hook

    handles = []
    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(make_attn_hook(name)))
        vp = _dormant_strip_suffix(name, _DORMANT_V_SUFFIXES)
        if vp is not None:
            handles.append(module.register_forward_hook(make_v_hook(vp)))
        fusedp = _dormant_strip_suffix(name, _DORMANT_QKV_FUSED)
        if fusedp is not None:
            handles.append(module.register_forward_hook(make_qkv_hook(fusedp)))
        op = _dormant_strip_suffix(name, _DORMANT_O_SUFFIXES)
        if op is not None and hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.dim() == 2:
            w_o_store[op] = module.weight.detach().cpu().float()

    with torch.no_grad():
        _run_forward_safe(model, input_tensor)

    for h in handles:
        h.remove()

    if not attn_store:
        emit_error("NoAttentionCaptured",
                   "No post-softmax attention weight tensors found during forward pass.",
                   "The model must expose attention weights (post-softmax, shape [batch, heads, seq, seq]) "
                   "as part of its forward() output. Rerun with --weights-only for a weights-based proxy.")

    layers_out = {}
    total_heads = 0
    total_dormant = 0
    layer_medians = []
    max_dormant_pct = 0.0
    threshold = args.dormancy_threshold

    for attn_name in sorted(attn_store.keys()):
        A = attn_store[attn_name]
        parent = _dormant_attn_parent(attn_name)
        V = _dormant_match_v(parent, attn_name, v_store)
        W_O = w_o_store.get(parent) or w_o_store.get(attn_name)

        if is_causal:
            A = _apply_causal_mask(A)
        if A.dim() == 3:
            A = A.unsqueeze(1)
        B, H, T, _ = A.shape

        if V is None:
            layers_out[parent] = {
                "n_heads": H,
                "status": "fallback_weights_only",
                "reason": "V tensor not identified for this layer",
            }
            continue

        V_h, n_kv_heads = _dormant_head_split_v(V, H)
        if V_h is None:
            layers_out[parent] = {
                "n_heads": H,
                "status": "error",
                "reason": f"V shape incompatible with H={H} (D_v={V.shape[-1]})",
            }
            continue

        d_head = V_h.shape[-1]
        head_out = torch.matmul(A, V_h)  # (B, H, T, d_head)

        projection_mode = "pre_W_O"
        if W_O is not None and W_O.dim() == 2 and W_O.shape[1] == H * d_head:
            W_O_h = W_O.view(W_O.shape[0], H, d_head)
            head_proj = torch.einsum("bhtd,ohd->bhto", head_out, W_O_h)
            norm_per_token = head_proj.norm(dim=-1)  # (B, H, T)
            projection_mode = "post_W_O"
        else:
            norm_per_token = head_out.norm(dim=-1)  # (B, H, T)

        head_norm = norm_per_token.mean(dim=(0, 2))  # (H,)
        layer_median = head_norm.median().item()
        heads = {}

        if layer_median <= 1e-9:
            for h in range(H):
                heads[f"head_{h}"] = {
                    "output_norm": round(head_norm[h].item(), 4),
                    "relative_to_median": None,
                    "is_dormant": False,
                }
            dormant_count = 0
            status_layer = "warning"
        else:
            relative = head_norm / layer_median
            is_dormant = relative < threshold
            for h in range(H):
                heads[f"head_{h}"] = {
                    "output_norm": round(head_norm[h].item(), 4),
                    "relative_to_median": round(relative[h].item(), 4),
                    "is_dormant": bool(is_dormant[h].item()),
                }
            dormant_count = int(is_dormant.sum().item())
            status_layer = _dormant_layer_status(dormant_count, H)

        total_heads += H
        total_dormant += dormant_count
        layer_medians.append(layer_median)
        max_dormant_pct = max(max_dormant_pct, dormant_count / max(H, 1) * 100.0)

        layers_out[parent] = {
            "n_heads": H,
            "n_kv_heads": n_kv_heads,
            "projection": projection_mode,
            "layer_median_norm": round(layer_median, 4),
            "heads": heads,
            "dormant_heads": dormant_count,
            "status": status_layer,
        }

    mean_median = (sum(layer_medians) / len(layer_medians)) if layer_medians else 0.0
    dormant_pct = (total_dormant / max(total_heads, 1)) * 100.0

    emit_result({
        "causal_masking_applied": is_causal,
        "mode": "activation",
        "dormancy_threshold": threshold,
        "input_tokens": tokens,
        "attention_layers_found": len(attn_store),
        "layers": layers_out,
        "summary": {
            "n_layers": len(layers_out),
            "total_heads": total_heads,
            "dormant_heads": total_dormant,
            "dormant_pct": round(dormant_pct, 1),
            "mean_layer_median_norm": round(mean_median, 4),
            "max_dormant_pct_any_layer": round(max_dormant_pct, 1),
        },
        "agent_hint": "A head is dormant when its contribution norm (||A@V@W_O||) falls below dormancy_threshold x the layer median (Sanyal et al. 2025). Healthy pretrained LLMs show 8-15% dormant heads; >30% in a layer suggests under-trained or collapsed attention and safe-to-ablate candidates. Layers with status 'fallback_weights_only' had no matchable V; rerun with --weights-only for a complete per-layer weights-based picture."
    })


# --- COMPOSABLE PRIMITIVES ---

def cmd_slice(args):
    sd = load_weights(args.weights)
    tensor = get_layer(sd, args.layer).float()

    try:
        slices = []
        for part in args.index.split(","):
            part = part.strip()
            if ":" in part:
                pieces = part.split(":")
                start = int(pieces[0]) if pieces[0] else None
                end = int(pieces[1]) if pieces[1] else None
                slices.append(slice(start, end))
            else:
                slices.append(int(part))
        result = tensor[tuple(slices)]
    except Exception as e:
        emit_error("SliceError", str(e),
                   f"Use numpy-style indexing: '0:10,5:15' or ':,3' or '0'. Shape is {list(tensor.shape)}.")

    if isinstance(result, torch.Tensor) and result.numel() > 1:
        stats = _compute_layer_stats(result)
        emit_result({
            "layer": args.layer,
            "original_shape": list(tensor.shape),
            "slice_spec": args.index,
            "result_shape": list(result.shape) if result.dim() > 0 else [1],
            "result_numel": result.numel(),
            "stats": stats,
            "agent_hint": "Use slice to isolate sub-regions: specific neurons, head projections, or feature groups. "
                          "Chain with 'topk' or 'histogram' for deeper analysis of the slice."
        })
    else:
        emit_result({
            "layer": args.layer,
            "slice_spec": args.index,
            "scalar_value": result.item() if isinstance(result, torch.Tensor) else float(result),
        })


def cmd_topk(args):
    sd = load_weights(args.weights)
    tensor = get_layer(sd, args.layer).float()
    flat = tensor.flatten()

    k = min(args.k, flat.numel())

    values, indices_flat = torch.topk(flat, k, largest=not args.smallest)

    entries = []
    for val, idx in zip(values, indices_flat):
        multi_idx = []
        remaining = idx.item()
        for dim_size in reversed(tensor.shape):
            multi_idx.insert(0, remaining % dim_size)
            remaining //= dim_size
        entries.append({
            "value": round(val.item(), 6),
            "flat_index": idx.item(),
            "index": multi_idx,
        })

    emit_result({
        "layer": args.layer,
        "shape": list(tensor.shape),
        "k": k,
        "mode": "smallest" if args.smallest else "largest",
        "entries": entries,
        "agent_hint": "Use topk to find outlier weights, strongest connections, or most biased neurons. "
                      "Combine with 'slice' to inspect regions around outliers."
    })


def cmd_cosine(args):
    sd1 = load_weights(args.weights1)
    t1 = get_layer(sd1, args.layer1).float().flatten()

    sd2 = load_weights(args.weights2)
    t2 = get_layer(sd2, args.layer2).float().flatten()

    if t1.numel() != t2.numel():
        emit_error("SizeMismatch",
                   f"Cannot compute cosine: {args.layer1} has {t1.numel()} elements, {args.layer2} has {t2.numel()}.",
                   "Cosine similarity requires same total elements. Use 'slice' to extract matching sub-tensors first.")

    cosine_sim = torch.nn.functional.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
    dot_product = torch.dot(t1, t2).item()

    interpretation = (
        "identical" if cosine_sim > 0.9999 else
        "very similar" if cosine_sim > 0.95 else
        "similar" if cosine_sim > 0.8 else
        "somewhat related" if cosine_sim > 0.5 else
        "weakly related" if cosine_sim > 0.0 else
        "orthogonal" if abs(cosine_sim) < 0.05 else
        "opposing"
    )

    emit_result({
        "layer1": f"{args.weights1}:{args.layer1}",
        "layer2": f"{args.weights2}:{args.layer2}",
        "cosine_similarity": round(cosine_sim, 6),
        "dot_product": round(dot_product, 6),
        "l2_norm_1": round(torch.norm(t1).item(), 6),
        "l2_norm_2": round(torch.norm(t2).item(), 6),
        "interpretation": interpretation,
        "agent_hint": "Cosine~1 = redundant/identical layers. Cosine~0 = independent. Negative = opposing. "
                      "Compare same layer across checkpoints to measure training progress. "
                      "Compare different heads to detect redundancy."
    })


def cmd_reduce(args):
    sd = load_weights(args.weights)
    tensor = get_layer(sd, args.layer).float()

    dim = args.dim
    if dim >= tensor.dim() or dim < -tensor.dim():
        emit_error("InvalidDim", f"Dimension {dim} invalid for tensor with shape {list(tensor.shape)}.",
                   f"Use a dimension between 0 and {tensor.dim() - 1}.")

    ops = {
        "mean": lambda t, d: t.mean(dim=d),
        "sum": lambda t, d: t.sum(dim=d),
        "max": lambda t, d: t.max(dim=d).values,
        "min": lambda t, d: t.min(dim=d).values,
        "norm": lambda t, d: torch.norm(t, p=2, dim=d),
        "var": lambda t, d: t.var(dim=d),
        "absmax": lambda t, d: t.abs().max(dim=d).values,
    }

    if args.op not in ops:
        emit_error("InvalidOp", f"Unknown operation '{args.op}'.",
                   f"Supported: {', '.join(ops.keys())}")

    result = ops[args.op](tensor, dim)
    stats = _compute_layer_stats(result)

    emit_result({
        "layer": args.layer,
        "original_shape": list(tensor.shape),
        "operation": args.op,
        "reduced_dim": dim,
        "result_shape": list(result.shape),
        "stats": stats,
        "agent_hint": "reduce dim=0 on weights = per-input-feature summary. "
                      "reduce dim=1 = per-output-neuron summary. "
                      "Chain: reduce -> topk to find strongest/weakest neurons."
    })


def cmd_matmul(args):
    sd = load_weights(args.weights)
    t1 = get_layer(sd, args.layer1).float()
    t2 = get_layer(sd, args.layer2).float()

    if args.transpose1:
        t1 = t1.t() if t1.dim() == 2 else t1.transpose(-2, -1)
    if args.transpose2:
        t2 = t2.t() if t2.dim() == 2 else t2.transpose(-2, -1)

    if t1.dim() != 2 or t2.dim() != 2:
        emit_error("DimensionError",
                   f"matmul requires 2D tensors after transpose. Got {list(t1.shape)} and {list(t2.shape)}.",
                   "Use --transpose1/--transpose2, or 'slice' to extract 2D sub-tensors first.")

    if t1.shape[1] != t2.shape[0]:
        emit_error("ShapeMismatch",
                   f"Inner dimensions don't match: {list(t1.shape)} @ {list(t2.shape)}.",
                   f"Try --transpose2 for {list(t1.shape)} @ {list(t2.t().shape)}, "
                   f"or --transpose1 for {list(t1.t().shape)} @ {list(t2.shape)}.")

    result = t1 @ t2
    stats = _compute_layer_stats(result)

    try:
        rank = torch.linalg.matrix_rank(result).item()
    except Exception:
        rank = None

    emit_result({
        "layer1": args.layer1 + (" (transposed)" if args.transpose1 else ""),
        "layer2": args.layer2 + (" (transposed)" if args.transpose2 else ""),
        "input_shapes": [list(t1.shape), list(t2.shape)],
        "result_shape": list(result.shape),
        "rank": rank,
        "stats": stats,
        "agent_hint": "Compose weight matrices: Q@K^T for attention logits, W_out@W_value for OV circuits, "
                      "W_up@W_down for MLP effective mapping. Rank reveals effective dimensionality."
    })


# --- INTERPRETIVE ANALYSIS COMMANDS ---

def _detect_block_structure(keys):
    """Auto-detect repeating block prefix and indices from state_dict keys."""
    pattern_counts = {}
    for name in keys:
        match = re.match(r'^(.+?)\.(\d+)\.(.+)$', name)
        if match:
            prefix = match.group(1)
            pattern_counts[prefix] = pattern_counts.get(prefix, 0) + 1
    if not pattern_counts:
        return None, {}
    best_prefix = max(pattern_counts, key=lambda k: pattern_counts[k])
    blocks = {}
    for name in keys:
        match = re.match(r'^' + re.escape(best_prefix) + r'\.(\d+)\.(.+)$', name)
        if match:
            idx = int(match.group(1))
            subkey = match.group(2)
            if idx not in blocks:
                blocks[idx] = {}
            blocks[idx][subkey] = name  # store full key name
    return best_prefix, blocks


def cmd_scalars(args):
    sd = load_weights(args.weights)
    threshold = args.max_numel

    scalars = {}
    for name, v in sd.items():
        if isinstance(v, torch.Tensor) and v.numel() <= threshold:
            scalars[name] = {
                "shape": list(v.shape),
                "numel": v.numel(),
                "values": [round(x, 6) for x in v.float().flatten().tolist()],
                "mean": round(v.float().mean().item(), 6),
            }

    if not scalars:
        emit_result({
            "total_scalar_params": 0,
            "groups": {},
            "agent_hint": "No scalar parameters found. This model does not use learnable gates, scales, or temperatures."
        })

    # Group by common prefix: "zeroskip_params.0" -> group "zeroskip_params"
    groups = {}
    for name in scalars:
        parts = name.rsplit(".", 1)
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0]
        else:
            # Try block-level grouping: "blocks.0.gate" -> "*.gate"
            match = re.match(r'^(.+?)\.(\d+)\.(.+)$', name)
            if match:
                prefix = f"{match.group(1)}.*.{match.group(3)}"
            else:
                prefix = name
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(name)

    group_analysis = {}
    for prefix, names in groups.items():
        names_sorted = sorted(names, key=lambda n: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', n)])
        values = [scalars[n]["mean"] for n in names_sorted]

        # Trend detection
        if len(values) >= 3:
            diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            pos = sum(1 for d in diffs if d > 0)
            neg = sum(1 for d in diffs if d < 0)
            mid = len(values) // 2
            first_half = sum(values[:max(mid, 1)]) / max(mid, 1)
            second_half = sum(values[mid:]) / max(len(values) - mid, 1)
            center = values[mid] if mid < len(values) else values[0]

            if pos >= len(diffs) * 0.7:
                trend = "increasing"
            elif neg >= len(diffs) * 0.7:
                trend = "decreasing"
            elif center < first_half and center < second_half:
                trend = "valley"
            elif center > first_half and center > second_half:
                trend = "peak"
            else:
                trend = "non-monotonic"
        elif len(values) == 2:
            trend = "increasing" if values[1] > values[0] else "decreasing"
        else:
            trend = "single"

        has_sign_change = any(values[i] * values[i + 1] < 0 for i in range(len(values) - 1)) if len(values) > 1 else False
        all_positive = all(v > 0 for v in values)
        all_negative = all(v < 0 for v in values)
        near_zero = all(abs(v) < 0.05 for v in values)
        near_one = all(abs(v - 1.0) < 0.15 for v in values)

        interpretation = []
        if near_one:
            interpretation.append("Near 1.0 -- identity gates/scales, barely moved from init.")
        elif near_zero:
            interpretation.append("Near 0.0 -- these controls are nearly inactive.")
        if has_sign_change:
            interpretation.append("Sign changes across depth -- model selectively amplifies (+) or suppresses (-) blocks.")
        if all_negative and not near_zero:
            interpretation.append("All negative -- model is subtracting these blocks' contributions from the residual stream.")
        if all_positive and not near_zero and not near_one:
            interpretation.append("All positive -- model is actively using all these blocks.")
        if trend == "valley":
            interpretation.append("Valley pattern (high-low-high): middle layers suppressed, model relies on first and last blocks.")
        elif trend == "peak":
            interpretation.append("Peak pattern (low-high-low): middle layers amplified, early/late layers are pass-through.")
        elif trend == "decreasing" and has_sign_change:
            interpretation.append("Decreasing with sign flip: model transitions from using early blocks to suppressing later ones.")

        group_analysis[prefix] = {
            "parameters": names_sorted,
            "values": [round(v, 6) for v in values],
            "count": len(values),
            "range": [round(min(values), 6), round(max(values), 6)],
            "trend": trend,
            "has_sign_change": has_sign_change,
            "interpretation": interpretation,
        }

    emit_result({
        "total_scalar_params": len(scalars),
        "groups_found": len(group_analysis),
        "groups": group_analysis,
        "all_scalars": scalars,
        "agent_hint": "Scalar parameters are the model's opinions. Every learnable scalar is the model voting "
                      "on something: which layers to use, how sharp its attention should be, how much to scale. "
                      "Negative gates = model wants to REMOVE that block's contribution. "
                      "Sign changes across depth = the model is learning depth preferences. "
                      "Valley pattern (pos-neg-pos) often means: 'I need the embedding processing (early) "
                      "and the output preparation (late), but the middle blocks aren't useful yet.' "
                      "This is normal in early training and resolves as middle layers learn features."
    })


def cmd_block_profile(args):
    sd = load_weights(args.weights)
    prefix, block_map = _detect_block_structure(sd.keys())

    if not block_map:
        emit_error("NoBlocks", "Could not detect repeating block structure in layer names.",
                   "This model may not use numbered blocks. Check 'tree' output for the naming pattern.")

    n_blocks = len(block_map)
    profiles = {}
    trends = {"weight_stds": {}, "norm_means": {}, "scalars": {}}

    for idx in sorted(block_map.keys()):
        subkeys = block_map[idx]
        profile = {"weights": {}, "norms": {}, "scalars": {}, "total_params": 0}

        for subkey, full_name in subkeys.items():
            tensor = sd[full_name]
            if not isinstance(tensor, torch.Tensor):
                continue
            t = tensor.float()
            profile["total_params"] += t.numel()

            if t.numel() <= 16:
                val = round(t.mean().item(), 6)
                profile["scalars"][subkey] = val
                if subkey not in trends["scalars"]:
                    trends["scalars"][subkey] = []
                trends["scalars"][subkey].append(val)
            elif ("norm" in subkey or "ln" in subkey or "rmsnorm" in subkey) and t.dim() == 1:
                m = round(t.mean().item(), 4)
                s = round(t.std().item(), 4) if t.numel() > 1 else 0.0
                profile["norms"][subkey] = {"mean": m, "std": s}
                if subkey not in trends["norm_means"]:
                    trends["norm_means"][subkey] = []
                trends["norm_means"][subkey].append(m)
            elif "weight" in subkey and t.dim() >= 2:
                std = round(t.std().item(), 6)
                profile["weights"][subkey] = {
                    "shape": list(t.shape),
                    "std": std,
                    "l2_norm": round(torch.norm(t).item(), 2),
                }
                if subkey not in trends["weight_stds"]:
                    trends["weight_stds"][subkey] = []
                trends["weight_stds"][subkey].append(std)

        profiles[f"{prefix}.{idx}"] = profile

    # Analyze cross-block trends
    trend_analysis = {}
    for category in ["weight_stds", "norm_means", "scalars"]:
        for subkey, vals in trends[category].items():
            if len(vals) < 2:
                continue
            first, last = vals[0], vals[-1]
            change = last - first
            pct_change = (change / abs(first) * 100) if abs(first) > 1e-9 else 0
            direction = "increasing" if change > abs(first) * 0.05 else "decreasing" if change < -abs(first) * 0.05 else "stable"

            trend_analysis[subkey] = {
                "category": category.replace("_", " "),
                "first_block": round(first, 6),
                "last_block": round(last, 6),
                "change_pct": round(pct_change, 1),
                "direction": direction,
                "values": [round(v, 6) for v in vals],
            }

    # Build interpretation
    interpretations = []
    for subkey, t in trend_analysis.items():
        if t["category"] == "weight stds" and t["direction"] == "increasing":
            interpretations.append(f"'{subkey}' std increases with depth ({t['first_block']:.4f} -> {t['last_block']:.4f}): "
                                   f"deeper layers are drifting more from init, receiving stronger gradient signal.")
        if t["category"] == "weight stds" and t["direction"] == "decreasing":
            interpretations.append(f"'{subkey}' std decreases with depth: possible vanishing gradients in later layers.")
        if t["category"] == "norm means" and t["direction"] != "stable":
            interpretations.append(f"'{subkey}' norm mean {t['direction']} across depth ({t['first_block']:.4f} -> {t['last_block']:.4f}): "
                                   f"model is learning different residual stream scales at different depths.")

    emit_result({
        "block_prefix": prefix,
        "n_blocks": n_blocks,
        "profiles": profiles,
        "cross_block_trends": trend_analysis,
        "interpretations": interpretations,
        "agent_hint": "Block profiles reveal how the model differentiates its layers. "
                      "Increasing weight std with depth = stronger gradients at later layers (healthy, close to loss). "
                      "Norm mean != 1.0 = learned scaling correction. "
                      "Compare scalars across blocks to find which blocks the model trusts most. "
                      "Use 'scalars' for deeper analysis of gate/scale parameters. "
                      "Use 'residual-contrib' to see what each block ACTUALLY contributes during a forward pass."
    })


def cmd_residual_contrib(args):
    model, input_tensor, tokens, is_causal = _load_model_for_analysis(args)

    # Detect block-level modules
    block_prefix = getattr(args, 'block_prefix', None)
    block_modules = []

    if block_prefix:
        for name, module in model.named_modules():
            if re.match(r'^' + re.escape(block_prefix) + r'\.\d+$', name):
                block_modules.append((name, module))
    else:
        # Auto-detect: find numbered sequential children
        candidates = {}
        for name, module in model.named_modules():
            match = re.match(r'^(.+?)\.(\d+)$', name)
            if match:
                prefix = match.group(1)
                if prefix not in candidates:
                    candidates[prefix] = []
                candidates[prefix].append((int(match.group(2)), name, module))
        if candidates:
            best = max(candidates, key=lambda k: len(candidates[k]))
            block_modules = [(name, mod) for _, name, mod in sorted(candidates[best])]
            block_prefix = best

    if not block_modules:
        emit_error("NoBlocksDetected",
                   "Could not find block-level modules in the model.",
                   "Specify --block-prefix (e.g., 'blocks' or 'transformer.h') or verify the model has numbered sequential sub-modules.")

    # Hook each block to capture input and output
    block_ios = {}

    def make_io_hook(name):
        def hook(module, inp, output):
            inp_t = inp[0] if isinstance(inp, tuple) else inp
            out_t = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(inp_t, torch.Tensor) and isinstance(out_t, torch.Tensor):
                block_ios[name] = {
                    "input": inp_t.detach().cpu().float(),
                    "output": out_t.detach().cpu().float(),
                }
        return hook

    handles = []
    for name, module in block_modules:
        handles.append(module.register_forward_hook(make_io_hook(name)))

    # Also capture the very first input (pre-blocks) and final output
    first_input = [None]
    final_output = [None]

    def capture_first(module, inp, output):
        if first_input[0] is None:
            out_t = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(out_t, torch.Tensor) and out_t.dim() == 3:
                first_input[0] = out_t.detach().cpu().float()

    # Hook the module right before the first block
    parent_name = block_prefix.rsplit(".", 1)[0] if "." in block_prefix else ""
    for name, module in model.named_modules():
        if name == parent_name or (not parent_name and name == ""):
            continue
        # Hook embedding-like modules (before blocks)
        if any(kw in name.lower() for kw in ["embed", "wte", "token", "input"]):
            handles.append(module.register_forward_hook(capture_first))

    with torch.no_grad():
        _run_forward_safe(model, input_tensor)

    for h in handles:
        h.remove()

    if not block_ios:
        emit_error("NoCaptured", "No block inputs/outputs captured.",
                   "The blocks may not process 3D tensors, or the block prefix is wrong.")

    # Analyze each block's contribution
    contributions = []
    cumulative_stream = None

    for name, module in block_modules:
        if name not in block_ios:
            continue
        io = block_ios[name]
        inp = io["input"]
        out = io["output"]

        if inp.shape != out.shape:
            contributions.append({
                "block": name,
                "note": f"Shape mismatch: input {list(inp.shape)} vs output {list(out.shape)}. Skipped.",
            })
            continue

        # The update is what this block added to the stream
        update = out - inp
        inp_norm = torch.norm(inp).item()
        out_norm = torch.norm(out).item()
        update_norm = torch.norm(update).item()

        # Direction: is the update aligned with the input (reinforcing) or opposing (correcting)?
        cos_update_input = torch.nn.functional.cosine_similarity(
            update.flatten().unsqueeze(0), inp.flatten().unsqueeze(0)).item()

        # Is the output bigger or smaller than the input?
        norm_change = out_norm - inp_norm
        norm_change_pct = (norm_change / inp_norm * 100) if inp_norm > 1e-9 else 0

        # Contribution ratio: how much does this block change the stream?
        contrib_ratio = update_norm / inp_norm if inp_norm > 1e-9 else 0

        # Interpret the block's behavior
        if contrib_ratio < 0.01:
            behavior = "pass-through"
            detail = "Block barely modifies the stream. Near-identity transformation."
        elif cos_update_input > 0.3:
            behavior = "amplifying"
            detail = "Block's update is aligned with input -- reinforcing existing features."
        elif cos_update_input < -0.3:
            behavior = "correcting"
            detail = "Block's update opposes the input -- subtracting or correcting features."
        elif contrib_ratio > 0.5:
            behavior = "transforming"
            detail = "Large orthogonal update -- block is adding substantially new information."
        else:
            behavior = "refining"
            detail = "Moderate orthogonal update -- block adds new features without reinforcing or opposing."

        contributions.append({
            "block": name,
            "input_norm": round(inp_norm, 4),
            "output_norm": round(out_norm, 4),
            "update_norm": round(update_norm, 4),
            "contribution_ratio": round(contrib_ratio, 4),
            "norm_change_pct": round(norm_change_pct, 2),
            "cos_update_vs_input": round(cos_update_input, 4),
            "behavior": behavior,
            "detail": detail,
        })

    # Summary
    behaviors = [c.get("behavior", "?") for c in contributions if "behavior" in c]
    behavior_counts = {}
    for b in behaviors:
        behavior_counts[b] = behavior_counts.get(b, 0) + 1

    contrib_ratios = [c["contribution_ratio"] for c in contributions if "contribution_ratio" in c]
    most_active = max(contributions, key=lambda c: c.get("contribution_ratio", 0)) if contributions else {}
    least_active = min(contributions, key=lambda c: c.get("contribution_ratio", float("inf"))) if contributions else {}

    # Norm trajectory
    norms = [c["output_norm"] for c in contributions if "output_norm" in c]
    if norms:
        norm_first = contributions[0].get("input_norm", 0)
        norm_last = norms[-1]
        norm_growth = (norm_last / norm_first) if norm_first > 0 else 0
    else:
        norm_growth = 0

    emit_result({
        "block_prefix": block_prefix,
        "blocks_analyzed": len(contributions),
        "contributions": contributions,
        "summary": {
            "behavior_distribution": behavior_counts,
            "most_active_block": most_active.get("block"),
            "most_active_ratio": most_active.get("contribution_ratio"),
            "least_active_block": least_active.get("block"),
            "least_active_ratio": least_active.get("contribution_ratio"),
            "stream_norm_growth": round(norm_growth, 4),
        },
        "agent_hint": "Behavior key: "
                      "'pass-through' = block does nothing (dead or untrained). "
                      "'amplifying' = block reinforces existing features in the stream. "
                      "'correcting' = block subtracts/opposes the current stream (removing noise or undoing earlier blocks). "
                      "'transforming' = block adds entirely new information (orthogonal to input). "
                      "'refining' = moderate new information added. "
                      "A healthy trained model has mostly 'refining' and 'transforming' blocks. "
                      "Early training often shows 'pass-through' in middle blocks and 'amplifying' at edges. "
                      "Combine with 'scalars' to see if gate values explain the contribution patterns. "
                      "Combine with 'block-profile' to see if weight statistics correlate with contribution."
    })


# --- CLI SETUP ---

BANNER = r"""
  ___ ____  ___ ____  _____
 |_ _|  _ \|_ _|  _ \| ____|
  | || |_) || || | | |  _|
  | ||  _ < | || |_| | |___
 |___|_| \_\___|____/|_____|
"""

DESCRIPTION = f"""{BANNER}
  Iride  --  The Agent-Computer Interface for PyTorch
  Analyze, debug, and introspect .pt checkpoints from the CLI.
  All output is machine-readable JSON.

QUICK START:
  python iride.py tree model.pt                 List layers
  python iride.py scan model.pt                 Bulk anomaly scan
  python iride.py attention \\
    --script m.py --model-class M \\
    --weights model.pt --input-shape 1,8,128    Attention analysis
"""

EPILOG = """
COMMANDS BY CATEGORY:

  Weight Inspection (static, no forward pass):
    tree            List all layers, shapes, dtypes, and parameter counts
    scan            Bulk stats + anomaly flags for every layer
    stats           Numerical statistics for a single layer
    histogram       Distribution bucketing, percentiles, shape analysis
    sparsity        Dead neuron detection, structured sparsity
    compare-init    Drift from Kaiming/Xavier/LeCun initialization
    svd             Rank, condition number, singular values
    stable-rank     per-layer stable rank, effective rank, spectral entropy
    qk-spectral     per-head Q @ K^T spectral norm; attention divergence risk
    super-weights   outlier parameters in FFN down_proj (Yu et al. 2024)
    diff            Compare a layer between two checkpoints

  Transformer Analysis (dynamic, requires forward pass):
    residual-stream Trace norm growth, cosine drift, dead layers
    attention       Per-head entropy, diagonality, verticality, patterns
    attention-plot  HTML heatmap with optional HF tokenizer labels
    run-forward     Per-layer activation stats (NaN/Inf tracing)
    massive-activations  outlier hidden-state scalars (Sun et al. 2024)
    dormant-heads   per-head output-norm dormancy (Sanyal et al. 2025)

  Interpretive Analysis (what the model is trying to do):
    scalars         Find and interpret all gates, scales, skip weights
    block-profile   Per-block comparison of weights, norms, scalars
    residual-contrib Measure each block's contribution to the residual stream

  Composable Primitives (stackable building blocks):
    slice           Extract sub-tensor by numpy-style index
    topk            Find largest/smallest values with coordinates
    cosine          Cosine similarity between any two layers
    reduce          Aggregate along a dimension (mean/norm/var/...)
    matmul          Multiply two weight matrices, report product rank

EXAMPLES:
  # Bulk scan for anomalies across all layers
  python iride.py scan checkpoint.pt

  # Drill into a flagged layer
  python iride.py histogram checkpoint.pt --layer "fc2.weight" --bins 30
  python iride.py sparsity checkpoint.pt --layer "fc2.weight"

  # Attention analysis with real text and token labels
  python iride.py attention \\
    --script model.py --model-class GPT \\
    --weights gpt.pt --tokenizer gpt2 --text "The cat sat on"

  # Compose QK^T for mechanistic interpretability
  python iride.py matmul model.pt \\
    --layer1 "attn.q.weight" --layer2 "attn.k.weight" --transpose2

  # Track training progress between epochs
  python iride.py diff epoch1.pt epoch5.pt --layer "fc1.weight"
  python iride.py cosine epoch1.pt epoch5.pt \\
    --layer1 "fc1.weight" --layer2 "fc1.weight"

NOTES:
  - All output is strict JSON. Parse "data" on success, "suggested_fix" on error.
  - Transformer commands auto-detect causal models and apply masking.
    Force with --causal true/false if auto-detection is wrong.
  - Primitives are cheap individually but add up. Prefer high-level commands
    (scan, attention) and only drop to primitives when needed.
"""


def _make_sub(subparsers, name, help_text, description=None):
    """Create a subparser with consistent formatting."""
    return subparsers.add_parser(
        name,
        help=help_text,
        description=description or help_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="iride.py",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True,
                                        metavar="COMMAND")

    # ── Weight Inspection ────────────────────────────────────────────

    p = _make_sub(subparsers, "tree",
                  "List all layers and shapes in a .pt file.",
                  "Show every key in the state_dict with its shape, dtype, and\n"
                  "parameter count. Use this first to discover layer names.\n\n"
                  "Example:\n"
                  "  python iride.py tree checkpoint.pt")
    p.add_argument("weights", help="Path to the .pt / .pth file.")

    p = _make_sub(subparsers, "scan",
                  "Bulk stats + anomaly detection on ALL layers at once.",
                  "Computes stats for every tensor and flags anomalies:\n"
                  "  - constant tensors, exploding weights, NaN/Inf\n"
                  "  - near-dead layers, high sparsity\n"
                  "Returns per-layer status (ok / warning / critical).\n\n"
                  "Example:\n"
                  "  python iride.py scan checkpoint.pt")
    p.add_argument("weights", help="Path to the .pt file.")

    p = _make_sub(subparsers, "stats",
                  "Get numerical statistics for a single layer.",
                  "Returns mean, std, min, max, L1/L2 norms, zeros %%, NaN/Inf counts.\n\n"
                  "Example:\n"
                  "  python iride.py stats model.pt --layer \"fc1.weight\"")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--layer", required=True, help="Exact layer name (from 'tree').")

    p = _make_sub(subparsers, "histogram",
                  "Weight distribution: bins, percentiles, and shape analysis.",
                  "Buckets tensor values, computes percentiles (p1-p99), skewness,\n"
                  "kurtosis, and classifies the distribution type:\n"
                  "  normal | degenerate | heavy-tailed | bimodal\n\n"
                  "Examples:\n"
                  "  python iride.py histogram model.pt --layer \"fc1.weight\"\n"
                  "  python iride.py histogram model.pt --layer \"fc1.weight\" --bins 30")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--layer", required=True, help="Exact layer name.")
    p.add_argument("--bins", type=int, default=20, help="Number of histogram bins (default: 20).")

    p = _make_sub(subparsers, "sparsity",
                  "Dead neuron and structured sparsity analysis.",
                  "For 2D weight matrices: reports dead/near-dead rows (output neurons)\n"
                  "and columns (input features), weakest neuron indices by L2 norm,\n"
                  "and per-neuron norm variance.\n"
                  "Also handles 1D biases and N-D conv filters.\n\n"
                  "Examples:\n"
                  "  python iride.py sparsity model.pt --layer \"fc1.weight\"\n"
                  "  python iride.py sparsity model.pt --layer \"conv1.weight\" --threshold 1e-4")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--layer", required=True, help="Exact layer name.")
    p.add_argument("--threshold", type=float, default=1e-5,
                   help="Absolute value below which a weight counts as near-zero (default: 1e-5).")

    p = _make_sub(subparsers, "compare-init",
                  "Compare weight distributions against expected initialization.",
                  "For every weight matrix, computes drift_ratio = actual_std / expected_std\n"
                  "and assigns a status:\n"
                  "  dead | shrunk | low_drift | near_init | trained | high_drift | exploded\n\n"
                  "Examples:\n"
                  "  python iride.py compare-init model.pt\n"
                  "  python iride.py compare-init model.pt --init kaiming")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--init", choices=["auto", "kaiming", "xavier", "lecun"], default="auto",
                   help="Expected initialization (default: auto-detect closest match).")

    p = _make_sub(subparsers, "svd",
                  "Singular Value Decomposition: rank, condition number, singular values.",
                  "Requires a 2D tensor. For Conv2D (4D), use --flatten to reshape\n"
                  "to (out_channels, -1) before decomposition.\n\n"
                  "Examples:\n"
                  "  python iride.py svd model.pt --layer \"fc1.weight\"\n"
                  "  python iride.py svd model.pt --layer \"conv1.weight\" --flatten")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--layer", required=True, help="Exact layer name.")
    p.add_argument("--flatten", action="store_true",
                   help="Reshape N-D tensors to 2D (flatten all dims after the first).")

    p = _make_sub(subparsers, "stable-rank",
                  "Per-layer stable rank, effective rank, spectral entropy.",
                  "Weights-only scanner. For every 2D weight matrix W, computes:\n"
                  "  - stable_rank  = ||W||_F^2 / sigma_max^2  (number of 'loud' dirs)\n"
                  "  - effective_rank = exp(spectral entropy of normalized sigmas)\n"
                  "  - srank_ratio / erank_ratio = normalized by min(shape)\n"
                  "Flags rank-collapsed layers (stable_rank < 3 is critical).\n\n"
                  "Skips 1D tensors, large embeddings (by default), and layers with\n"
                  "min dim below --min-dim. Pass --flatten to include N-D tensors.\n\n"
                  "Examples:\n"
                  "  python iride.py stable-rank model.pt\n"
                  "  python iride.py stable-rank model.pt --flatten --no-skip-embeddings")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--flatten", action="store_true",
                   help="Reshape N-D tensors to 2D (flatten all dims after the first).")
    p.add_argument("--skip-embeddings", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip likely embedding tables (name contains 'embed' and max dim > --embed-threshold). Default: on.")
    p.add_argument("--min-dim", type=int, default=8,
                   help="Skip matrices where min(shape) is below this (default: 8).")
    p.add_argument("--embed-threshold", type=int, default=32768,
                   help="Max-dim threshold above which an 'embed'-named tensor is treated as an embedding table (default: 32768).")

    p = _make_sub(subparsers, "qk-spectral",
                  "Per-head Q@K^T spectral norm; attention divergence risk.",
                  "Weights-only scanner. For every attention layer with Q/K projections\n"
                  "(separate q_proj/k_proj or fused c_attn/qkv_proj), computes per-head\n"
                  "spectral norm sigma_max(W_Q W_K^T) / sqrt(d_head). Values above\n"
                  "--flag-threshold predict attention entropy collapse and loss spikes\n"
                  "(Zhai 2023, Takase 2025, OLMo 2 2025).\n\n"
                  "Handles GQA/MQA (K heads fewer than Q heads) and fused QKV layouts.\n"
                  "num_heads is inferred from d_model when possible; pass --num-heads\n"
                  "if the model uses non-standard d_head values.\n\n"
                  "Examples:\n"
                  "  python iride.py qk-spectral model.pt\n"
                  "  python iride.py qk-spectral model.pt --num-heads 32 --flag-threshold 100")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--num-heads", type=int, default=None,
                   help="Number of attention (Q) heads. If omitted, inferred from d_model (tries d_head in {64,96,128,80,256}).")
    p.add_argument("--flag-threshold", type=float, default=100.0,
                   help="Per-head sigma_max threshold for 'critical' (default: 100.0). Half this value flags 'warning'.")

    p = _make_sub(subparsers, "super-weights",
                  "Outlier FFN parameters: single-scalar super weights (Yu et al. 2024).",
                  "Weights-only scanner for 'super weights' -- individual parameters in\n"
                  "FFN down-projection matrices whose removal catastrophically collapses\n"
                  "model quality (Yu et al., 'The Super Weight in Large Language Models',\n"
                  "ICLR 2025, arXiv 2411.07191).\n\n"
                  "For each row of W, computes:\n"
                  "  - ratio_vs_row_median = |W[r,c]| / median(|W[r,:]|)\n"
                  "  - mad_z = |W[r,c] - median(W[r,:])| / (1.4826 * MAD)\n"
                  "Flags entries where both exceed thresholds. Ratios >= 1000x are\n"
                  "'critical' (near-certain super weights); otherwise 'warning'.\n\n"
                  "Scans *.mlp.down_proj.weight / *.mlp.c_proj.weight / *.feed_forward.w2.weight\n"
                  "and friends. Pass --include-o-proj to also scan attention output\n"
                  "projections.\n\n"
                  "Examples:\n"
                  "  python iride.py super-weights model.pt\n"
                  "  python iride.py super-weights model.pt --include-o-proj\n"
                  "  python iride.py super-weights model.pt --ratio-threshold 200 --mad-threshold 75")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--ratio-threshold", type=float, default=100.0,
                   help="Min |W[r,c]| / row-median-abs ratio to flag (default: 100.0). >=1000 is 'critical'.")
    p.add_argument("--mad-threshold", type=float, default=50.0,
                   help="Min MAD-based robust z-score to flag (default: 50.0).")
    p.add_argument("--max-report", type=int, default=20,
                   help="Max number of top global entries to include in summary (default: 20).")
    p.add_argument("--include-o-proj", action="store_true",
                   help="Also scan attention output projections (*.self_attn.o_proj.weight, *.attn.c_proj.weight, etc.).")

    p = _make_sub(subparsers, "diff",
                  "Compare a layer between two checkpoints.",
                  "Reports L2 distance, max abs difference, cosine similarity.\n"
                  "l2_distance=0 and cosine=1.0 means frozen gradients.\n\n"
                  "Example:\n"
                  "  python iride.py diff epoch1.pt epoch5.pt --layer \"fc1.weight\"")
    p.add_argument("weights1", help="Path to first .pt file.")
    p.add_argument("weights2", help="Path to second .pt file.")
    p.add_argument("--layer", required=True, help="Layer name to compare (must exist in both files).")

    # ── Transformer Analysis ─────────────────────────────────────────

    p = _make_sub(subparsers, "residual-stream",
                  "Trace the residual stream through a transformer forward pass.",
                  "Hooks every module, captures 3D hidden states, and tracks:\n"
                  "  - L2 norm growth across layers\n"
                  "  - cosine similarity between consecutive states\n"
                  "  - update ratio (how much each layer changes the stream)\n"
                  "  - per-position variance (collapse detection)\n\n"
                  "Example:\n"
                  "  python iride.py residual-stream \\\n"
                  "    --script model.py --model-class GPT \\\n"
                  "    --weights gpt.pt --tokenizer gpt2 --text \"Hello world\"")
    _add_analysis_args(p)

    p = _make_sub(subparsers, "attention",
                  "Capture and analyze all attention heads.",
                  "Hooks every module, finds post-softmax attention weight tensors,\n"
                  "and computes per-head metrics:\n"
                  "  - normalized entropy (0=peaked, 1=uniform/dead)\n"
                  "  - diagonality & local diagonality\n"
                  "  - verticality (BOS/CLS attention sink)\n"
                  "  - pattern class: uniform|diagonal|local|vertical|sparse|diffuse\n"
                  "  - top-k attention pairs (with token labels if --tokenizer given)\n\n"
                  "IMPORTANT: The model must return attention weights (post-softmax)\n"
                  "as part of its forward() output tuple.\n\n"
                  "Example:\n"
                  "  python iride.py attention \\\n"
                  "    --script model.py --model-class CausalLM \\\n"
                  "    --weights model.pt --tokenizer gpt2 --text \"The cat sat on\"")
    _add_analysis_args(p)

    p = _make_sub(subparsers, "attention-plot",
                  "Generate an HTML heatmap for a specific attention head.",
                  "Produces a self-contained HTML file (dark theme, color-coded cells)\n"
                  "showing the full attention matrix for one head. Axes are labeled\n"
                  "with token strings when --tokenizer and --text are provided.\n"
                  "Also returns all head metrics as JSON.\n\n"
                  "Run 'attention' first to identify layer/head indices of interest.\n\n"
                  "Example:\n"
                  "  python iride.py attention-plot \\\n"
                  "    --script model.py --model-class CausalLM \\\n"
                  "    --weights model.pt --tokenizer gpt2 --text \"The cat sat\" \\\n"
                  "    --layer-idx 2 --head-idx 3")
    _add_analysis_args(p)
    p.add_argument("--layer-idx", type=int, default=0,
                   help="0-based index of the attention layer to plot (default: 0).")
    p.add_argument("--head-idx", type=int, default=0,
                   help="0-based index of the attention head to plot (default: 0).")

    p = _make_sub(subparsers, "run-forward",
                  "Run a forward pass and capture activation stats per layer.",
                  "Hooks every submodule and records output shape, mean, std, min,\n"
                  "max, and NaN presence. Use when weights look fine but inference\n"
                  "produces NaN or Inf -- the stats show exactly where it breaks.\n\n"
                  "Example:\n"
                  "  python iride.py run-forward \\\n"
                  "    --script model.py --model-class MLP \\\n"
                  "    --weights model.pt --input-shape 1,128")
    p.add_argument("--script", required=True, help="Python file containing the Model class.")
    p.add_argument("--model-class", required=True, help="Name of the nn.Module class.")
    p.add_argument("--weights", required=True, help="Path to the .pt weights file.")
    p.add_argument("--input-shape", required=True,
                   help="Comma-separated input shape, e.g. 1,3,224,224.")

    p = _make_sub(subparsers, "massive-activations",
                  "Detect outlier scalars in the residual stream (Sun et al. 2024).",
                  "Forward-pass detector of 'massive activations' -- individual hidden-state\n"
                  "scalars whose magnitude is orders larger than the median of the tensor\n"
                  "(Sun et al., 'Massive Activations in Large Language Models', COLM 2024,\n"
                  "arXiv 2402.17762). In healthy decoder LLMs these appear on specific\n"
                  "tokens/features in mid-to-late layers and are essential to the model's\n"
                  "behavior. Their absence or disappearance between checkpoints indicates\n"
                  "training pathology or aggressive quantization damage.\n\n"
                  "Hooks block-like modules (Block / DecoderLayer / TransformerBlock) first\n"
                  "and falls back to all modules if none match. Filters hooks to 3D outputs\n"
                  "whose last dim equals the input-embedding hidden size. For each captured\n"
                  "tensor, flags entries where |a| >= --abs-threshold AND |a| >= --ratio-threshold * median(|a|).\n\n"
                  "Examples:\n"
                  "  python iride.py massive-activations \\\n"
                  "    --script model.py --model-class LlamaForCausalLM \\\n"
                  "    --weights model.pt --tokenizer gpt2 --text \"Hello world\"\n"
                  "  python iride.py massive-activations \\\n"
                  "    --script model.py --model-class GPT \\\n"
                  "    --weights model.pt --input-shape 1,128 \\\n"
                  "    --abs-threshold 50 --ratio-threshold 500 --top-k 20")
    _add_analysis_args(p)
    p.add_argument("--abs-threshold", type=float, default=100.0,
                   help="Min absolute value for a scalar to count as 'massive' (default: 100.0).")
    p.add_argument("--ratio-threshold", type=float, default=1000.0,
                   help="Min |a| / median(|a|) ratio for a scalar to count as 'massive' (default: 1000.0).")
    p.add_argument("--top-k", type=int, default=10,
                   help="Max outliers to report per layer (default: 10).")

    p = _make_sub(subparsers, "dormant-heads",
                  "Per-head output-norm dormancy detector (Sanyal et al. 2025).",
                  "Identifies inactive attention heads by the L2 norm of their\n"
                  "per-head output contribution (Sanyal et al., 'Identifying and\n"
                  "Evaluating Inactive Heads in Pretrained LLMs', arXiv 2504.03889, 2025).\n\n"
                  "Default (activation) mode runs a forward pass, captures A and V per\n"
                  "layer, optionally projects through a W_O slice, and computes\n"
                  "head_norm = ||A @ V [@ W_O^(h)]||_2 averaged over batch/tokens. A\n"
                  "head is flagged as dormant when head_norm < --dormancy-threshold x\n"
                  "the layer median. Use --weights-only to skip the forward pass and\n"
                  "use the weights-only proxy ||W_V^(h)||_F * ||W_O^(h)||_F instead.\n\n"
                  "Examples:\n"
                  "  python iride.py dormant-heads \\\n"
                  "    --script model.py --model-class LlamaForCausalLM \\\n"
                  "    --weights model.pt --tokenizer gpt2 --text \"Hello world\"\n"
                  "  python iride.py dormant-heads --weights model.pt --weights-only\n"
                  "  python iride.py dormant-heads --weights model.pt --weights-only \\\n"
                  "    --num-heads 12 --dormancy-threshold 0.05")
    p.add_argument("--script", default=None,
                   help="Python file containing the Model class (not needed with --weights-only).")
    p.add_argument("--model-class", default=None,
                   help="Name of the PyTorch Module class (not needed with --weights-only).")
    p.add_argument("--weights", required=True, help="Path to the .pt weights file.")
    p.add_argument("--input-shape", default=None,
                   help="Shape of dummy input, e.g. 1,128 (not needed if --text is used).")
    p.add_argument("--tokenizer", default=None,
                   help="HuggingFace tokenizer name (e.g. 'gpt2'). Required with --text.")
    p.add_argument("--text", default=None, help="Input text string. Requires --tokenizer.")
    p.add_argument("--causal", choices=["auto", "true", "false"], default="auto",
                   help="Causal masking: 'auto' detects from model class, 'true'/'false' forces it.")
    p.add_argument("--dormancy-threshold", type=float, default=0.1,
                   help="Fraction of the layer median below which a head is flagged as dormant (default: 0.1).")
    p.add_argument("--weights-only", action="store_true",
                   help="Skip the forward pass and use the weights-only proxy ||W_V||_F * ||W_O||_F.")
    p.add_argument("--num-heads", type=int, default=None,
                   help="Override num_heads inference (required with --weights-only when inference fails).")

    # ── Composable Primitives ────────────────────────────────────────

    p = _make_sub(subparsers, "slice",
                  "Extract a sub-tensor by numpy-style index range.",
                  "Returns stats on the extracted region. Supports ranges, single\n"
                  "indices, and open-ended slices.\n\n"
                  "Examples:\n"
                  "  python iride.py slice model.pt --layer \"fc1.weight\" --index \"0:10,5:15\"\n"
                  "  python iride.py slice model.pt --layer \"fc1.weight\" --index \":,3\"")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--layer", required=True, help="Exact layer name.")
    p.add_argument("--index", required=True,
                   help="Numpy-style index: '0:10,5:15', ':,3', '0', etc.")

    p = _make_sub(subparsers, "topk",
                  "Find the top-k largest or smallest values and their indices.",
                  "Returns values with multi-dimensional index coordinates so you\n"
                  "can locate exactly which neuron/connection is the outlier.\n\n"
                  "Examples:\n"
                  "  python iride.py topk model.pt --layer \"fc1.weight\" --k 10\n"
                  "  python iride.py topk model.pt --layer \"fc1.bias\" --k 5 --smallest")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--layer", required=True, help="Exact layer name.")
    p.add_argument("--k", type=int, default=10, help="How many values to return (default: 10).")
    p.add_argument("--smallest", action="store_true",
                   help="Return the smallest values instead of the largest.")

    p = _make_sub(subparsers, "cosine",
                  "Cosine similarity between two layers (same or different files).",
                  "Pass the same file twice to compare layers within one checkpoint,\n"
                  "or two different files to track training progress.\n\n"
                  "Examples:\n"
                  "  python iride.py cosine m.pt m.pt --layer1 \"h.0.weight\" --layer2 \"h.1.weight\"\n"
                  "  python iride.py cosine e1.pt e5.pt --layer1 \"fc.weight\" --layer2 \"fc.weight\"")
    p.add_argument("weights1", help="Path to first .pt file.")
    p.add_argument("weights2", help="Path to second .pt file (can be the same file).")
    p.add_argument("--layer1", required=True, help="Layer name in first file.")
    p.add_argument("--layer2", required=True, help="Layer name in second file.")

    p = _make_sub(subparsers, "reduce",
                  "Reduce a tensor along a dimension (mean, norm, var, ...).",
                  "Collapses one dimension and returns stats on the result.\n"
                  "  dim=0 on weights -> per-input-feature summary\n"
                  "  dim=1 on weights -> per-output-neuron summary\n\n"
                  "Operations: mean | sum | max | min | norm | var | absmax\n\n"
                  "Example:\n"
                  "  python iride.py reduce model.pt --layer \"fc1.weight\" --dim 0 --op norm")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--layer", required=True, help="Exact layer name.")
    p.add_argument("--dim", type=int, required=True, help="Dimension to reduce.")
    p.add_argument("--op", choices=["mean", "sum", "max", "min", "norm", "var", "absmax"],
                   default="mean", help="Reduction operation (default: mean).")

    p = _make_sub(subparsers, "matmul",
                  "Multiply two weight matrices and analyze the product.",
                  "Computes layer1 @ layer2 (with optional transposes), then reports\n"
                  "stats and rank of the result.\n\n"
                  "Use cases:\n"
                  "  Q @ K^T   -> attention logit patterns\n"
                  "  W_O @ W_V -> OV circuits\n"
                  "  W_up @ W_down -> MLP effective mapping\n\n"
                  "Example:\n"
                  "  python iride.py matmul model.pt \\\n"
                  "    --layer1 \"attn.q.weight\" --layer2 \"attn.k.weight\" --transpose2")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--layer1", required=True, help="First matrix layer name.")
    p.add_argument("--layer2", required=True, help="Second matrix layer name.")
    p.add_argument("--transpose1", action="store_true",
                   help="Transpose the first matrix before multiplication.")
    p.add_argument("--transpose2", action="store_true",
                   help="Transpose the second matrix before multiplication.")

    # ── Interpretive Analysis ────────────────────────────────────────

    p = _make_sub(subparsers, "scalars",
                  "Find and interpret all scalar/small learned parameters.",
                  "Discovers gates, scales, temperatures, skip weights, and any other\n"
                  "small (numel <= threshold) learned parameters. Groups them by naming\n"
                  "pattern, detects trends across depth (increasing, valley, sign changes),\n"
                  "and interprets what the model is doing with them.\n\n"
                  "Examples:\n"
                  "  python iride.py scalars model.pt\n"
                  "  python iride.py scalars model.pt --max-numel 32")
    p.add_argument("weights", help="Path to the .pt file.")
    p.add_argument("--max-numel", type=int, default=16,
                   help="Max elements for a param to count as 'scalar' (default: 16).")

    p = _make_sub(subparsers, "block-profile",
                  "Per-block comparison of weights, norms, and scalars across depth.",
                  "Auto-detects the repeating block structure and shows how weight stds,\n"
                  "norm means, and scalar values change from the first block to the last.\n"
                  "Reveals gradient flow, learned scaling, and depth preferences.\n\n"
                  "Example:\n"
                  "  python iride.py block-profile model.pt")
    p.add_argument("weights", help="Path to the .pt file.")

    p = _make_sub(subparsers, "residual-contrib",
                  "Measure each block's actual contribution to the residual stream.",
                  "Runs a forward pass and hooks every block to capture (input, output).\n"
                  "For each block, computes:\n"
                  "  - contribution_ratio: how much the block changes the stream\n"
                  "  - cos_update_vs_input: aligned (amplifying) or opposing (correcting)\n"
                  "  - behavior: pass-through | amplifying | correcting | transforming | refining\n\n"
                  "Requires a model script and weights, same as 'attention'.\n\n"
                  "Example:\n"
                  "  python iride.py residual-contrib \\\n"
                  "    --script model.py --model-class GPT \\\n"
                  "    --weights model.pt --input-shape 1,8,128")
    _add_analysis_args(p)
    p.add_argument("--block-prefix", default=None,
                   help="Module name prefix for blocks (e.g. 'blocks', 'transformer.h'). Auto-detected if omitted.")

    # ── Dispatch ─────────────────────────────────────────────────────

    dispatch = {
        "tree": cmd_tree,
        "scan": cmd_scan,
        "stats": cmd_stats,
        "histogram": cmd_histogram,
        "sparsity": cmd_sparsity,
        "compare-init": cmd_compare_init,
        "svd": cmd_svd,
        "stable-rank": cmd_stable_rank,
        "qk-spectral": cmd_qk_spectral,
        "super-weights": cmd_super_weights,
        "diff": cmd_diff,
        "residual-stream": cmd_residual_stream,
        "attention": cmd_attention,
        "attention-plot": cmd_attention_plot,
        "run-forward": cmd_run_forward,
        "massive-activations": cmd_massive_activations,
        "dormant-heads": cmd_dormant_heads,
        "slice": cmd_slice,
        "topk": cmd_topk,
        "cosine": cmd_cosine,
        "reduce": cmd_reduce,
        "matmul": cmd_matmul,
        "scalars": cmd_scalars,
        "block-profile": cmd_block_profile,
        "residual-contrib": cmd_residual_contrib,
    }

    try:
        args = parser.parse_args()
        handler = dispatch.get(args.command)
        if handler:
            handler(args)
        else:
            parser.print_help()
    except SystemExit:
        pass
    except Exception as e:
        emit_error("UnhandledException", str(e), traceback.format_exc())

if __name__ == "__main__":
    main()