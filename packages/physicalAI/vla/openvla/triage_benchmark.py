#!/usr/bin/env python3
"""VLA latency triage: attribute inference time to CPU launch vs GPU, and
measure how stacking optimizations changes each one's benefit.

For HuggingFace VLA checkpoints that follow the OpenVLA serving contract
(``AutoModelForVision2Seq`` + ``predict_action``), this decomposes latency of
the perception-action loop into GPU busy time (CUDA events) and CPU
launch/dispatch share, records peak memory, and sweeps the optimization
families the way a triage study would:

  * dtype / quantization    (bfloat16, float16, float32, 4-bit, 8-bit)
  * attention implementation (sdpa, flash_attention_2, eager)
  * execution mode          (eager vs CUDA-graph replay of the forward step)

For every pair of families it reports an interaction term - the classic
`savings(A+B) - savings(A) - savings(B)` - so it is visible when one
optimization eats another's benefit instead of stacking additively.
Adapted from the measurement methodology of "Where Is VLA Inference Time
Spent? Bottlenecks and Optimization Interactions Across GPUs"
(arXiv:2611.06710); profiling is done with CUDA events + `perf_counter`
instead of Nsight so it runs inside the stock containers, and the device
context is stamped from `jetson_containers.l4t_version` when a repo checkout
is available (torch-only fallback inside plain containers).

CUDA graphs are captured on the static forward pass; HF `generate()` inside
`predict_action` has data-dependent control flow that can't be captured, so
the graphs axis applies to the `forward` metric (the default for interaction
terms), while end-to-end `action` latency is reported for every config.

Run inside the openvla container, or from a repo checkout on the device:

    python3 packages/physicalAI/vla/openvla/triage_benchmark.py \
        --model openvla/openvla-7b \
        --dtypes bfloat16,float16,4bit --attns sdpa,flash_attention_2 \
        --graphs on --json /data/openvla_triage.json
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

AXES = ('dtype', 'attn', 'graphs')
DTYPES = ('bfloat16', 'float16', 'float32', '4bit', '8bit')
ATTNS = ('sdpa', 'flash_attention_2', 'eager')
DEFAULT_IMAGE = '/data/images/lake.jpg'
DEFAULT_PROMPT = 'In: What action should the robot take to stop?\nOut:'


def axis_value(axis, value):
    """Render a config value for reports ('on' reads better than 'True')."""
    return 'on' if axis == 'graphs' else str(value)


def config_name(cfg):
    """Render a (dtype, attn, use_graphs) config tuple for reports/json keys."""
    return '/'.join((cfg[0], cfg[1], 'graph' if cfg[2] else 'eager'))


def model_load_kwargs(dtype, attn):
    """Map a (dtype, attn) config to `from_pretrained` kwargs as plain data.

    Returned as strings/dicts (not torch objects) so the mapping is usable
    without torch installed; `load_model()` resolves them.
    """
    kwargs = {
        'attn_implementation': attn,
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
    }
    if dtype == '4bit':
        kwargs['quantization_config'] = {
            'load_in_4bit': True,
            'bnb_4bit_compute_dtype': 'torch.bfloat16',
            'bnb_4bit_quant_type': 'nf4',
        }
    elif dtype == '8bit':
        kwargs['quantization_config'] = {'load_in_8bit': True}
    else:
        kwargs['torch_dtype'] = f'torch.{dtype}'
    return kwargs


def select_configs(dtypes, attns, graphs, full):
    """Pick the configs to benchmark.  By default a probe set: the baseline,
    one family changed at a time, plus the pairwise stacks needed for
    interaction terms.  `full` takes the whole cross product instead.
    Configs are (dtype, attn, use_graphs) tuples, baseline first.
    """
    base = (dtypes[0], attns[0], False)
    if full:
        cfgs = [
            (d, a, g)
            for d in dtypes
            for a in attns
            for g in ((False, True) if graphs else (False,))
        ]
        cfgs.sort(key=lambda c: c != base)
        return cfgs
    cfgs = [base]

    def add(cfg):
        if cfg not in cfgs:
            cfgs.append(cfg)

    for d in dtypes[1:]:
        add((d, attns[0], False))
    for a in attns[1:]:
        add((dtypes[0], a, False))
    if graphs:
        add((dtypes[0], attns[0], True))
    if len(dtypes) > 1 and len(attns) > 1:
        add((dtypes[1], attns[1], False))
    if len(dtypes) > 1 and graphs:
        add((dtypes[1], attns[0], True))
    if len(attns) > 1 and graphs:
        add((dtypes[0], attns[1], True))
    return cfgs


def decompose(wall_ms, gpu_ms):
    """Split a wall-clock latency into GPU busy time and CPU launch/dispatch.

    `gpu_ms` comes from CUDA events bracketing the step; whatever wall time
    the GPU was idle for is host-side (kernel launch, preprocessing, python).
    """
    cpu_launch_ms = max(wall_ms - gpu_ms, 0.0)
    return {
        'wall_ms': wall_ms,
        'gpu_ms': gpu_ms,
        'cpu_launch_ms': cpu_launch_ms,
        'cpu_launch_share': cpu_launch_ms / wall_ms if wall_ms > 0 else 0.0,
    }


def interaction_terms(results, metric='forward', baseline=None):
    """Pairwise interaction terms: does applying optimization A change the
    benefit of optimization B?  For every quadruple baseline / +A / +B / +A+B
    present in `results`, interaction = savings(A+B) - savings(A) - savings(B)
    (negative = the optimizations interfere, positive = synergistic).
    """
    walls = {
        cfg: r[metric]['wall_ms']
        for cfg, r in results.items()
        if r.get('status') == 'ok' and metric in r
    }
    if baseline not in walls:
        return []
    base_t = walls[baseline]
    singles = {}
    for cfg in walls:
        diff = [i for i in range(len(AXES)) if cfg[i] != baseline[i]]
        if len(diff) == 1:
            singles.setdefault(diff[0], []).append(cfg)
    terms = []
    for i in range(len(AXES)):
        for j in range(i + 1, len(AXES)):
            for cx in singles.get(i, []):
                for cy in singles.get(j, []):
                    both = list(baseline)
                    both[i], both[j] = cx[i], cy[j]
                    both = tuple(both)
                    if both not in walls:
                        continue
                    savings = {
                        'a_ms': base_t - walls[cx],
                        'b_ms': base_t - walls[cy],
                        'ab_ms': base_t - walls[both],
                    }
                    interaction_ms = (
                        savings['ab_ms'] - savings['a_ms'] - savings['b_ms']
                    )
                    pct = 100.0 * interaction_ms / base_t if base_t > 0 else 0.0
                    verdict = (
                        'synergistic'
                        if pct > 2
                        else 'interfering' if pct < -2 else 'additive'
                    )
                    terms.append(
                        {
                            'families': [AXES[i], AXES[j]],
                            'variants': [cx[i], cy[j]],
                            'stacked': list(both),
                            'savings': savings,
                            'interaction_ms': interaction_ms,
                            'interaction_pct': pct,
                            'verdict': verdict,
                        }
                    )
    return terms


def triage_hints(results, metric='forward', baseline=None):
    """Rule-based reading of the measurements: which bottleneck dominates,
    which family pays off first, and where stacking stops being additive."""
    base = results.get(baseline, {}).get(metric)
    if not base:
        return ['no successful baseline measurement']
    hints = []
    if base['cpu_launch_share'] > 0.30:
        hints.append(
            f"CPU launch/dispatch is {base['cpu_launch_share']:.0%} of {metric} latency"
            ' - cut host overhead / use CUDA graphs before kernel-side work'
        )
    else:
        hints.append(
            f"GPU-bound: {1 - base['cpu_launch_share']:.0%} of {metric} latency is GPU"
            ' busy time - kernel efficiency and memory bandwidth dominate'
        )
    if base['wall_ms'] > 0:
        for i, axis in enumerate(AXES):
            for cfg, res in results.items():
                if res.get('status') != 'ok' or metric not in res:
                    continue
                changed = [j for j in range(len(AXES)) if cfg[j] != baseline[j]]
                if changed != [i]:
                    continue
                savings_pct = 100.0 * (1 - res[metric]['wall_ms'] / base['wall_ms'])
                if savings_pct > 5:
                    hints.append(
                        f'{axis}={axis_value(axis, cfg[i])}:'
                        f' -{savings_pct:.0f}% {metric} wall time on its own'
                    )
    return hints


def device_context():
    """Collect the hardware/software context for the report.  Prefers the
    repo's own `jetson_containers.l4t_version` (L4T/JetPack/CUDA/arch, the
    canonical version variables used across the build system) when importable,
    and always records what torch can see, so reports from different GPUs can
    be diffed.
    """
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
    )
    repo_dir = os.path.join(repo_root, 'jetson_containers')
    if os.path.isdir(repo_dir) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    ctx = {'python': sys.version.split()[0]}
    try:
        import torch

        props = torch.cuda.get_device_properties(0)
        ctx['gpu'] = props.name
        ctx['gpu_capability'] = f'{props.major}.{props.minor}'
        ctx['gpu_memory_gb'] = round(props.total_memory / 1e9, 1)
        ctx['torch'] = torch.__version__
    except Exception as err:
        ctx['gpu_error'] = str(err)
    try:
        from jetson_containers import l4t_version

        ctx['platform'] = l4t_version.CUDA_ARCH
        ctx['l4t'] = str(l4t_version.get_l4t_version())
        ctx['jetpack'] = str(l4t_version.get_jetpack_version())
        ctx['cuda'] = str(l4t_version.get_cuda_version())
        ctx['cuda_arch'] = str(l4t_version.get_cuda_arch(format=str))
    except Exception as err:
        ctx['l4t_error'] = str(err)
    return ctx


def _device_index(device):
    """Extract the numeric device index from e.g. 'cuda:0' (default 0)."""
    try:
        return int(device.split(':')[1])
    except (IndexError, ValueError):
        return 0


def load_model(model_name, dtype, attn, device):
    """Load the VLA for a (dtype, attn) config; heavy imports stay lazy so the
    module itself imports anywhere (e.g. in repo unit tests without torch)."""
    import torch

    from transformers import AutoModelForVision2Seq, AutoProcessor

    kwargs = model_load_kwargs(dtype, attn)
    quantization_config = kwargs.pop('quantization_config', None)
    if quantization_config is not None:
        from transformers import BitsAndBytesConfig

        for key, value in quantization_config.items():
            if isinstance(value, str) and value.startswith('torch.'):
                quantization_config[key] = getattr(torch, value.split('.', 1)[1])
        kwargs['quantization_config'] = BitsAndBytesConfig(**quantization_config)
        # bitsandbytes weights can't be .to()-moved after load - place at load
        kwargs['device_map'] = {'': _device_index(device)}
    elif kwargs.get('torch_dtype', '').startswith('torch.'):
        kwargs['torch_dtype'] = getattr(torch, kwargs['torch_dtype'].split('.', 1)[1])
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)
    if 'device_map' not in kwargs:
        vla = vla.to(device)
    return vla, processor


def load_image(image_path):
    """Load the benchmark image, falling back to a synthetic frame so the
    timing loop stays runnable on hosts without the sample image (timing is
    unaffected; only the tokenized content differs)."""
    from PIL import Image

    if os.path.isfile(image_path):
        return Image.open(image_path).convert('RGB')
    return Image.new('RGB', (256, 256), (128, 128, 128))


def measure_step(step, iters):
    """Run `step` iters times and decompose latency via CUDA events +
    `perf_counter` (median over the iters); also capture peak device memory."""
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_ms, gpu_ms = [], []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        step()
        end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        gpu_ms.append(start.elapsed_time(end))
    metrics = decompose(statistics.median(wall_ms), statistics.median(gpu_ms))
    metrics['peak_mem_mb'] = torch.cuda.max_memory_allocated() / 1e6
    return metrics


def capture_forward_graph(model, inputs):
    """Capture the static forward pass into a CUDA graph and return a replay
    closure.  Follows the warmup-on-side-stream protocol from the CUDA graphs
    docs; raises if the model's forward can't be captured (the caller records
    it as an 'unsupported' triage outcome)."""
    import torch

    static_inputs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(**static_inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model(**static_inputs)
    return lambda: graph.replay()


def benchmark_config(model_name, dtype, attn, use_graphs, args):
    """Load one config and measure the action step (end-to-end contract, same
    as the other VLA benchmarks) and the forward step (static, where CUDA
    graphs applies)."""
    import torch

    vla, processor = load_model(model_name, dtype, attn, args.device)
    image = load_image(args.image)

    model_dtype = getattr(vla, 'dtype', None)

    def make_inputs():
        return processor(args.prompt, image).to(args.device, dtype=model_dtype)

    def action_step():
        return vla.predict_action(
            **make_inputs(), unnorm_key=args.unnorm_key, do_sample=False
        )

    for _ in range(args.warmup):
        action_step()
    result = {'status': 'ok', 'action': measure_step(action_step, args.iters)}
    result['action_sample'] = [
        float(x) for x in torch.as_tensor(action_step()).flatten().tolist()
    ]
    inputs = make_inputs()

    result['forward'] = measure_step(lambda: vla(**inputs), args.iters)
    result['graphs'] = 'not requested'
    if use_graphs:
        try:
            replay = capture_forward_graph(vla, inputs)
            for _ in range(args.warmup):
                replay()
            result['forward'] = measure_step(replay, args.iters)
            result['graphs'] = 'ok'
        except Exception as err:
            result['graphs'] = f'unsupported: {err}'
    # the model reference dies with this frame; collect before emptying the
    # cuda cache so the next 7B load starts from a clean allocator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def print_report(results, interactions, hints, args, ctx):
    baseline = (args.dtypes[0], args.attns[0], False)
    base_wall = None
    if baseline in results and results[baseline].get('status') == 'ok':
        base_wall = results[baseline][args.metric]['wall_ms']
    print(f"\n== VLA latency triage: {args.model} ==")
    print(
        f"   device: {ctx.get('gpu', '?')}  l4t={ctx.get('l4t', '?')}"
        f"  jetpack={ctx.get('jetpack', '?')}  cuda={ctx.get('cuda', '?')}"
        f"  platform={ctx.get('platform', '?')}"
    )
    header = (
        f"{'config':<38} {'wall_ms':>8} {'gpu_ms':>8} {'cpu_ms':>8}"
        f" {'cpu%':>6} {'peakMB':>8} {'vs_base':>8}"
    )
    print(f"\n-- per-config ({args.metric} metric) --\n{header}")
    for cfg, res in results.items():
        name = config_name(cfg)
        if res.get('status') != 'ok':
            print(f"{name:<38} FAILED: {res.get('error', '?')[:60]}")
            continue
        m = res[args.metric]
        vs_base = ''
        if base_wall:
            vs_base = f"{100.0 * (m['wall_ms'] / base_wall - 1):>+7.1f}%"
        print(
            f"{name:<38} {m['wall_ms']:>8.2f} {m['gpu_ms']:>8.2f}"
            f" {m['cpu_launch_ms']:>8.2f} {m['cpu_launch_share']:>6.0%}"
            f" {m.get('peak_mem_mb', 0):>8.0f} {vs_base:>8}"
        )
        if res.get('graphs', '').startswith('unsupported'):
            print(f"{'':<38} cuda-graphs {res['graphs']}")
    base_name = '/'.join(baseline[:2]) + '/eager'
    print(f"\n-- interaction terms ({args.metric}, baseline {base_name}) --")
    for term in interactions:
        variants = ' x '.join(
            axis_value(family, value)
            for family, value in zip(term['families'], term['variants'])
        )
        print(
            f"   {' x '.join(term['families'])} ({variants}):"
            f" stacked {term['savings']['ab_ms']:+.1f} ms vs"
            f" {term['savings']['a_ms']:+.1f} + {term['savings']['b_ms']:+.1f} ms"
            f" -> interaction {term['interaction_ms']:+.1f} ms"
            f" ({term['interaction_pct']:+.1f}%, {term['verdict']})"
        )
    if not interactions:
        print('   (none - run the default probe set or --full to get pairwise stacks)')
    print('\n-- triage hints --')
    for hint in hints:
        print(f'   * {hint}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', type=str, default='openvla/openvla-7b')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE)
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT)
    parser.add_argument('--unnorm-key', type=str, default='bridge_orig')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        '--dtypes',
        type=str,
        default='bfloat16,float16,4bit',
        help=f'comma list from {DTYPES}; first entry is the baseline',
    )
    parser.add_argument(
        '--attns',
        type=str,
        default='sdpa,flash_attention_2',
        help=f'comma list from {ATTNS}; first entry is the baseline',
    )
    parser.add_argument(
        '--graphs',
        type=str,
        default='on',
        choices=['on', 'off'],
        help='also benchmark CUDA-graph replay of the forward step',
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='benchmark the full cross product instead of the probe set',
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='forward',
        choices=['forward', 'action'],
        help='latency used for interaction terms (forward is where graphs applies)',
    )
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument(
        '--json', type=str, default=None, help='write results json here'
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.dtypes = [d.strip() for d in args.dtypes.split(',')]
    args.attns = [a.strip() for a in args.attns.split(',')]
    for d in args.dtypes:
        if d not in DTYPES:
            raise SystemExit(f"unknown dtype '{d}' (choose from {DTYPES})")
    for a in args.attns:
        if a not in ATTNS:
            raise SystemExit(f"unknown attn '{a}' (choose from {ATTNS})")

    ctx = device_context()
    configs = select_configs(args.dtypes, args.attns, args.graphs == 'on', args.full)
    print(f"triaging {len(configs)} configs on {ctx.get('gpu', args.device)}:")
    for cfg in configs:
        print(f"   {config_name(cfg)}")

    results = {}
    for cfg in configs:
        name = config_name(cfg)
        print(f"\nbenchmarking {name} ...")
        try:
            results[cfg] = benchmark_config(args.model, cfg[0], cfg[1], cfg[2], args)
        except Exception as err:
            results[cfg] = {'status': 'failed', 'error': str(err)}

    baseline = (args.dtypes[0], args.attns[0], False)
    interactions = interaction_terms(results, args.metric, baseline)
    hints = triage_hints(results, args.metric, baseline)
    print_report(results, interactions, hints, args, ctx)

    if args.json:
        with open(args.json, 'w') as file:
            json.dump(
                {
                    'device': ctx,
                    'model': args.model,
                    'metric': args.metric,
                    'baseline': list(baseline),
                    'configs': {config_name(cfg): r for cfg, r in results.items()},
                    'interactions': interactions,
                    'hints': hints,
                },
                file,
                indent=2,
            )
        print(f"\nwrote {args.json}")


if __name__ == '__main__':
    main()
