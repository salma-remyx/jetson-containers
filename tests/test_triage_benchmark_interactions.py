import os
import sys
import unittest
from unittest.mock import patch

# make the repo root importable regardless of the cwd the test is invoked from,
# then import jetson_containers so the `packages.*` meta-path finder is active
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jetson_containers  # noqa: F401  (registers the packages.* finder)
import packages.physicalAI.vla.openvla.triage_benchmark as triage
from jetson_containers import l4t_version


class TestTriageBenchmarkImport(unittest.TestCase):
    def test_importable_through_packages_finder(self):
        # the benchmark must load through the repo's cross-package import
        # machinery (same path config.py files use), without torch installed
        for func in ("main", "device_context", "select_configs", "interaction_terms"):
            self.assertTrue(callable(getattr(triage, func)), func)


class TestDeviceContext(unittest.TestCase):
    """The report stamps device identity from jetson_containers.l4t_version -
    the repo's canonical L4T/JetPack/CUDA source - so results from different
    GPUs (Orin, Thor, x86) can be diffed."""

    def test_device_context_stamps_l4t_version(self):
        with (
            patch.object(l4t_version, "get_l4t_version", return_value="36.4.0"),
            patch.object(l4t_version, "get_jetpack_version", return_value="6.2"),
            patch.object(l4t_version, "get_cuda_version", return_value="12.6"),
            patch.object(l4t_version, "get_cuda_arch", return_value="8.7"),
            patch.object(l4t_version, "CUDA_ARCH", "tegra-aarch64"),
        ):
            ctx = triage.device_context()
        self.assertEqual(ctx["l4t"], "36.4.0")
        self.assertEqual(ctx["jetpack"], "6.2")
        self.assertEqual(ctx["cuda"], "12.6")
        self.assertEqual(ctx["cuda_arch"], "8.7")
        self.assertEqual(ctx["platform"], "tegra-aarch64")

    def test_device_context_survives_missing_jetson_containers(self):
        # inside plain containers the jetson_containers package is absent -
        # the report must still be produced (torch-only fallback)
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("jetson_containers"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=blocked):
            ctx = triage.device_context()
        self.assertNotIn("l4t", ctx)
        self.assertIn(ctx.get("l4t_error", ""), str(ctx))


class TestInteractionTerms(unittest.TestCase):
    BASELINE = ("bfloat16", "sdpa", False)

    @staticmethod
    def result(wall_ms):
        return {"status": "ok", "forward": {"wall_ms": wall_ms, "gpu_ms": wall_ms}}

    def test_interfering_stack(self):
        # dtype saves 20ms alone, attn 10ms alone, but stacked only 25ms:
        # interaction = 25 - 20 - 10 = -5ms (one eats the other's benefit)
        results = {
            self.BASELINE: self.result(100.0),
            ("float16", "sdpa", False): self.result(80.0),
            ("bfloat16", "flash_attention_2", False): self.result(90.0),
            ("float16", "flash_attention_2", False): self.result(75.0),
        }
        terms = triage.interaction_terms(results, "forward", self.BASELINE)
        self.assertEqual(len(terms), 1)
        term = terms[0]
        self.assertEqual(term["families"], ["dtype", "attn"])
        self.assertAlmostEqual(term["interaction_ms"], -5.0)
        self.assertAlmostEqual(term["interaction_pct"], -5.0)
        self.assertEqual(term["verdict"], "interfering")

    def test_additive_stack(self):
        results = {
            self.BASELINE: self.result(100.0),
            ("float16", "sdpa", False): self.result(80.0),
            ("bfloat16", "flash_attention_2", False): self.result(90.0),
            ("float16", "flash_attention_2", False): self.result(70.0),
        }
        terms = triage.interaction_terms(results, "forward", self.BASELINE)
        self.assertEqual(terms[0]["verdict"], "additive")
        self.assertAlmostEqual(terms[0]["interaction_ms"], 0.0)

    def test_missing_corner_yields_no_term(self):
        results = {
            self.BASELINE: self.result(100.0),
            ("float16", "sdpa", False): self.result(80.0),
            ("bfloat16", "flash_attention_2", False): self.result(90.0),
        }
        self.assertEqual(
            triage.interaction_terms(results, "forward", self.BASELINE), []
        )

    def test_failed_configs_ignored(self):
        results = {
            self.BASELINE: self.result(100.0),
            ("float16", "sdpa", False): {"status": "failed", "error": "OOM"},
        }
        self.assertEqual(
            triage.interaction_terms(results, "forward", self.BASELINE), []
        )


class TestConfigSelection(unittest.TestCase):
    def test_probe_set_covers_singles_and_pair_stacks(self):
        configs = triage.select_configs(
            ["bfloat16", "float16", "4bit"], ["sdpa", "flash_attention_2"], True, False
        )
        self.assertEqual(configs[0], ("bfloat16", "sdpa", False))  # baseline first
        self.assertEqual(len(configs), 8)
        for stacked in [
            ("float16", "flash_attention_2", False),
            ("float16", "sdpa", True),
            ("bfloat16", "flash_attention_2", True),
        ]:
            self.assertIn(stacked, configs)

    def test_full_cross_product(self):
        configs = triage.select_configs(
            ["bfloat16", "float16", "4bit"], ["sdpa", "flash_attention_2"], True, True
        )
        self.assertEqual(len(configs), 12)
        self.assertEqual(configs[0], ("bfloat16", "sdpa", False))


class TestHelpers(unittest.TestCase):
    def test_decompose_attributes_cpu_launch(self):
        m = triage.decompose(10.0, 7.0)
        self.assertAlmostEqual(m["cpu_launch_ms"], 3.0)
        self.assertAlmostEqual(m["cpu_launch_share"], 0.3)
        clamped = triage.decompose(5.0, 9.0)  # noisy timers: gpu > wall
        self.assertEqual(clamped["cpu_launch_ms"], 0.0)

    def test_model_load_kwargs_are_portable(self):
        quantized = triage.model_load_kwargs("4bit", "eager")
        self.assertEqual(quantized["quantization_config"]["load_in_4bit"], True)
        self.assertEqual(
            quantized["quantization_config"]["bnb_4bit_compute_dtype"], "torch.bfloat16"
        )
        self.assertNotIn("torch_dtype", quantized)
        plain = triage.model_load_kwargs("bfloat16", "flash_attention_2")
        self.assertEqual(plain["torch_dtype"], "torch.bfloat16")
        self.assertEqual(plain["attn_implementation"], "flash_attention_2")


if __name__ == "__main__":
    unittest.main()
