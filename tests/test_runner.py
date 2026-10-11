"""Verify source fidelity, notebook semantics, routing, and terminal execution."""

import ast
import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiment_runner import JournalExperimentRunner, _NotebookScope, _RuntimePaths


class TestJournalRunner(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def setUp(self):
        self.runner = JournalExperimentRunner(self.ROOT)

    @staticmethod
    def write_liar2(directory):
        rows = []
        labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
        for i in range(120):
            rows.append({"statement": f"Policy claim number {i} about {'legacy budget' if i < 84 else 'recent climate'}",
                         "label": labels[i % 6], "date": str((pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)).date()),
                         "subject": "budget" if i < 84 else "climate", "speaker": f"speaker{i % 4}",
                         "speaker_description": "official", "state_info": "north" if i % 2 else "south",
                         "context": "speech" if i < 84 else "interview", "justification": "excluded", "true_counts": 20})
        duplicate = dict(rows[6]); duplicate["date"] = "2022-01-01"; rows.append(duplicate)
        frame = pd.DataFrame(rows)
        for j, name in enumerate(("train.csv", "valid.csv", "test.csv")):
            frame.iloc[j::3].iloc[::-1].to_csv(directory / name, index=False)

    def test_every_notebook_cell_and_segment_is_exact(self):
        verified = self.runner.verify_sources()
        self.assertEqual(verified["notebooks"], 7)
        self.assertEqual(verified["code_cells"], 88)
        self.assertEqual(verified["entrypoint_segments"], 37)
        self.assertEqual(verified["routes"], 28)

    def test_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "bundle"
            shutil.copytree(self.ROOT, root, ignore=shutil.ignore_patterns("__pycache__", "results"))
            source = root / "original_sources/drift_analysis/cell_002.py"
            source.write_bytes(source.read_bytes() + b"\n# altered\n")
            with self.assertRaisesRegex(ValueError, "differs from its notebook"):
                JournalExperimentRunner(root).verify_sources()

    def test_three_liar2_preprocessor_versions_stay_distinct(self):
        hashes = {}
        for experiment, index in (("drift_analysis", 2), ("baseline_comparison", 2), ("da_minilm_rwac", 2)):
            cell = next(c for c in self.runner.sources["experiments"][experiment]["cells"] if c["index"] == index)
            hashes[experiment] = cell["sha256"]
            plan = self.runner.execution_plan(experiment)
            match = next(step for step in plan["steps"] if step["kind"] == "cell" and step["index"] == index)
            self.assertEqual(match["sha256"], cell["sha256"])
        self.assertEqual(len(set(hashes.values())), 3)

    def test_notebook_replay_keeps_every_original_cell_in_order(self):
        for experiment, data in self.runner.sources["experiments"].items():
            plan = self.runner.execution_plan(experiment, "notebook")
            self.assertEqual([s["index"] for s in plan["steps"]], [s["index"] for s in data["cells"]])
            self.assertEqual([s["sha256"] for s in plan["steps"]], [s["sha256"] for s in data["cells"]])

    def test_preprocessor_behavior_is_not_unified_across_versions(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temp); self.write_liar2(directory)
            inputs = [directory / name for name in ("train.csv", "valid.csv", "test.csv")]
            instances = {}
            for experiment in ("drift_analysis", "baseline_comparison", "da_minilm_rwac"):
                cell = next(c for c in self.runner.sources["experiments"][experiment]["cells"] if c["index"] == 2)
                namespace = {"__name__": "version_reference"}
                exec((self.ROOT / cell["path"]).read_text(), namespace)
                instances[experiment] = namespace["Liar2Preprocessor"]()
            self.assertFalse(hasattr(instances["drift_analysis"], "get_split"))
            self.assertTrue(hasattr(instances["baseline_comparison"], "get_split"))
            self.assertTrue(hasattr(instances["da_minilm_rwac"], "get_split"))
            for experiment in ("drift_analysis", "baseline_comparison"):
                self.assertEqual(len(instances[experiment].run_pipeline(*inputs)), 121)
            # The latest version deduplicates internally and then enforces the
            # original journal cohort size. The runner must not remove this.
            with self.assertRaisesRegex(ValueError, "got 120"):
                instances["da_minilm_rwac"].run_pipeline(*inputs)

    def test_drift_repeated_run_is_preserved(self):
        plan = self.runner.execution_plan("drift_analysis")
        indices = [s["index"] for s in plan["steps"]]
        self.assertEqual(indices, [2, 4, 6, 7, 8, 9, 10, 11])
        for index in (8, 10):
            step = next(s for s in plan["steps"] if s["index"] == index)
            source = (self.ROOT / step["path"]).read_text()
            self.assertIn("include_embedding_shift=True", source)

    def test_stage_dependencies_use_original_slices(self):
        for experiment in ("baseline_comparison", "da_minilm_rwac"):
            bootstrap = self.runner.execution_plan(experiment, "paired_bootstrap")
            names = [s.get("name") for s in bootstrap["steps"] if s["kind"] == "segment"]
            self.assertEqual(names[-2:], ["section_06", "section_07"])
            for stage, expected in (("component_ablation", "section_08"), ("shuffle_control", "section_09"), ("rolling_windows", "section_10")):
                plan = self.runner.execution_plan(experiment, stage)
                names = [s.get("name") for s in plan["steps"] if s["kind"] == "segment"]
                self.assertIn("prepare_main", names)
                self.assertNotIn("section_06", names)
                self.assertEqual(names[-1], expected)

    def test_scope_preserves_functions_comprehensions_and_global_lookup(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _RuntimePaths("drift_analysis", temp, {})
            globals_dict = {"__builtins__": builtins.__dict__}
            scope = _NotebookScope(globals_dict, paths)
            source = "x = 3\nvalues = [x * i for i in range(4)]\ndef f():\n    return x, values\nx = 5\nresult = f()\n"
            exec(compile(source, "semantic_fixture", "exec"), globals_dict, scope)
            expected = {}
            exec(source, expected)
            self.assertEqual(globals_dict["result"], expected["result"])
            self.assertIs(globals_dict["f"].__globals__, globals_dict)

    def test_scope_alters_only_filesystem_values(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _RuntimePaths("drift_analysis", temp, {"liar2": temp})
            values = {"SEEDS": [42, 43, 44, 45, 46], "RECENCY_LAMBDAS": [0.0, 0.05, 0.1],
                      "TEXT_MODE": "claim_plus_safe_metadata", "NUM_CLASSES": 6,
                      "OUTPUT_DIR": "/kaggle/working/native_name"}
            globals_dict = {}
            scope = _NotebookScope(globals_dict, paths)
            for key, value in values.items(): scope[key] = value
            for key in values:
                if key != "OUTPUT_DIR": self.assertIs(globals_dict[key], values[key])
            self.assertEqual(globals_dict["OUTPUT_DIR"], str(Path(temp) / "native_name"))

    def test_terminal_prepare_with_paths_containing_spaces(self):
        with tempfile.TemporaryDirectory(prefix="journal runner ") as temp:
            directory = Path(temp)
            data = directory / "input CSVs"; data.mkdir()
            self.write_liar2(data)
            output = directory / "output files"
            process = subprocess.run(
                [sys.executable, str(self.ROOT / "main.py"), "run", "drift_analysis", "--stage", "prepare",
                 "--liar2-dir", str(data), "--output-dir", str(output)],
                capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads((output / "_runner/run_manifest.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["experimental_code_modified"])
            self.assertEqual(len(report["completed_steps"]), 4)
            self.assertIn("84 18 18", process.stdout)
            self.assertTrue((output / "_runner/execution.log").is_file())

    def test_original_offline_calculations_match_unadapted_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); self.write_liar2(directory)
            paths = _RuntimePaths("drift_analysis", directory / "results", {"liar2": directory})
            direct = {"__name__": "archived_reference"}
            adapted = {"__name__": "archived_reference", "__builtins__": builtins.__dict__}
            scope = _NotebookScope(adapted, paths)
            with contextlib.redirect_stdout(io.StringIO()):
                for index in (2, 4):
                    cell = next(c for c in self.runner.sources["experiments"]["drift_analysis"]["cells"] if c["index"] == index)
                    source = (self.ROOT / cell["path"]).read_text()
                    exec(source, direct)
                    exec(source, adapted, scope)
                frames = []
                for namespace, input_paths in ((direct, [directory / n for n in ("train.csv", "valid.csv", "test.csv")]),
                                               (adapted, ["/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/" + n for n in ("train.csv", "valid.csv", "test.csv")])):
                    p = namespace["Liar2Preprocessor"]()
                    frame = p.run_pipeline(*input_paths)
                    frame = p.drop_duplicate_statements(frame, policy="keep_first")
                    train, valid, test = p.chronological_split(frame)
                    da = namespace["DriftAnalysis"]()
                    summary = da.run_all_drift_experiments(frame.iloc[train], frame.iloc[test], include_embedding_shift=False)
                    frames.append((frame, summary))
            pd.testing.assert_frame_equal(frames[0][0], frames[1][0], check_exact=True)
            pd.testing.assert_frame_equal(frames[0][1], frames[1][1], check_exact=True)

    def test_literal_csv_export_and_path_factory_are_routed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = _RuntimePaths("drift_analysis", directory, {})
            namespace = {"__builtins__": builtins.__dict__}
            scope = _NotebookScope(namespace, paths)
            before = pd.DataFrame.to_csv
            with self.runner._runtime(paths):
                exec("from pathlib import Path\nimport pandas as pd\np = Path('/kaggle/working/native')\np.mkdir()\npd.DataFrame({'value':[7]}).to_csv('/kaggle/working/export.csv', index=False)\n", namespace, scope)
                self.assertEqual(Path.cwd(), directory)
            self.assertIs(pd.DataFrame.to_csv, before)
            self.assertTrue((directory / "native").is_dir())
            self.assertEqual(pd.read_csv(directory / "export.csv")["value"].tolist(), [7])

    def test_failed_execution_records_original_error(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            process = subprocess.run(
                [sys.executable, str(self.ROOT / "main.py"), "run", "drift_analysis", "--stage", "prepare",
                 "--liar2-dir", str(Path(temp) / "missing"), "--output-dir", str(output)],
                capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, 1)
            report = json.loads((output / "_runner/run_manifest.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error_type"], "FileNotFoundError")
            self.assertEqual(report["completed_steps"], [])

    def test_all_public_experiment_methods_have_correct_route(self):
        names = ["baseline_comparison", "drift_analysis", "da_minilm_rwac", "liar2_label_granularity",
                 "ensemble", "multifc_da_minilm", "multifc_label_granularity"]
        for name in names:
            method = getattr(self.runner, "run_" + name)
            self.assertEqual(method(dry_run=True)["experiment"], name)
        for method, stage in (("run_paired_bootstrap", "paired_bootstrap"), ("run_component_ablation", "component_ablation"),
                              ("run_shuffle_control", "shuffle_control"), ("run_rolling_windows", "rolling_windows")):
            self.assertEqual(getattr(self.runner, method)(dry_run=True)["stage"], stage)

    def test_terminal_list_verify_and_dry_run_need_no_model_imports(self):
        for arguments in (["list"], ["verify"], ["run", "multifc_label_granularity", "--dry-run"]):
            process = subprocess.run([sys.executable, str(self.ROOT / "main.py"), *arguments], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
        process = subprocess.run([sys.executable, "-c", "import sys; import experiment_runner; assert 'torch' not in sys.modules; assert 'pandas' not in sys.modules"], cwd=self.ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_new_module_has_only_main_as_standalone_function(self):
        for name in ("experiment_runner.py", "main.py"):
            tree = ast.parse((self.ROOT / name).read_text())
            functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            self.assertEqual(functions, ["main"] if name == "main.py" else [])
