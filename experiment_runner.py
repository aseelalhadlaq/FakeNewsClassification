"""Run archived journal experiments without rewriting their original code."""

import argparse
import ast
import builtins
from collections.abc import MutableMapping
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


class _RuntimePaths:
    """Translate filesystem locations only; experimental values remain original."""

    def __init__(self, experiment, output_dir, dataset_dirs):
        self.output_dir = Path(output_dir).resolve()
        self.mappings = []
        original_roots = {
            "liar": "/kaggle/input/datasets/badalahmmed/liar-dataset/liar dataset/liar dataset",
            "liar2": "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2",
            "multifc": "/kaggle/input/datasets/badalahmmed/multifc",
        }
        for name, directory in dataset_dirs.items():
            if directory is not None:
                self.mappings.append((original_roots[name], str(Path(directory).resolve())))
        if experiment.startswith("multifc") and dataset_dirs.get("multifc") is not None:
            directory = str(Path(dataset_dirs["multifc"]).resolve())
            for prefix in ("/kaggle/input/multifc-drift-aware", "/kaggle/input/multifc-dataset",
                           "/kaggle/input/multifc", "/kaggle/input"):
                self.mappings.append((prefix, directory))
        self.mappings.extend([
            ("/kaggle/working", str(self.output_dir)),
            ("/mnt/data", str(self.output_dir)),
        ])
        self.mappings.sort(key=lambda pair: len(pair[0]), reverse=True)

    def map(self, value):
        if not isinstance(value, (str, os.PathLike)):
            return value
        string = os.fspath(value)
        if not isinstance(string, str):
            return value
        for original, local in self.mappings:
            if string == original or string.startswith(original + "/"):
                suffix = string[len(original):].lstrip("/")
                mapped = str(Path(local).joinpath(*suffix.split("/"))) if suffix else local
                return Path(mapped) if isinstance(value, Path) else mapped
        return value

    def path(self, *parts, **kwargs):
        return Path(self.map(Path(*parts, **kwargs)))

    def open(self, file, *args, **kwargs):
        return builtins.open(self.map(file), *args, **kwargs)


class _PandasPaths:
    """Forward pandas operations, translating only input file paths."""

    def __init__(self, module, paths):
        self._module = module
        self._paths = paths

    def __getattr__(self, name):
        return getattr(self._module, name)

    def read_csv(self, filepath_or_buffer, *args, **kwargs):
        return self._module.read_csv(self._paths.map(filepath_or_buffer), *args, **kwargs)


class _NotebookScope(MutableMapping):
    """A single notebook namespace; file paths are the only runtime adapters."""

    def __init__(self, globals_dict, paths):
        self.globals = globals_dict
        self.paths = paths

    def __getitem__(self, key):
        return self.globals[key]

    def __setitem__(self, key, value):
        if key == "pd" and getattr(value, "__name__", None) == "pandas":
            value = _PandasPaths(value, self.paths)
        elif key == "Path" and value is Path:
            value = self.paths.path
        elif isinstance(value, (str, os.PathLike)):
            value = self.paths.map(value)
        self.globals[key] = value

    def __delitem__(self, key):
        del self.globals[key]

    def __iter__(self):
        return iter(self.globals)

    def __len__(self):
        return len(self.globals)


class _Tee:
    """Show original terminal output while retaining a plain-text execution log."""

    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, text):
        self.stream.write(text)
        self.logfile.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class JournalExperimentRunner:
    """One public runner for seven independent, version-preserving experiments.

    Each public run starts a fresh Python process. Original notebook cells and
    exact source slices execute in their own shared namespace, preserving the
    notebook's global lookup and definition order. No algorithm is patched.
    """

    CORE_PACKAGES = {
        "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "sklearn": "scikit-learn",
    }
    MODEL_PACKAGES = {
        "torch": "torch", "transformers": "transformers",
        "sentence_transformers": "sentence-transformers", "accelerate": "accelerate",
    }

    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parent).resolve()
        self.sources = json.loads((self.root / "source_manifest.json").read_text(encoding="utf-8"))
        self.registry = json.loads((self.root / "experiment_registry.json").read_text(encoding="utf-8"))

    @staticmethod
    def _sha256(data):
        return hashlib.sha256(data).hexdigest()

    def list_experiments(self):
        return {key: {"title": value["title"], "default_stage": value["default_stage"],
                      "stages": list(value["stages"]),
                      "notebook": self.sources["experiments"][key]["notebook_path"]}
                for key, value in self.registry["experiments"].items()}

    def _source(self, experiment, step):
        experiment_sources = self.sources["experiments"][experiment]
        if step["kind"] == "cell":
            return next(cell for cell in experiment_sources["cells"] if cell["index"] == step["index"])
        if step["kind"] == "segment":
            return experiment_sources["segments"][step["name"]]
        raise ValueError(f"Unsupported step kind: {step['kind']}")

    def verify_sources(self):
        """Compare notebooks, all cell bytes, and every slice against the archive."""
        counts = {"notebooks": 0, "code_cells": 0, "entrypoint_segments": 0,
                  "class_or_function_definitions": 0}
        for experiment, data in self.sources["experiments"].items():
            notebook_path = self.root / data["notebook_path"]
            notebook_bytes = notebook_path.read_bytes()
            if self._sha256(notebook_bytes) != data["notebook_sha256"]:
                raise ValueError(f"Original notebook changed: {notebook_path}")
            notebook = json.loads(notebook_bytes)
            indices = [i for i, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"]
            if indices != [cell["index"] for cell in data["cells"]]:
                raise ValueError(f"Cell index mapping changed for {experiment}.")
            counts["notebooks"] += 1
            for cell in data["cells"]:
                original = "".join(notebook["cells"][cell["index"]]["source"]).encode("utf-8")
                copied = (self.root / cell["path"]).read_bytes()
                if copied != original or self._sha256(copied) != cell["sha256"]:
                    raise ValueError(f"Source cell differs from its notebook: {cell['path']}")
                compile(copied.decode("utf-8"), str(self.root / cell["path"]), "exec")
                counts["code_cells"] += 1
                counts["class_or_function_definitions"] += len(cell["definitions"])
            for name, segment in data.get("segments", {}).items():
                original = "".join(notebook["cells"][segment["source_cell"]]["source"])
                expected = "".join(original.splitlines(keepends=True)[segment["start_line"] - 1:segment["end_line"]]).encode("utf-8")
                copied = (self.root / segment["path"]).read_bytes()
                if copied != expected or self._sha256(copied) != segment["sha256"]:
                    raise ValueError(f"Entrypoint segment differs from its notebook: {segment['path']}")
                compile(copied.decode("utf-8"), str(self.root / segment["path"]), "exec")
                counts["entrypoint_segments"] += 1
        routes = 0
        for experiment, details in self.registry["experiments"].items():
            if details["default_stage"] not in details["stages"]:
                raise ValueError(f"Invalid default stage for {experiment}.")
            original_cells = [cell["index"] for cell in self.sources["experiments"][experiment]["cells"]]
            notebook_steps = details["stages"]["notebook"]["steps"]
            if notebook_steps != [{"kind": "cell", "index": i} for i in original_cells]:
                raise ValueError(f"Notebook replay order changed for {experiment}.")
            for stage in details["stages"].values():
                for step in stage["steps"]:
                    self._source(experiment, step)
                routes += 1
        counts["routes"] = routes
        return {"status": "verified", **counts}

    def execution_plan(self, experiment, stage=None):
        if experiment not in self.registry["experiments"]:
            raise ValueError(f"Unknown experiment: {experiment}")
        details = self.registry["experiments"][experiment]
        stage = stage or details["default_stage"]
        if stage not in details["stages"]:
            raise ValueError(f"Unknown stage {stage!r} for {experiment}; choose {list(details['stages'])}")
        specification = details["stages"][stage]
        steps = []
        for step in specification["steps"]:
            source = self._source(experiment, step)
            steps.append({**step, **source})
        source_data = self.sources["experiments"][experiment]
        return {"experiment": experiment, "stage": stage, "description": specification["description"],
                "original_notebook": source_data["notebook_path"],
                "notebook_sha256": source_data["notebook_sha256"],
                "required_datasets": specification["datasets"], "steps": steps}

    def _check_environment(self, plan):
        packages = dict(self.CORE_PACKAGES)
        if plan["experiment"] == "drift_analysis":
            if plan["stage"] != "prepare":
                packages.update({"torch": "torch", "transformers": "transformers"})
        else:
            packages.update(self.MODEL_PACKAGES)
        if plan["experiment"] == "multifc_da_minilm":
            packages["matplotlib"] = "matplotlib"
        missing = [distribution for name, distribution in packages.items() if importlib.util.find_spec(name) is None]
        if missing:
            raise ImportError("Missing dependencies: " + ", ".join(missing) +
                              ". Install requirements.txt before running; archived installer cells remain unchanged.")
        versions = {}
        for distribution in packages.values():
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = "unknown"
        if plan["experiment"] == "multifc_da_minilm":
            import inspect
            from sklearn.linear_model import LogisticRegression
            if "multi_class" not in inspect.signature(LogisticRegression).parameters:
                raise RuntimeError("This original MultiFC code requires LogisticRegression(multi_class=...). "
                                   "Use the scikit-learn<1.8 constraint in requirements-core.txt; the source is not patched.")
        return versions

    @staticmethod
    def _check_dataset_dirs(required, dataset_dirs):
        for name in required:
            directory = dataset_dirs.get(name)
            if directory is not None and not Path(directory).is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {directory}")

    @contextlib.contextmanager
    def _runtime(self, paths):
        """Route literal CSV exports and relative output paths inside the worker."""
        import pandas as pd
        previous_dir = Path.cwd()
        original_to_csv = pd.DataFrame.to_csv
        adapter = paths

        class CSVWriter:
            @staticmethod
            def write(frame, path_or_buf=None, *args, **kwargs):
                return original_to_csv(frame, adapter.map(path_or_buf), *args, **kwargs)

        try:
            pd.DataFrame.to_csv = CSVWriter.write
            os.chdir(paths.output_dir)
            yield
        finally:
            os.chdir(previous_dir)
            pd.DataFrame.to_csv = original_to_csv

    @staticmethod
    def _terminal_display(*objects, **options):
        """Provide the notebook display service when IPython is unavailable."""
        for value in objects:
            print(value)

    def _execute(self, experiment, stage=None, output_dir=None, liar_dir=None, liar2_dir=None, multifc_dir=None):
        """Worker implementation; public methods use process isolation."""
        plan = self.execution_plan(experiment, stage)
        output_dir = Path(output_dir or self.root / "results" / experiment).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir = output_dir / "_runner"
        metadata_dir.mkdir(exist_ok=True)
        report_path = metadata_dir / "run_manifest.json"
        log_path = metadata_dir / "execution.log"
        dataset_dirs = {"liar": liar_dir, "liar2": liar2_dir, "multifc": multifc_dir}
        dataset_dirs = {key: str(Path(value).expanduser().resolve()) if value is not None else None
                        for key, value in dataset_dirs.items()}
        paths = _RuntimePaths(experiment, output_dir, dataset_dirs)
        report = {**plan, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "output_directory": str(output_dir), "dataset_directories": dataset_dirs,
                  "path_aliases": paths.mappings, "completed_steps": [],
                  "experimental_code_modified": False}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as logfile:
            with contextlib.redirect_stdout(_Tee(sys.stdout, logfile)), contextlib.redirect_stderr(_Tee(sys.stderr, logfile)):
                try:
                    report["source_verification"] = self.verify_sources()
                    self._check_dataset_dirs(plan["required_datasets"], dataset_dirs)
                    report["installed_versions"] = self._check_environment(plan)
                    try:
                        from IPython.display import display
                    except ImportError:
                        display = self._terminal_display
                    globals_dict = {"__name__": "__main__", "__builtins__": builtins.__dict__,
                                    "display": display, "open": paths.open}
                    scope = _NotebookScope(globals_dict, paths)
                    with self._runtime(paths):
                        for position, step in enumerate(plan["steps"], 1):
                            source_path = self.root / step["path"]
                            source_bytes = source_path.read_bytes()
                            if self._sha256(source_bytes) != step["sha256"]:
                                raise ValueError(f"Source changed after verification: {source_path}")
                            print(f"[runner] {experiment}/{plan['stage']} step {position}/{len(plan['steps'])}: {step['path']}", flush=True)
                            globals_dict["__file__"] = str(source_path)
                            exec(compile(source_bytes.decode("utf-8"), str(source_path), "exec"), globals_dict, scope)
                            report["completed_steps"].append(step["path"])
                    report["status"] = "completed"
                except Exception as exc:
                    report["status"] = "failed"
                    report["error_type"] = type(exc).__name__
                    report["error"] = str(exc)
                    report["traceback"] = traceback.format_exc()
                    traceback.print_exc()
                finally:
                    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    def run(self, experiment, stage=None, output_dir=None, liar_dir=None, liar2_dir=None, multifc_dir=None, dry_run=False):
        """Execute one route in a fresh interpreter; return its execution manifest."""
        self.verify_sources()
        plan = self.execution_plan(experiment, stage)
        if dry_run:
            return plan
        output_dir = Path(output_dir or self.root / "results" / experiment).expanduser().resolve()
        command = [sys.executable, str(self.root / "main.py"), "run", experiment,
                   "--stage", plan["stage"], "--output-dir", str(output_dir), "--worker"]
        for name, value in (("liar", liar_dir), ("liar2", liar2_dir), ("multifc", multifc_dir)):
            if value is not None:
                command.extend([f"--{name}-dir", str(Path(value).expanduser().resolve())])
        process = subprocess.run(command, cwd=self.root)
        report_path = output_dir / "_runner" / "run_manifest.json"
        if process.returncode:
            raise RuntimeError(f"Original experiment execution failed (exit {process.returncode}). See {report_path}")
        return json.loads(report_path.read_text(encoding="utf-8"))

    def run_baseline_comparison(self, **options):
        return self.run("baseline_comparison", **options)

    def run_drift_analysis(self, **options):
        return self.run("drift_analysis", **options)

    def run_da_minilm_rwac(self, **options):
        return self.run("da_minilm_rwac", **options)

    def run_liar2_label_granularity(self, **options):
        return self.run("liar2_label_granularity", **options)

    def run_ensemble(self, **options):
        return self.run("ensemble", **options)

    def run_multifc_da_minilm(self, **options):
        return self.run("multifc_da_minilm", **options)

    def run_multifc_label_granularity(self, **options):
        return self.run("multifc_label_granularity", **options)

    def run_paired_bootstrap(self, **options):
        return self.run("da_minilm_rwac", stage="paired_bootstrap", **options)

    def run_component_ablation(self, **options):
        return self.run("da_minilm_rwac", stage="component_ablation", **options)

    def run_shuffle_control(self, **options):
        return self.run("da_minilm_rwac", stage="shuffle_control", **options)

    def run_rolling_windows(self, **options):
        return self.run("da_minilm_rwac", stage="rolling_windows", **options)

    def run_da_minilm_main(self, **options):
        return self.run("da_minilm_rwac", stage="main", **options)

    def run_distilbert_chronological(self, **options):
        return self.run("da_minilm_rwac", stage="distilbert_chronological", **options)

    def run_notebook(self, experiment, **options):
        return self.run(experiment, stage="notebook", **options)

    def build_argument_parser(self):
        parser = argparse.ArgumentParser(description="Run unchanged, version-specific journal experiments.")
        commands = parser.add_subparsers(dest="command", required=True)
        commands.add_parser("list", help="List the seven experiments and their independent stages.")
        commands.add_parser("verify", help="Verify original notebook, cell, and entrypoint hashes.")
        run = commands.add_parser("run", help="Run one experiment or an independent stage.")
        run.add_argument("experiment", choices=list(self.registry["experiments"]))
        run.add_argument("--stage", help="Stage from the list command; default is the experiment's application.")
        run.add_argument("--liar-dir", type=Path, help="LIAR directory containing train.tsv, valid.tsv, test.tsv.")
        run.add_argument("--liar2-dir", type=Path, help="LIAR2 directory containing train.csv, valid.csv, test.csv.")
        run.add_argument("--multifc-dir", type=Path, help="MultiFC directory containing train.tsv, dev.tsv, test.tsv.")
        run.add_argument("--output-dir", type=Path, help="Output root; original artifact/subdirectory names are retained.")
        run.add_argument("--dry-run", action="store_true", help="Print the source plan without executing experiment code.")
        run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        return parser

    @classmethod
    def run_cli(cls, argv=None):
        runner = cls()
        parser = runner.build_argument_parser()
        args = parser.parse_args(argv)
        if args.command == "list":
            for name, details in runner.list_experiments().items():
                print(f"{name}: {details['title']}")
                print(f"  default={details['default_stage']}; stages={', '.join(details['stages'])}")
            return 0
        if args.command == "verify":
            print(json.dumps(runner.verify_sources(), indent=2))
            return 0
        options = {"stage": args.stage, "output_dir": args.output_dir,
                   "liar_dir": args.liar_dir, "liar2_dir": args.liar2_dir, "multifc_dir": args.multifc_dir}
        if args.worker:
            report = runner._execute(args.experiment, **options)
            return 0 if report["status"] == "completed" else 1
        report = runner.run(args.experiment, dry_run=args.dry_run, **options)
        if args.dry_run:
            print(json.dumps(report, indent=2))
        else:
            print(f"Completed {report['experiment']}/{report['stage']}; outputs: {report['output_directory']}")
        return 0
