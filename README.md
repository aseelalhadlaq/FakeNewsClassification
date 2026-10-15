# Fake News Classification: Reproducible Experiments and Temporal-Drift Analysis

**Research code and reproducibility guide**  
Repository: https://github.com/aseelalhadlaq/FakeNewsClassification

## Project description

This repository contains the research implementation of a collection of fake-news / fact-checking claim classification experiments, including baseline comparisons, temporal/concept drift analyses, DA-MiniLM-RWAC models, ensemble evaluations, and label-granularity studies. Experiments use the **LIAR**, **LIAR2**, and **MultiFC** datasets. The archive provides seven original Jupyter notebooks together with extracted Python source code and a command-line/Python experiment runner. It does **not** include source datasets or downloaded pretrained model weights.

The purpose of the repository is to document the methodology and make the provided implementations reusable and inspectable. **Reproducing exactly the numerical results in the associated paper has not been independently verified using the original datasets and historical software environment.** See [Verification and limitations](#verification-and-limitations).

## Contents and code information

| File / folder | Purpose |
| --- | --- |
| `README.md` | Project documentation, installation, data preparation, execution and reproducibility notes |
| `main.py` | Terminal interface for listing, checking, and running experiments |
| `experiment_runner.py` | Python API and isolated experiment execution |
| `experiment_registry.json` | Experiment and stage definitions |
| `original_notebooks/` | Seven original Jupyter experiment notebooks |
| `original_sources/` | Python code cells extracted from the original notebooks |
| `entrypoints/` | Source-preserving entrypoint segments for selected stages |
| `requirements.txt`, `requirements-core.txt` | Software dependency compatibility specifications |
| `source_manifest.json`, `version_index.json` | Source integrity and provenance records |
| `tests/`, `VALIDATION.json` | Runner tests and the recorded validation scope |

### Available experiments

| Experiment ID | Study / analysis | Input dataset(s) |
| --- | --- | --- |
| `baseline_comparison` | Conventional and transformer baseline comparisons | LIAR, LIAR2 |
| `drift_analysis` | Temporal drift diagnostics | LIAR2 |
| `da_minilm_rwac` | Drift-aware MiniLM and reweighting/calibration analyses | LIAR, LIAR2 |
| `liar2_label_granularity` | Effect of label granularity | LIAR2 |
| `ensemble` | Model and seed ensembles | LIAR2 |
| `multifc_da_minilm` | MiniLM-based experiments with MultiFC | MultiFC |
| `multifc_label_granularity` | MultiFC label granularity | MultiFC |

Run `python main.py list` for the definitive available route names. Stage availability and commands are documented in the [detailed reference](#detailed-experiment-and-command-reference).

## Dataset information and access

**Data files are not bundled in this repository.** Download the datasets from their respective distributors and follow their terms of use and citation instructions. The listed sources are access points, not guaranteed identical pinned revisions of the copies used for the manuscript.

| Dataset | Dataset/source page | Required local files | Format |
| --- | --- | --- | --- |
| LIAR | [LIAR dataset documentation](https://datasets-dev.10web.me/docs/ml/datasets/liar-dataset/) | `train.tsv`, `valid.tsv`, `test.tsv` | 14-column headerless TSV |
| LIAR2 | [Hugging Face: chengxuphd/liar2](https://huggingface.co/datasets/chengxuphd/liar2) | `train.csv`, `valid.csv`, `test.csv` | CSV with original LIAR2 columns |
| MultiFC | [Hugging Face: pszemraj/multi_fc](https://huggingface.co/datasets/pszemraj/multi_fc) | `train.tsv`, `dev.tsv`, `test.tsv` | Headerless TSV; labelled and unlabelled layouts differ |

Organize files as follows before executing the corresponding experiments:

```text
project-data/
├── liar/
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
├── liar2/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── multifc/
    ├── train.tsv
    ├── dev.tsv
    └── test.tsv
```

**Important:** Downloaded Hugging Face dataset exports may require conversion to the original formats and column order expected by the scripts. Merely downloading a dataset page or using a Hugging Face dataset object does not automatically produce the required CSV/TSV files. Check the original schemas in the [dataset preparation reference](#2-dataset-directories-and-formats). Preserve official split membership and dates as required by the selected experiment. Do not invent missing column values or evaluate against unlabelled test records. For stronger version control, record your exact dataset release/revision and SHA-256 checksums before publication.

## System requirements and installation

Python **3.10–3.12** is the documented target range. A virtual environment is recommended. The full training runs also require sufficient compute resources, disk space, and access to pretrained transformer models. GPU availability can substantially affect runtime.

```bash
git clone https://github.com/aseelalhadlaq/FakeNewsClassification.git
cd FakeNewsClassification
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment using `.venv\Scripts\Activate.ps1`. When using the supplied ZIP rather than `git clone`, enter its extracted top-level folder instead.

For the original DistilBERT-related routes and the default DA-MiniLM-RWAC application, the code also imports `datasets`; install it when needed:

```bash
python -m pip install datasets
```

Core dependencies are NumPy, pandas, SciPy, and scikit-learn. Additional requirements include PyTorch, Transformers, Sentence Transformers, Accelerate, Matplotlib, tqdm and IPython. See `requirements.txt` and `requirements-core.txt` for declared version ranges. **These are compatibility requirements, not an exact record of the original paper's software environment.** The notebook-derived code may also contain installer cells. For a documented run, record `python --version` and `python -m pip freeze` along with the repository commit and dataset revisions.

The code references model checkpoints such as `sentence-transformers/all-MiniLM-L6-v2`, `microsoft/MiniLM-L12-H384-uncased`, and `distilbert-base-uncased`; initial runs may require downloading them from their providers. Reusing non-pinned model names at a later time is not equivalent to a frozen model revision.

## Quick start: verify and run

From the repository root:

```bash
# View available experiments
python main.py list

# Verify archived source integrity (does not train models)
python main.py verify

# Run the included unit tests
python -m unittest discover -s tests -v

# Preview the LIAR2 drift-analysis execution plan (no training)
python main.py run drift_analysis --liar2-dir /data/liar2 --output-dir results/drift_preview --dry-run

# Execute the LIAR2 drift-analysis application
python main.py run drift_analysis --liar2-dir /data/liar2 --output-dir results/drift

# Execute the main RWAC experiment (requires both LIAR and LIAR2 directories)
python main.py run da_minilm_rwac --stage main --liar-dir /data/liar --liar2-dir /data/liar2 --output-dir results/rwac_main

# Execute the MultiFC application
python main.py run multifc_da_minilm --multifc-dir /data/multifc --output-dir results/multifc
```

Replace `/data/liar`, `/data/liar2`, and `/data/multifc` with the paths to **your actual prepared dataset directories**. On Windows, use the corresponding local paths and quote paths containing spaces. Commands that call training/inference can be computationally expensive. Use `--dry-run` to inspect a route first. Not every stage uses the same dataset flags; refer to the detailed per-experiment commands below.

## Methodology overview

The experiments follow the methodological steps implemented in the original notebooks; individual variants differ in their selected stages and evaluation design.

1. **Load data:** Read the original LIAR, LIAR2, or MultiFC splits and retain source/split metadata as needed.
2. **Clean and normalize:** Standardize statement or claim text, handle missing fields, harmonize labels, and exclude unsupported rows.
3. **Prepare prediction inputs:** Form claim-only and selected restricted-metadata variants. Exclude ground-truth labels, verdict justifications and prohibited credibility-history fields from model inputs, as specified by the experiments.
4. **Prepare evaluation splits:** Use official or configured random-stratified/chronological splits. Temporal analyses depend on available dates and recorded ordering decisions.
5. **Train/evaluate:** Run the specified baseline, transformer, drift-aware, ensemble or label-granularity stage with its source-defined configuration.
6. **Analyze:** Generate evaluation summaries and, where applicable, temporal window, bootstrap, ablation, permutation/shuffle-control and calibration outputs.
7. **Preserve provenance:** Inspect the log, manifest, code hash information, and output files for each execution.

**Do not interpret this overview as proof that all seven model-training pipelines were executed successfully in this release.** Exact filtering, hyperparameters, seeds and stage-specific logic are in the original source and the detailed reference.

## Outputs and result interpretation

The selected `--output-dir` contains experiment outputs and a `_runner/` subdirectory:

- `_runner/execution.log`: console output, step progress, and errors.
- `_runner/run_manifest.json`: status, source hashes, dataset paths, package versions and output mapping.
- Stage-specific result tables, e.g. `da_minilm_main_results.csv`, `da_minilm_lambda_selection.csv`, `da_minilm_summary.csv`, or drift-analysis CSVs, when the corresponding stages complete.

The manifest indicates whether a selected route ran to completion; **it does not by itself establish agreement with manuscript numbers**. Retain the generated files and environment record when reporting results.

## Verification and limitations

`VALIDATION.json` records **16 passing bundled checks** of source preservation, routing, runner behavior, and synthetic-fixture preprocessing. It records seven original notebooks, 88 extracted code cells, 37 entrypoint segments and 28 routes. These checks **do not represent independent complete training and inference across all datasets**. The supplied validation explicitly records:

- `full_model_training_executed: false`
- `numerical_reproduction_of_journal_results_verified: false`

The reason recorded is that the original dataset files, pretrained weights, and original environment lockfile were not supplied with the archive. Reusers should run the target experiments with documented datasets, record model revisions and package versions, then compare output metrics with the paper. If results differ, check data releases, split composition, random seeds and hardware-dependent library behavior.

## Citations and acknowledgement

If you use this repository, please cite the associated research article **using its verified publication metadata** and cite the original datasets and pretrained models as required by their providers. The archive does not include a confirmed journal article DOI or full bibliographic citation, so this README intentionally does not invent one. The dataset access links are listed above; please consult their source pages for official papers and recommended citations.

**For the repository maintainer before journal resubmission:** add the final manuscript citation (authors, title, year, venue, DOI if available), exact dataset source/revision identifiers, and any permissible pretrained checkpoint revisions used in the reported experiments. This is necessary for a reproducibility claim stronger than the present archive supports.

## License and contributions

**No repository-wide license file was present in the supplied archive.** Therefore, the repository should not be assumed to grant open-source reuse permission merely because its code is publicly visible. Dataset/model licenses remain controlled by their respective owners. To permit reuse, the repository maintainer should add an explicit `LICENSE` file after selecting a suitable license and confirming the rights to distribute each component. Until then, contributors can suggest improvements through GitHub issues or pull requests, subject to the maintainer's approval; do not include external data or weights without the right to redistribute them.

## Detailed experiment and command reference

The following reference retains the original per-stage documentation, inputs, exact CLI examples, Python API, stage names, output filenames and troubleshooting guidance. Its numbered sections are retained for cross-reference with the original implementation.

---

# Detailed experiment and command reference

`JournalExperimentRunner` provides a common Python interface and terminal entry point for the seven supplied journal notebooks. Users can run each experiment separately and select independent stages where the original application supports them. The experimental code, seeds, label mappings, model settings, and assertions are preserved exactly.

This guide covers installation, data preparation, every experiment, stage selection, Python usage, and output files. Commands assume that the terminal is inside the extracted repository root (`FakeNewsClassification-main/`). Replace `/data/liar`, `/data/liar2`, and `/data/multifc` with your actual dataset directories.

## 1. Installation and first checks

Use Python 3.10–3.12. Clone this repository (or extract its ZIP archive), enter the repository directory, and create an environment:

```bash
cd FakeNewsClassification-main
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. The multiline commands below use Bash continuations; in PowerShell, place each command on one line.

The original DistilBERT routines also import Hugging Face `datasets`. Install it when running the baseline battery, a `distilbert_chronological` stage, or the default `da_minilm_rwac` application:

```bash
python -m pip install datasets
```

Inspect the available routes and verify the archived sources:

```bash
python main.py list
python main.py verify
python main.py --help
python main.py run --help
```

`list` and `verify` use the Python standard library and do not train models. Verification checks seven original notebooks, 88 extracted code cells, 37 entrypoint slices, and 28 registered routes against their provenance records.

Training and embedding routes need the relevant pretrained weights, either already cached or accessible for download. The main MiniLM applications use `sentence-transformers/all-MiniLM-L6-v2`; drift diagnostic D04 uses the separate model `microsoft/MiniLM-L12-H384-uncased`. DistilBERT uses `distilbert-base-uncased`. Device selection follows each original implementation.

The supplied requirements are a compatibility profile, not the journal's recovered environment lockfile. Original installer cells are retained and may install missing packages when reached.

## 2. Dataset directories and formats

Dataset links:

| Dataset | Link |
| --- | --- |
| LIAR | [LIAR dataset documentation](https://datasets-dev.10web.me/docs/ml/datasets/liar-dataset/) |
| LIAR2 | [chengxuphd/liar2 on Hugging Face](https://huggingface.co/datasets/chengxuphd/liar2) |
| MultiFC | [pszemraj/multi_fc on Hugging Face](https://huggingface.co/datasets/pszemraj/multi_fc) |

Supply directories containing the original split files. Filenames must match the names below.

| Dataset | Directory flag | Required files | Expected format |
| --- | --- | --- | --- |
| LIAR | `--liar-dir` | `train.tsv`, `valid.tsv`, `test.tsv` | Headerless TSVs with the original 14-column LIAR layout |
| LIAR2 | `--liar2-dir` | `train.csv`, `valid.csv`, `test.csv` | CSVs with column headers and the original LIAR2 schema |
| MultiFC | `--multifc-dir` | `train.tsv`, `dev.tsv`, `test.tsv` | Headerless TSVs; 13 columns for labelled rows, 12 for unlabelled rows |

LIAR's column order is `id`, `label`, `statement`, `subject`, `speaker`, `job_title`, `state_info`, `party_affiliation`, `barely_true_count`, `false_count`, `half_true_count`, `mostly_true_count`, `pants_on_fire_count`, and `context`. LIAR has no date field; its original preprocessor does not provide chronological splitting.

For LIAR2, retain the original `statement`, `label`, `date`, identifier, and metadata columns. The six canonical labels are `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, and `true`. The ensemble application directly accesses `subject`, `speaker`, `state_info`, and `context`; these columns must be present. Retain the original dates for temporal comparisons.

MultiFC's labelled order is `claim_id`, `claim`, `raw_label`, `url`, `evidence_text`, `category`, `speaker`, `checker`, `tags`, `title`, `published_date`, `claim_date`, and `entities`. The unlabelled layout omits `raw_label`. The MultiFC DA-MiniLM application can generate predictions for the official unlabelled test file. The MultiFC label-granularity application uses that file for row-count reporting and evaluates the labelled cohort.

Both `baseline_comparison` and `da_minilm_rwac` require LIAR and LIAR2 directories for all of their registered stages: their original preparation loads both datasets even when the selected analysis concerns LIAR2 only. The other LIAR2 applications require only LIAR2; the MultiFC applications require only MultiFC.

Quote paths containing spaces. An omitted dataset flag retains the corresponding original Kaggle input location. The archive does not contain datasets or pretrained model weights.

### 2.1 LIAR: basic preprocessing steps

1. **Load the splits.** Read `train.tsv`, `valid.tsv`, and `test.tsv`, assign the 14 column names listed above, and retain each row's official split.
2. **Clean statement text.** Replace missing text with an empty string, collapse repeated whitespace, and trim leading and trailing spaces.
3. **Standardize labels.** Normalize labels to the six canonical classes: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, and `true`.
4. **Filter and encode.** Remove empty statements and unsupported labels, then assign integer `label_id` values.
5. **Build model inputs.** Prepare claim-only text and claim-plus-restricted-metadata text using the permitted metadata fields.
6. **Finalize the dataset.** Exclude speaker credibility-history counts from model inputs and retain official train/validation/test membership. LIAR has no native date column for chronological evaluation.

### 2.2 LIAR2: basic preprocessing steps

1. **Load the splits.** Read `train.csv`, `valid.csv`, and `test.csv`, combine the records, and retain their official split provenance.
2. **Clean statement text.** Handle missing statements, collapse repeated whitespace, and trim spaces.
3. **Standardize and encode labels.** Convert numeric or textual labels to the same six canonical classes used for LIAR, remove invalid rows, and assign `label_id`.
4. **Prepare temporal order.** Parse the `date` column and arrange records for temporal evaluation. Check date coverage and the reported ordering source.
5. **Build model inputs.** Prepare claim-only and claim-plus-restricted-metadata text. Exclude `justification` and speaker credibility-history counts from model inputs.
6. **Check duplicates.** Audit normalized statements and apply the selected experiment's duplicate-handling policy before constructing its evaluation cohort.
7. **Create evaluation splits.** Use the configured chronological or random-stratified train/validation/test split; the temporal splitter uses 70%/15%/15%. Retain row identifiers and temporal ranks where used.

### 2.3 MultiFC: basic preprocessing steps

1. **Load the splits.** Read `train.tsv`, `dev.tsv`, and `test.tsv`, recognize labelled and unlabelled layouts, and retain official split membership.
2. **Clean claims and metadata.** Handle missing values and literal `None` strings, normalize whitespace, and remove empty claims from the analysis cohort.
3. **Harmonize labels.** Map raw verdicts to `false`, `mixed`, or `true`, assign integer labels, and exclude unsupported verdicts from supervised evaluation. Record the mapping in the harmonization report.
4. **Prepare dates.** Parse available publication and claim dates to support temporal ordering; inspect missing-date coverage.
5. **Build model inputs.** Prepare claim-only and claim-plus-restricted-metadata text using the permitted fields. Keep evidence text and verdict labels out of prediction inputs.
6. **Separate evaluation and prediction data.** Use labelled records for the configured official, chronological, or random-stratified evaluation. Keep unlabelled official-test records separate for prediction when enabled; they cannot provide accuracy scores.

Preprocessing runs automatically as part of the selected experiment. Supply the original split files through the dataset directory flags; no separate manual preprocessing script is required. Section 5.2 shows the LIAR2 preparation-only command.

## 3. Code structure and experiment sources

| File or directory | Responsibility |
| --- | --- |
| `main.py` | Standalone `main()` function and terminal entry point |
| `experiment_runner.py` | `JournalExperimentRunner`, route execution, source verification, path adapters, logging, and process isolation |
| `original_notebooks/` | Byte-identical notebooks, including their saved outputs |
| `original_sources/<experiment>/` | Exact original code cells for that notebook |
| `entrypoints/<experiment>/` | Exact source slices used for independent stages in the baseline and RWAC applications |
| `experiment_registry.json` | Default stages, dataset dependencies, and execution order |
| `source_manifest.json` | Notebook, cell, and slice hashes with source locations |
| `version_index.json` | Class and function versions and the notebooks that use them |
| `requirements-core.txt`, `requirements.txt` | Installation profiles |
| `tests/`, `VALIDATION.json` | Verification checks and the recorded validation scope |

The experiment names resolve to these original notebooks:

| Experiment name | Original notebook |
| --- | --- |
| `baseline_comparison` | `exp1-baseline-models-comparisons-codebase-v10-0-final.ipynb` |
| `drift_analysis` | `Exp-2--drift-analysis-experimentations-v8-0-final(1).ipynb` |
| `da_minilm_rwac` | `da-minilm-rwac-codebase-v3-0.ipynb` |
| `liar2_label_granularity` | `exp-4-liar2-label-granularity-experimentsv6-final.ipynb` |
| `ensemble` | `da-ensemble-exp-v3-0(1).ipynb` |
| `multifc_da_minilm` | `da-minilm-multifc-dataset-final.ipynb` |
| `multifc_label_granularity` | `multifc-label-granularity-experiments(2).ipynb` |

Sources are grouped by experiment. Each route loads the definitions belonging to its corresponding notebook in the original order.

## 4. Common terminal options

The command form is:

```text
python main.py run <experiment> [--stage <stage>] [--liar-dir <directory>] [--liar2-dir <directory>] [--multifc-dir <directory>] [--output-dir <directory>] [--dry-run]
```

| Option | Use |
| --- | --- |
| `<experiment>` | One of the seven names listed above |
| `--stage` | A registered stage for that experiment; omission selects its default |
| Dataset directory flags | Redirect the original input paths to local directories |
| `--output-dir` | Set the output root while retaining original artifact and subdirectory names |
| `--dry-run` | Verify sources and print the execution plan without running scientific code |

Without `--output-dir`, results go to `results/<experiment>/` under the codebase directory. Explicit relative output paths resolve from the directory where the command is invoked. Original `/kaggle/working` and `/mnt/data` output paths are mapped under that output root.

Use a fresh output directory for each experiment or stage. Reusing a directory can mix outputs, overwrite runner logs, or reuse original embedding caches. A dry run does not create an output directory and does not validate dataset contents, installed scientific dependencies, or model downloads.

The interface exposes experiment selection, stage selection, and filesystem paths. Seeds, text modes, label schemes, and training settings come from the original experiment code; they have no terminal override flags.

## 5. Utilization of every experiment

### 5.1 Baseline model comparison

Runs the original classical and transformer baseline battery across LIAR, LIAR2, their merged condition, and the supplied chronological and cross-dataset comparisons. The default `baselines` stage includes the five-seed chronological DistilBERT comparison as well as the baseline battery. Transformer baselines are enabled in the original entrypoint.

```bash
python main.py run baseline_comparison \
  --liar-dir /data/liar --liar2-dir /data/liar2 \
  --output-dir results/baselines
```

Read `results/baselines/full_run_results/baseline_results.csv` and `distilbert_chronological_multiseed.csv` in the same folder. The original baseline models include TF-IDF logistic regression, linear SVM, metadata-enhanced logistic regression, DistilBERT, and MiniLM.

To run only its chronological DistilBERT comparison or drift diagnostics:

```bash
python main.py run baseline_comparison --stage distilbert_chronological \
  --liar-dir /data/liar --liar2-dir /data/liar2 \
  --output-dir results/baseline_distilbert

python main.py run baseline_comparison --stage drift \
  --liar-dir /data/liar --liar2-dir /data/liar2 \
  --output-dir results/baseline_drift
```

The `drift` stage exports `full_run_results/drift_summary.csv` and `drift_comparison_vs_reported.csv`, with embedding diagnostic D04 enabled. Additional independent stages are described in Section 6.

### 5.2 Standalone drift analysis: D01–D05

Loads LIAR2, audits and removes duplicate statements with the original `keep_first` policy, then constructs the original chronological train/validation/test split. Drift comparisons use train versus test.

```bash
python main.py run drift_analysis \
  --liar2-dir /data/liar2 --output-dir results/drift
```

| Diagnostic | Analysis |
| --- | --- |
| D01 | Label-distribution drift |
| D02 | Vocabulary-distribution drift |
| D03 | Metadata-distribution drift |
| D04 | MiniLM embedding shift, including MMD and centroid distance |
| D05 | Domain classifier distinguishing train from test |

The original application runs the complete diagnostics twice, with embedding shift enabled both times. Its comparison table is computed after the first run; its final summary comes from the second run. This original sequence is preserved.

The two CSVs are written directly under `results/drift/`: `drift_summary.csv` and `drift_comparison_vs_reported.csv`.

For preprocessing, deduplication, and splitting alone:

```bash
python main.py run drift_analysis --stage prepare \
  --liar2-dir /data/liar2 --output-dir results/drift_prepare
```

This stage records the preparation output in the execution log; it does not export a prepared dataset or run D01–D05. It needs only `requirements-core.txt`. The application route retains D04 and requires its model weights; there is no CLI switch for individual diagnostics or disabling embeddings.

### 5.3 DA-MiniLM-RWAC: complete LIAR2 application

Runs the RWAC notebook's selected application sequence: data preparation, chronological DistilBERT, the main DA-MiniLM and MiniLM-LogReg battery, paired bootstrap, component ablation, shuffle control, and rolling-window evaluation.

```bash
python main.py run da_minilm_rwac \
  --liar-dir /data/liar --liar2-dir /data/liar2 \
  --output-dir results/rwac
```

The main battery uses seeds 42–46, both `claim_only` and `claim_plus_safe_metadata`, and chronological and random-stratified splits. The default application follows the notebook's selected `run_all` sections; the classical baseline and drift sections are absent from that original selected sequence.

The original final-cohort assertions require 22,923 deduplicated LIAR2 rows. A reduced or altered dataset can fail those assertions.

Outputs are under `results/rwac/full_run_results/`. Start with `da_minilm_summary.csv`, then consult `da_minilm_main_results.csv`, `da_minilm_lambda_selection.csv`, and the stage-specific files in Section 6. The `da_minilm/` subdirectory holds original caches and additional CSVs emitted by the original model runner.

### 5.4 LIAR2 label-granularity experiment

Compares the six-class reference with three-class, two-class, and alternative two-class labels using the original deduplicated, chronologically ordered cohort. Each scheme evaluates DA-MiniLM-RWAC, MiniLM-LogReg-RW, and Uniform-LR with seeds 0–4 and 2,000 paired-bootstrap resamples.

```bash
python main.py run liar2_label_granularity \
  --liar2-dir /data/liar2 --output-dir results/liar2_labels
```

The original headline configuration uses `claim_plus_safe_metadata` and chronological splitting. Claim-only execution is disabled in that application's configuration.

| Scheme | Mapping |
| --- | --- |
| `6class` | Original six labels |
| `3class` | Pants-fire/false → false; barely-true/half-true → mixed; mostly-true/true → true |
| `2class` | Pants-fire/false/barely-true → false; half-true/mostly-true/true → true |
| `2class_halftrue_false` | Pants-fire/false/barely-true/half-true → false; mostly-true/true → true |

Read files under `results/liar2_labels/label_collapse_results/`: `collapse_main_results.csv`, `collapse_bootstrap_results.csv`, `collapse_summary_by_model.csv`, `collapse_effect_size_claim_plus_safe_metadata.csv`, `collapse_seed_level_effect.csv`, `collapse_per_class_recall.csv`, `collapse_chance_floors.csv`, and `collapse_margin_over_chance.csv`. The combined files retain the `scheme` identifier. Per-class-count files are also written in `K6/`, `K3/`, and `K2/`; use the combined files when comparing the two binary schemes, which share `K2/`.

The registered stages are `application` and `notebook`. Both execute the supplied cell sequence; scheme selection is defined inside that preserved application.

### 5.5 Model and seed ensemble experiment

Builds the original LIAR2 chronological ensemble analysis with restricted-metadata text and the required metadata columns. Its seven registered models are TF-IDF+LogReg, TF-IDF+LinearSVM, TF-IDF+LogReg+metadata, Uniform-LR, Uniform-MLP, MiniLM-LogReg-RW, and DA-MiniLM-RWAC. The MiniLM family averages calibrated probabilities across seeds 42–46. DistilBERT is disabled in this application's original call.

```bash
python main.py run ensemble \
  --liar2-dir /data/liar2 --output-dir results/ensemble
```

The analysis reports individual models, pairwise weighted blends, the equal-weight ensemble, a greedy subset search, and an overall leaderboard, using its original validation selection and 2,000-bootstrap configuration.

**This original application prints its result tables and does not export ensemble result CSVs.** Read `results/ensemble/_runner/execution.log` for those tables. Embedding caches are written under `results/ensemble/ensemble_results/`. The public runner returns an execution manifest; the notebook's in-memory `results` dictionary is not returned to the calling process.

Its stages are `application` and `notebook`, with the same original cell sequence.

### 5.6 MultiFC DA-MiniLM application

Runs the original MultiFC DA-MiniLM++ and MiniLM-LogReg-RW comparison. It harmonizes labels into false/mixed/true, records sample-size and split audits, and evaluates seeds 42–46 in both claim-only and restricted-metadata modes.

```bash
python main.py run multifc_da_minilm \
  --multifc-dir /data/multifc --output-dir results/multifc_main
```

The evaluation setups are:

| Setup | Evaluation meaning |
| --- | --- |
| `official_train_to_dev` | Hold out official dev; create inner training/validation subsets from official train |
| `internal_chronological` | Chronological train/validation/test split of the labelled cohort |
| `internal_random_stratified` | Random-stratified train/validation/test split of the labelled cohort |

The original application also runs paired bootstrap, per-class reports, confusion matrices, rolling windows, and official-test prediction generation. Official unlabelled test predictions are prediction files and do not establish test-set accuracy.

Outputs are under `results/multifc_main/multifc_da_minilm_plus_v1_0/`:

| Output | Use |
| --- | --- |
| `multifc_label_harmonization_report.csv`, `sample_size_audit.csv`, `split_summary.csv` | Audit mapped labels, retained rows, and evaluation splits |
| `main_results_per_seed.csv`, `main_results_mean_std_summary.csv` | Inspect individual runs and aggregated model comparisons |
| `lambda_selection_records.csv` | Inspect validation selection |
| `paired_bootstrap_da_minilm_vs_logreg_with_bh.csv`, `paired_bootstrap_condition_summary.csv` | Inspect paired comparisons and condition summaries |
| `per_class_reports.csv`, `confusion_matrix_*.csv` | Inspect class-level performance |
| `rolling_window_da_minilm_plus_results.csv`, `rolling_window_da_minilm_plus_summary.csv`, `rolling_window_macro_f1.png` | Inspect temporal-window performance |
| `official_test_predictions_all_seeds.csv`, `official_test_predictions_seed_ensemble.csv` | Inspect official-test predictions when the unlabelled test cohort is available |
| `experiment_summary.txt` | Read the application's final text summary |

The original final cell also creates `multifc_da_minilm_plus_v1_0.zip` beside the output subdirectory. The registered stages are `application` and `notebook`; each replays the complete supplied application sequence, including its enabled analyses and packaging cell.

### 5.7 MultiFC label-granularity experiment

Compares MultiFC's native three-class labels with a binary collapse that assigns mixed to false. It evaluates DA-MiniLM-RWAC, MiniLM-LogReg-RW, and Uniform-LR using chronological splitting, restricted-metadata text, seeds 0–4, and 2,000 paired-bootstrap resamples. MultiFC has no six-class reference in this application.

```bash
python main.py run multifc_label_granularity \
  --multifc-dir /data/multifc --output-dir results/multifc_labels
```

Read `results/multifc_labels/multifc_label_collapse_results/`. The principal files are `collapse_main_results.csv`, `collapse_bootstrap_results.csv`, `collapse_summary_by_model.csv`, `collapse_effect_size_claim_plus_safe_metadata.csv`, `collapse_seed_level_effect.csv`, `collapse_per_class_recall.csv`, `collapse_chance_floors.csv`, `collapse_margin_over_chance.csv`, and `multifc_label_harmonization_report.csv`.

The harmonization report records how raw verdicts map to the analysis labels. The stages are `application` and `notebook`, with the same supplied cell sequence.

## 6. Independent stages and full notebook replay

These stages are available in **both** `da_minilm_rwac` and `baseline_comparison`. Each starts a fresh process and includes its required preparation. Selecting the experiment name chooses the corresponding notebook's code.

| Stage | Work performed | Principal files under `full_run_results/` |
| --- | --- | --- |
| `distilbert_chronological` | Original five-seed chronological DistilBERT comparison | `distilbert_chronological_multiseed.csv` |
| `main` | DA-MiniLM-RWAC and MiniLM-LogReg-RW main battery with validation selection and summaries | `da_minilm_main_results.csv`, `da_minilm_lambda_selection.csv`, `da_minilm_summary.csv` |
| `paired_bootstrap` | Run the main battery, then bootstrap its paired predictions with 2,000 resamples | Main files plus `da_minilm_paired_bootstrap.csv` |
| `component_ablation` | Prepare models/embeddings, then isolate reweighting and calibration on chronological restricted-metadata data using seed 42 | `da_minilm_component_ablation.csv` |
| `shuffle_control` | Prepare embeddings/splits, then run the original 300-shuffle label-permutation null test with the LogReg head | `shuffle_control_summary.csv`, `shuffle_control_null_scores.npy` |
| `rolling_windows` | Prepare models/embeddings, then evaluate the DA-MiniLM MLP across temporal windows with seeds 42–46 | `da_minilm_rolling_window.csv` |

Run any independent RWAC stage with these complete commands:

```bash
python main.py run da_minilm_rwac --stage distilbert_chronological \
  --liar-dir /data/liar --liar2-dir /data/liar2 --output-dir results/rwac_distilbert

python main.py run da_minilm_rwac --stage main \
  --liar-dir /data/liar --liar2-dir /data/liar2 --output-dir results/rwac_main

python main.py run da_minilm_rwac --stage paired_bootstrap \
  --liar-dir /data/liar --liar2-dir /data/liar2 --output-dir results/rwac_bootstrap

python main.py run da_minilm_rwac --stage component_ablation \
  --liar-dir /data/liar --liar2-dir /data/liar2 --output-dir results/rwac_ablation

python main.py run da_minilm_rwac --stage shuffle_control \
  --liar-dir /data/liar --liar2-dir /data/liar2 --output-dir results/rwac_shuffle

python main.py run da_minilm_rwac --stage rolling_windows \
  --liar-dir /data/liar --liar2-dir /data/liar2 --output-dir results/rwac_rolling
```

To run the corresponding stage in the baseline experiment, replace `da_minilm_rwac` with `baseline_comparison` and choose a new output directory. Bootstrap does not load a previous run's predictions; it creates them by rerunning the original main battery. Ablation, shuffle control, and rolling windows prepare their dependencies without executing the full main battery first.

All available routes are summarized here:

| Experiment | Default stage | Available stages |
| --- | --- | --- |
| `baseline_comparison` | `baselines` | `baselines`, `distilbert_chronological`, `drift`, `main`, `paired_bootstrap`, `component_ablation`, `shuffle_control`, `rolling_windows`, `notebook` |
| `drift_analysis` | `application` | `application`, `prepare`, `notebook` |
| `da_minilm_rwac` | `application` | `application`, `distilbert_chronological`, `main`, `paired_bootstrap`, `component_ablation`, `shuffle_control`, `rolling_windows`, `notebook` |
| `liar2_label_granularity` | `application` | `application`, `notebook` |
| `ensemble` | `application` | `application`, `notebook` |
| `multifc_da_minilm` | `application` | `application`, `notebook` |
| `multifc_label_granularity` | `application` | `application`, `notebook` |

`--stage notebook` is available for every experiment and replays every original code cell in order, including setup, repeated execution, and utility cells. For example:

```bash
python main.py run baseline_comparison --stage notebook \
  --liar-dir /data/liar --liar2-dir /data/liar2 --output-dir results/baseline_notebook

python main.py run da_minilm_rwac --stage notebook \
  --liar-dir /data/liar --liar2-dir /data/liar2 --output-dir results/rwac_notebook
```

Baseline notebook replay includes its later main-model analyses in addition to the default baseline sections. RWAC notebook replay also executes its final backup utility cell, which the `application` route excludes. For the other five experiments, `application` and `notebook` currently use the same original cell sequence.

Before a costly run, append `--dry-run` to inspect its source plan:

```bash
python main.py run da_minilm_rwac --stage component_ablation \
  --liar-dir /data/liar --liar2-dir /data/liar2 \
  --output-dir results/rwac_ablation_preview --dry-run
```

## 7. Python class utilization

Import the class from a script in the codebase directory. Importing it does not start an experiment. Each execution method starts a fresh Python worker and returns a manifest dictionary when the run completes.

```python
from experiment_runner import JournalExperimentRunner

def main():
    runner = JournalExperimentRunner()
    print(runner.list_experiments())
    print(runner.verify_sources())

    manifest = runner.run_drift_analysis(
        liar2_dir="/data/liar2",
        output_dir="results/drift_python",
    )
    print(manifest["status"], manifest["output_directory"])

if __name__ == "__main__":
    main()
```

Every experiment has its own method:

| Method | Dataset keyword arguments | Default stage |
| --- | --- | --- |
| `run_baseline_comparison(...)` | `liar_dir`, `liar2_dir` | `baselines` |
| `run_drift_analysis(...)` | `liar2_dir` | `application` |
| `run_da_minilm_rwac(...)` | `liar_dir`, `liar2_dir` | `application` |
| `run_liar2_label_granularity(...)` | `liar2_dir` | `application` |
| `run_ensemble(...)` | `liar2_dir` | `application` |
| `run_multifc_da_minilm(...)` | `multifc_dir` | `application` |
| `run_multifc_label_granularity(...)` | `multifc_dir` | `application` |

These methods accept `stage`, `output_dir`, and `dry_run` as additional keyword arguments. The generic `run()` has the same options and accepts the experiment name first:

```python
from experiment_runner import JournalExperimentRunner

runner = JournalExperimentRunner()

# Plan inspection: no scientific execution.
plan = runner.execution_plan("da_minilm_rwac", stage="shuffle_control")
preview = runner.run_liar2_label_granularity(dry_run=True)

# Execute the main battery from the baseline experiment.
manifest = runner.run(
    "baseline_comparison",
    stage="main",
    liar_dir="/data/liar",
    liar2_dir="/data/liar2",
    output_dir="results/baseline_main_python",
)
```

Convenience methods select stages from **the `da_minilm_rwac` experiment**:

| Convenience method | Selected stage |
| --- | --- |
| `run_distilbert_chronological(...)` | `distilbert_chronological` |
| `run_da_minilm_main(...)` | `main` |
| `run_paired_bootstrap(...)` | `paired_bootstrap` |
| `run_component_ablation(...)` | `component_ablation` |
| `run_shuffle_control(...)` | `shuffle_control` |
| `run_rolling_windows(...)` | `rolling_windows` |

Pass dataset and output arguments to these methods; the stage is already selected internally. Use generic `run()` when choosing the baseline notebook's corresponding stage. `run_notebook(experiment, **options)` selects full notebook replay for any experiment.

```python
from experiment_runner import JournalExperimentRunner

runner = JournalExperimentRunner()
manifest = runner.run_component_ablation(
    liar_dir="/data/liar",
    liar2_dir="/data/liar2",
    output_dir="results/ablation_python",
)

# Full replay can also be previewed before execution.
plan = runner.run_notebook("ensemble", dry_run=True)
```

Execution methods return the run manifest, not models, prepared DataFrames, or the notebook's scientific result variables. Read exported CSVs from the output locations described above. A failed worker causes `RuntimeError` in the parent; the original exception is recorded in its manifest and log.

## 8. Outputs, troubleshooting, and verification scope

Every worker execution creates two runner files directly under the chosen output root:

| File | Contents |
| --- | --- |
| `_runner/execution.log` | Original console output, step progress, printed tables, and exception tracebacks |
| `_runner/run_manifest.json` | Experiment/stage, source hashes, dataset paths, output mapping, installed package versions, timestamps, completed steps, and run status |

Scientific files retain the original names and subdirectory structure. A completed manifest confirms that the selected route finished. If a run fails, use the log and its `error`/`traceback` fields to identify the original failing step.

| Symptom | Next action |
| --- | --- |
| Missing dataset directory or split file | Check the selected directory flag and required filenames in Section 2 |
| Missing dependency | Install the requirements in the active Python environment; install `datasets` for DistilBERT routes |
| Model download/cache failure | Make the exact pretrained weights available to that environment |
| RWAC cohort assertion failure | Check that the original LIAR2 release and preprocessing inputs produce 22,923 final rows |
| Source verification failure | Restore the original archived cells or slices before running the preserved routes |
| Unexpected output location | Check the retained subdirectory name and `path_aliases` in the manifest |

The original MultiFC DA-MiniLM code explicitly uses `LogisticRegression(multi_class=...)`; the supplied core requirements retain `scikit-learn<1.8` for that call.

Run the bundled checks with:

```bash
python -m unittest discover -s tests -v
```

The recorded validation includes 16 passing checks for source preservation, routing, shared notebook namespaces, filesystem mapping, terminal preprocessing, failure manifests, and independent-stage dependencies. D01/D02/D03/D05 summary equality was checked against unadapted original execution on the same synthetic fixture. Full training, pretrained-model inference, and numerical reproduction of the journal results have not been verified because the original datasets, pretrained weights, and historical environment lockfile were not supplied.
