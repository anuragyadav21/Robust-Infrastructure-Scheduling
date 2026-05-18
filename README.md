# Robust optimisation of infrastructure scheduling

**Co-optimising urban construction schedules with traffic disruption under operational uncertainty**

| | |
|--|--|
| **Domain** | Systems engineering · operations research · smart infrastructure |
| **Case study** | ~2.4 km urban arterial corridor reconstruction (~40 working days, **24 tasks**) |
| **Methods** | Mixed-integer programming (MIP) · Monte Carlo simulation · scenario-based robust optimisation |
| **Stack** | Python 3.10+ · PuLP (CBC) · NumPy · Pandas · Matplotlib · SciPy |

Human-readable narrative and equations: **[`REPORT.md`](REPORT.md)**. Submission-friendly PDF: **[`Final_Report.pdf`](Final_Report.pdf)** — embeds **Figures 1–5** (distributions, Gantt, Pareto, tail risk, λ trade-off) plus a **01–09 thumbnail montage** when plots exist; rebuild with `scripts/build_final_report_pdf.py` after `pip install markdown xhtml2pdf` (run both model scripts first so `plots/` and `plots_tradeoff/` are populated).

**Repository:** [github.com/anuragyadav21/Robust-Infrastructure-Scheduling](https://github.com/anuragyadav21/Robust-Infrastructure-Scheduling)

---

## Contents

1. [What this repository contains](#what-this-repository-contains)
2. [Repository layout (current)](#repository-layout-current)
3. [Conceptual overview](#conceptual-overview)
4. [Problem at a glance](#problem-at-a-glance)
5. [Environment and dependencies](#environment-and-dependencies)
6. [Configure paths and run](#configure-paths-and-run)
7. [Data dictionary](#data-dictionary)
8. [Scripts and pipeline stages](#scripts-and-pipeline-stages)
9. [Results snapshot (committed CSVs)](#results-snapshot-committed-csvs)
10. [Outputs and figures](#outputs-and-figures)
11. [Solver and weight defaults](#solver-and-weight-defaults)
12. [Related files and housekeeping](#related-files-and-housekeeping)
13. [Limitations and extensions](#limitations-and-extensions)
14. [Citation](#citation)

---

## What this repository contains

| Layer | Purpose |
|-------|---------|
| **`Baseline reference datasets/`** | Case-study inputs: cleaned CSV/JSON under `Dependency2/`, plus an Excel workbook source. |
| **`Model and solutions/`** | Main script: baseline vs optimised vs robust MIP, 10k-run Monte Carlo per variant, sensitivity sweeps, figures **01–05**. |
| **`O:P with pareto fronts/`** | Trade-off script: Pareto (cost–duration, cost–traffic), λ risk–cost curve, tail tables, figures **06–09**. |
| **Documentation** | `README.md` (this file), `REPORT.md` (full technical report), `Final_Report.pdf` (PDF export). |
| **Packaging** | `requirements.txt`, `LICENSE` (MIT), `scripts/build_final_report_pdf.py` (optional PDF rebuild). |

There is **one** canonical dataset for code: **`Baseline reference datasets/Dependency2/`**. There is no `Redundant/`, `Initial/`, or duplicate output tree in this checkout.

---

## Repository layout (current)

Below is the **logical** layout of the repository. **Figures are not listed** at the root of each code folder: after you run the scripts, PNGs appear under `Model and solutions/plots/` and `O:P with pareto fronts/plots_tradeoff/` (those directories are **gitignored** so clones stay small—regenerate locally).

```
.
├── README.md
├── REPORT.md
├── Final_Report.pdf
├── LICENSE
├── requirements.txt
├── editROBUST OPTIMISATION OF INFRASTRUCTURE SCHEDULING.docx
├── scripts/
│   └── build_final_report_pdf.py
│
├── Baseline reference datasets/
│   ├── Infrastructure_Project_Dataset.xlsx
│   └── Dependency2/
│       ├── cost_function_spec.json
│       ├── cost_parameters_events.csv
│       ├── cost_parameters_resources.csv
│       ├── daily_schedule_grid.csv
│       ├── dependencies_clean.csv
│       ├── equipment_pool.csv
│       ├── resource_limits.csv
│       ├── resource_usage_daily.csv
│       ├── resources_clean.csv
│       ├── risk_clean.csv
│       ├── tasks_clean.csv
│       ├── time_blocks.csv
│       ├── traffic_schedule_multipliers.csv
│       └── uncertainty_distributions.json
│
├── Model and solutions/
│   ├── optimisation_model.py
│   ├── results_summary.csv
│   ├── actionable_decisions.csv
│   ├── sensitivity_lambda.csv
│   ├── sensitivity_gamma.csv
│   ├── sensitivity_alpha.csv
│   ├── plots/                    ← gitignored; 01–05 PNGs after `optimisation_model.py`
│   └── …
│
└── O:P with pareto fronts/
    ├── tradeoff_study.py
    ├── pareto_cost_duration.csv
    ├── pareto_cost_traffic.csv
    ├── risk_cost_curve.csv
    ├── tail_risk_summary.csv
    ├── plots_tradeoff/           ← gitignored; 06–09 PNGs after `tradeoff_study.py`
    └── …
```

### Folder roles

| Path | Role |
|------|------|
| `Baseline reference datasets/Dependency2/` | **Required inputs** for both Python scripts (tasks, precedence, costs, limits, traffic multipliers, uncertainty JSON, cost spec JSON). |
| `Baseline reference datasets/Infrastructure_Project_Dataset.xlsx` | Consolidated workbook; Dependency2 is the model-ready export. |
| `Model and solutions/` | Core optimisation + simulation + sensitivities; CSVs committed; figures **01–05** in `plots/` after you run the main script. |
| `O:P with pareto fronts/` | Pareto and tail-risk study; CSVs committed; figures **06–09** in `plots_tradeoff/` after you run the trade-off script. |
| `editROBUST OPTIMISATION OF INFRASTRUCTURE SCHEDULING.docx` | Editable report / thesis draft (optional; not used by code). |

---

## Conceptual overview

Construction and traffic are often planned in **silos**. This work couples them in one decision model:

1. **Simulate** — Monte Carlo (10,000 iterations, fixed seed in code) with duration shocks, equipment and weather events, and **cascade** propagation along finish-start links.
2. **Optimise** — A MIP chooses each task’s **calendar start day** and **one of four intraday time blocks** (B1–B4), trading labour cost, makespan, and a traffic-disruption proxy.
3. **Robustify** — A **λ** parameter blends expected labour cost across four scenarios with a worst-case-style labour multiplier.

Details, notation, and discussion are in [`REPORT.md`](REPORT.md).

---

## Problem at a glance

### Decision variables (per task *t*)

| Symbol / code idea | Type | Meaning |
|--------------------|------|---------|
| `s[t]` | Integer | Start day (within the model horizon, up to 55-day window in code) |
| `b[t,k]` | Binary | Task *t* assigned to block *k* ∈ {B1, B2, B3, B4} |

### Time blocks (traffic-aware proxy)

| Block | Typical period | Peak? | Role in model |
|-------|----------------|--------|----------------|
| **B1** | Early | No | Lower delay multiplier (DM) |
| **B2** | Morning peak | **Yes** | High DM |
| **B3** | Midday | No | Preferred off-peak window |
| **B4** | Afternoon peak | **Yes** | High DM + peak social penalty |

Task-level DMs live in `tasks_clean.csv` (`dm_block_*` columns). Precedence in the MIP uses **planned** durations; stochastic durations appear in the simulation layer.

### Objectives (high level)

The deterministic objective combines **labour cost** (block-dependent DM), **makespan**, **traffic disruption** (impact × DM × duration), and a **social / peak** term. The robust variant replaces/scales the labour part with a **scenario blend** controlled by **λ** (see [`REPORT.md`](REPORT.md) §4–5).

---

## Environment and dependencies

```bash
cd /path/to/this/repository
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists: `numpy`, `pandas`, `matplotlib`, `scipy`, `pulp`, `openpyxl` (useful if you load `Infrastructure_Project_Dataset.xlsx` from Python in future extensions).

- **Python:** 3.10+ recommended.  
- **Solver:** CBC via PuLP (bundled). Commercial solvers (Gurobi, CPLEX) can be used with PuLP if you have a licence.

---

## Configure paths and run

### Defaults (clone-and-run)

Both scripts set **`DATA`**, **`OUT`**, and **`PLOTS`** with `pathlib` relative to the repository root—**no manual path edits** are required for a standard checkout.

| Script | Reads data from | Writes CSVs to | Writes PNGs to |
|--------|-----------------|----------------|----------------|
| `Model and solutions/optimisation_model.py` | `Baseline reference datasets/Dependency2/` | Same folder as the script (`OUT`) | `Model and solutions/plots/` |
| `O:P with pareto fronts/tradeoff_study.py` | (same `DATA`) | Same folder as the script (`OUT`) | `O:P with pareto fronts/plots_tradeoff/` |

The `plots/` and `plots_tradeoff/` directories are **gitignored** (regenerate after clone). To customise locations, edit the `REPO` / `DATA` / `OUT` / `PLOTS` assignments at the top of each script.

### Commands

```bash
python "Model and solutions/optimisation_model.py"
python "O:P with pareto fronts/tradeoff_study.py"
```

Run **main model first**, then **trade-off study**. Runtime is typically **1–3 minutes** per script on a laptop (MIP + Monte Carlo).

### Rebuild `Final_Report.pdf`

```bash
pip install markdown xhtml2pdf
python scripts/build_final_report_pdf.py
```

(Uses `REPORT.md` as the single source of truth.)

---

## Data dictionary

**Primary directory:** `Baseline reference datasets/Dependency2/`

| File | Approx. scale | Description |
|------|----------------|-------------|
| `tasks_clean.csv` | 24 rows | Tasks: resources, workers, equipment flags, `road_impact_level`, planned duration, DM per block |
| `dependencies_clean.csv` | 33 rows | `FS` and `SS` precedence edges |
| `cost_parameters_resources.csv` | 9 resource types | Unit labour rates, overtime, idle fractions |
| `cost_parameters_events.csv` | 10 event types | Penalties and disruption parameters |
| `equipment_pool.csv` | 5 types | Shared equipment caps |
| `resource_limits.csv` | 72 rows | Weekly crew caps |
| `resource_usage_daily.csv` | — | Daily resource usage (supporting / calibration data) |
| `resources_clean.csv` | — | Resource catalogue |
| `risk_clean.csv` | 22 rows | Historical / narrative risk events for plots and attribution |
| `traffic_schedule_multipliers.csv` | 388 rows | Day × block dynamic DM and delay index (extension / validation) |
| `daily_schedule_grid.csv` | 97 rows | Fine schedule grid reference |
| `time_blocks.csv` | 4 rows | Block definitions |
| `uncertainty_distributions.json` | — | Monte Carlo factor definitions |
| `cost_function_spec.json` | — | Structured description of cost components and CSV linkage |

---

## Scripts and pipeline stages

### `Model and solutions/optimisation_model.py`

| Stage | Description |
|-------|-------------|
| Baseline | Forward-pass style schedule with **peak-biased** blocks for high–road-impact tasks (status-quo proxy). |
| MIP — optimised | Scalarised objective (default α, β, γ in code / report). |
| MIP — robust | Scenario-weighted labour term with **λ > 0**. |
| Monte Carlo | Large-sample simulation per variant (see code for *n*). |
| Sensitivity | Sweeps over γ (traffic), λ (risk), α (cost weight). |
| Plots | `01_distributions` … `05_cascade` (under `plots/` after run). |

### `O:P with pareto fronts/tradeoff_study.py`

| Analysis | Description |
|----------|-------------|
| Cost vs duration | Pareto-style sweep over makespan caps. |
| Cost vs traffic | Sweep forcing more tasks into peak blocks. |
| Risk–cost | λ grid with simulated cost statistics. |
| Tail risk | Percentile table and tail-focused plots. |
| Plots | `06_tradeoff_curves` … `09_lambda_tradeoff` (under `plots_tradeoff/` after run). |

---

## Results snapshot (committed CSVs)

**Headline deltas vs baseline (from `results_summary.csv`):** **−8.8%** mean cost, **−9.6%** P90 cost, **−26.6%** mean traffic impact; mean / P90 duration essentially unchanged under the current weights.

| Metric | Baseline | Optimised | Robust |
|--------|----------|-----------|--------|
| Mean duration (days) | 52.38 | 52.39 | 52.38 |
| P90 duration (days) | 53.03 | 53.03 | 53.04 |
| Mean cost (USD) | ~1.35M | ~1.24M | ~1.23M |
| P90 cost (USD) | ~1.62M | ~1.47M | ~1.46M |
| Mean traffic impact | 592 | 435 | 434 |

**Reading:** The optimiser **cuts mean and tail cost and traffic disruption** with **negligible change** in mean / P90 simulated duration under the current weights (see [`REPORT.md`](REPORT.md) §8 for full tables and tail percentiles). Interpretation of “overrun” flags is nuanced.

### Tail cost percentiles (`O:P with pareto fronts/tail_risk_summary.csv`)

| Percentile | Baseline cost (USD) | Optimised | Robust |
|------------|---------------------|-----------|--------|
| P50 | ~1.24M | ~1.21M | ~1.21M |
| P90 | ~1.51M | ~1.46M | ~1.46M |
| P99 | ~1.83M | ~1.80M | ~1.80M |

Per-task deltas (blocks, starts, DM savings, USD deltas) are in `actionable_decisions.csv`.

---

## Outputs and figures

### CSV artefacts

| File | Location | Role |
|------|----------|------|
| `results_summary.csv` | `Model and solutions/` | One row per headline KPI × variant |
| `actionable_decisions.csv` | `Model and solutions/` | Task-level baseline vs optimal |
| `sensitivity_lambda.csv`, `sensitivity_gamma.csv`, `sensitivity_alpha.csv` | `Model and solutions/` | Weight sweeps |
| `pareto_cost_duration.csv`, `pareto_cost_traffic.csv` | `O:P with pareto fronts/` | Pareto sweeps |
| `risk_cost_curve.csv`, `tail_risk_summary.csv` | `O:P with pareto fronts/` | λ curve and tail table |

### Figures (after running scripts)

| ID | Filename | Location (local, gitignored) |
|----|----------|-------------------------------|
| 01–05 | `01_distributions.png` … `05_cascade.png` | `Model and solutions/plots/` |
| 06–09 | `06_tradeoff_curves.png` … `09_lambda_tradeoff.png` | `O:P with pareto fronts/plots_tradeoff/` |

To **commit figures** to git for GitHub Pages or a static report, remove or narrow the `plots/` entries in `.gitignore`, or copy PNGs into a tracked folder (e.g. `docs/figures/`) yourself.

---

## Solver and weight defaults

| Setting | Typical value in code |
|---------|------------------------|
| Engine | CBC (`pulp.PULP_CBC_CMD`) |
| Time limit | Order of 45–60 s per solve |
| Relative MIP gap | ~8% |
| Horizons | Planned ~40 days; optimisation window up to **55** days |

Scenario IDs **S1–S4** and default **α, β, γ, λ** for optimised vs robust variants are tabulated in [`REPORT.md`](REPORT.md) §5 and Appendix B.

---

## Related files and housekeeping

| Item | Note |
|------|------|
| [`REPORT.md`](REPORT.md) | Full methodology, reproducibility, file inventory, parameter tables. |
| [`Final_Report.pdf`](Final_Report.pdf) | PDF export of the technical report (regenerate with `scripts/build_final_report_pdf.py`). |
| [`LICENSE`](LICENSE) | MIT licence. |
| `editROBUST OPTIMISATION OF INFRASTRUCTURE SCHEDULING.docx` | Word source; not executed by Python. |
| `~$*.docx` | Microsoft Word lock file—safe to delete; listed in `.gitignore`. |

### Publishing to GitHub

Remote: `https://github.com/anuragyadav21/Robust-Infrastructure-Scheduling.git` (branch `main`).

---

## Limitations and extensions

**Limitations**

- Traffic is a **block-based proxy**, not a full dynamic traffic assignment model.  
- MIP uses **fixed planned durations** for precedence; duration risk is mainly in **simulation**.  
- Single project / single corridor scope.

**Natural extensions** (see also [`REPORT.md`](REPORT.md) §10–11)

- GIS / network models for delay propagation.  
- Multi-objective metaheuristics (e.g. NSGA-II) without scalarisation.  
- Discrete-event simulation (e.g. SimPy) for finer resource contention.  
- Dashboards for λ, γ, and scenario exploration.

---

## Citation

> *Robust Optimisation of Infrastructure Scheduling under Traffic and Operational Uncertainty*

For reproduction steps and validation checks, see [`REPORT.md`](REPORT.md) §9.

---

*README and outputs aligned for reproducible clone-and-run (May 2026).*
