# Technical Report

## Robust Optimisation of Infrastructure Scheduling under Traffic and Operational Uncertainty

**Document type:** Project technical report  
**Case study:** 2.4 km urban arterial corridor reconstruction  
**Horizon:** ~40 planned working days (55-day optimisation window)  
**Tasks:** 24 (T01–T24)  
**Methods:** Mixed-integer programming · Monte Carlo simulation · Scenario-based robust optimisation  

---

## Document map

| § | Section |
|---|---------|
| [1](#1-executive-summary) | Executive summary |
| [2](#2-introduction-and-motivation) | Introduction and motivation |
| [3](#3-system-description) | System description |
| [4](#4-mathematical-formulation) | Mathematical formulation |
| [5](#5-solution-methodology) | Solution methodology |
| [6](#6-uncertainty-and-simulation) | Uncertainty and simulation |
| [7](#7-experimental-design) | Experimental design |
| [8](#8-results) | Results |
| [9](#9-reproducibility) | Reproducibility |
| [10](#10-discussion-and-recommendations) | Discussion and recommendations |
| [11](#11-conclusions) | Conclusions |
| [A](#appendix-a-file-inventory) | Appendix A: File inventory |
| [B](#appendix-b-parameter-tables) | Appendix B: Parameter tables |
| [Figures](#key-result-figures-embedded) | Key result figures (embedded) |

---

## 1. Executive summary

### 1.1 Problem

Urban road reconstruction must satisfy **construction logic** (precedence, crews, equipment) and **traffic constraints** (peak volumes, lane closures, societal disruption). In practice these are planned in silos. Uncertainty — weather, breakdowns, material delays, absenteeism — turns small disturbances into **cascade delays** that deterministic critical-path methods do not quantify.

### 1.2 Approach

This project builds an integrated decision-support pipeline:

1. **Deterministic MIP** — assign each task a start day and one of four daily time blocks (B1–B4), minimising a weighted sum of labour cost, makespan, and traffic disruption, subject to precedence.
2. **Stochastic evaluation** — 10,000-iteration Monte Carlo with cascade propagation along finish-start dependencies.
3. **Robust optimisation** — blend expected and worst-case labour cost across four disruption scenarios (parameter λ).
4. **Trade-off analysis** — Pareto sweeps over makespan caps and peak-block usage; tail-risk and CVaR curves.

### 1.2.1 End-to-end workflow (system view)

```text
  Baseline reference datasets/Dependency2/  (tasks, precedence, costs, uncertainty JSON)
                        │
                        ▼
              Deterministic MIP (PuLP / CBC)
         start day + time block (B1–B4) per task
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Baseline policy              Optimised + robust
   (peak-biased blocks)         (scenario-weighted labour)
          │                           │
          └─────────────┬─────────────┘
                        ▼
           Monte Carlo (10k runs, fixed seed)
           cascades, delays, cost & traffic KPIs
                        │
                        ▼
        Pareto / λ sweeps / tail metrics (tradeoff_study.py)
                        │
                        ▼
     Decision outputs: CSV summaries, actionable_decisions, plots 01–09
```

### 1.3 Principal findings

*(Monte Carlo n = 10,000, seed = 42; see `Model and solutions/results_summary.csv`.)*

| Finding | Magnitude |
|---------|-----------|
| Mean total cost reduction (optimised vs baseline) | **8.8%** (~$119k) |
| P90 total cost reduction | **9.6%** (~$156k) |
| Mean traffic disruption index reduction | **26.6%** |
| Mean duration change (simulated) | **≈0%** (negligible vs baseline under current weights and CBC solution) |
| Robust vs optimised cost | Comparable mean cost; robust slightly improves some tail cost percentiles |

The optimised schedule **reduces expected cost and traffic disruption** by moving high-impact work out of peak traffic blocks. Under the **current** objective weights and solver output, the **mean simulated duration** is **essentially unchanged** versus baseline (see `results_summary.csv`); cost–duration trade-offs still appear explicitly on the **Pareto** and **tail** curves in §8.2–8.3 and in `O:P with pareto fronts/`.

### 1.4 Deliverables

- Executable models: `optimisation_model.py`, `tradeoff_study.py`
- Decision tables: `actionable_decisions.csv`, sensitivity and Pareto CSVs
- Nine publication-quality figures (distributions, Gantt, sensitivities, Pareto, tail risk)
- Human-readable documentation: `README.md`, this report (`REPORT.md`), and **`Final_Report.pdf`** — PDF export that **embeds key result figures** (see §8); rebuild with `scripts/build_final_report_pdf.py` after `pip install markdown xhtml2pdf`, with plot folders populated by running both Python scripts first.

---

## 2. Introduction and motivation

### 2.1 Context

The case study reflects a **mixed utility / pavement rehabilitation** project on a **2.4 km arterial** carrying up to **~3,200 vehicles/hour** at peak. The planned budget is on the order of **$4.25M** with contingency. Work spans mobilisation, utilities, traffic management, milling, subbase, drainage, paving, and handover — **24 linked tasks** with heterogeneous resource needs and road-impact levels (0–3).

### 2.2 Gap in current practice

| Practice | Limitation |
|----------|------------|
| CPM / Primavera scheduling | Ignores time-of-day traffic effects on productivity and social cost |
| Traffic management plans | React to construction dates rather than co-designing them |
| Deterministic plans | No tail-risk metric for board / city stakeholders |
| Independent KPIs | Contractor minimises duration; city minimises congestion — no Pareto surface |

### 2.3 Research questions

1. How much **cost and traffic impact** can joint scheduling recover vs an unoptimised baseline?
2. How do **cascade effects** under uncertainty inflate duration and cost distributions?
3. What is the **cost of robustness** (λ) in deterministic premium vs P90 improvement?
4. Which **task-level decisions** (block, start day) are actionable for a site manager?

### 2.4 Thesis

> Co-optimising construction start times and time-block assignments with traffic-sensitive delay multipliers — and evaluating schedules under explicit uncertainty — yields measurably better system outcomes than decoupled planning, with quantifiable trade-offs between mean performance and tail risk.

---

## 3. System description

### 3.1 Construction system

**Tasks** (`tasks_clean.csv`): each row defines:

- Planned start/finish, actuals (for calibration narrative)
- `primary_resource_type`, `workers_required`, equipment flags
- `road_impact_level` (0 = none, 3 = severe lane / corridor impact)
- Block-specific delay multipliers `dm_block_B1_early` … `dm_block_B4_peak_pm`

**Precedence** (`dependencies_clean.csv`):

- **FS** (finish-start): successor starts after predecessor duration
- **SS** (start-start): concurrent coordination (e.g. parallel utility runs)

**Resources:**

- `cost_parameters_resources.csv` — daily unit costs, overtime, idle penalties
- `equipment_pool.csv` — shared cranes, mills, pavers
- `resource_limits.csv` — weekly caps driving crew stacking risk

### 3.2 Traffic system

Traffic is not modelled as a full dynamic network assignment in the implemented MIP. Instead, **time-block choice** proxies congestion:

| Block | Label | Peak | Hours (approx.) | Role |
|-------|-------|------|-----------------|------|
| B1 | Early | No | 1 | Low DM, early mobilisation |
| B2 | Peak AM | **Yes** | 2 | Highest DM for sensitive tasks |
| B3 | Midday | No | 7 | Preferred work window |
| B4 | Peak PM | **Yes** | 2 | High DM + social cost term |

**Dynamic data** in `traffic_schedule_multipliers.csv` provides day-level `dynamic_dm` and `delay_index` for validation and future extension; the core MIP uses task-level DM columns.

### 3.3 Coupling mechanism

```
  Block assignment b[t,k]
           │
           ▼
  Delay multiplier DM[t,k]  ──►  Labour cost ∝ workers × unit_cost × duration × DM
           │
           ▼
  Traffic score ∝ Σ impact[t] × DM[t,k] × duration[t]
           │
           ▼
  (MC layer) Peak blocks draw higher stochastic traffic slowdown
```

### 3.4 Cost accounting

`cost_function_spec.json` decomposes total cost into labour, equipment, material escalation, penalties, social / traffic externalities, and overhead. The MIP implements the **labour + traffic proxy + social peak penalty** terms; Monte Carlo adds **delay penalties** ($8,500/day beyond planned makespan), **breakdown costs**, and stochastic duration.

---

## 4. Mathematical formulation

Notation below uses plain text and Unicode so the narrative stays readable in both Markdown viewers and the PDF export. (Vector forms match the PuLP implementation.)

### 4.1 Sets and indices

- **Tasks:** 𝒯, with |𝒯| = 24  
- **Blocks:** 𝒦 = {B1, B2, B3, B4}  
- **FS arcs:** 𝒜_FS; **SS arcs:** 𝒜_SS  

### 4.2 Decision variables

| Symbol | Domain | Meaning |
|--------|--------|---------|
| *s*<sub>*t*</sub> | non-negative integer | Start day of task *t* |
| *b*<sub>*tk*</sub> | binary | 1 if task *t* is assigned to block *k* |
| *C*<sub>max</sub> | non-negative integer | Project makespan |

### 4.3 Constraints

**Unique block** — each task occupies exactly one intraday block:

```
Σ_{k∈𝒦} b_{tk} = 1     for all t ∈ 𝒯
```

**FS precedence** (planned duration *d̄*<sub>*t*</sub> fixed in the MIP):

```
s_j ≥ s_i + d̄_i     for all (i,j) ∈ 𝒜_FS
```

**SS precedence:**

```
s_j ≥ s_i     for all (i,j) ∈ 𝒜_SS
```

**Makespan:**

```
C_max ≥ s_t + d̄_t     for all t ∈ 𝒯
```

**Optional (trade-off study):** makespan cap *C*<sub>max</sub> ≤ *C̄*; minimum count of peak-block tasks for the cost–traffic Pareto sweep.

### 4.4 Cost expressions

**Labour (deterministic):**

```
L = Σ_{t∈𝒯} Σ_{k∈𝒦} b_{tk} · w_t · c_t · d̄_t · DM_{tk}
```

**Traffic proxy:**

```
T = Σ_{t,k} b_{tk} · I_t · DM_{tk} · d̄_t
```

where *I*<sub>*t*</sub> is `road_impact_level` from `tasks_clean.csv`.

**Social peak penalty:**

```
S = Σ_{t,k} b_{tk} · 𝟙[k ∈ peak] · I_t · h_k · π
```

with block hours *h*<sub>*k*</sub> and scale *π* = 100 in code.

### 4.5 Objectives

**Deterministic weighted sum:**

```
min   α·L + β·10^4·C_max + γ·T + S
```

**Robust labour term** (scenarios *s* ∈ {S1…S4} with probabilities *p*<sub>*s*</sub> and labour multipliers *m*<sub>*s*</sub>):

```
L^rob = (1−λ)·Σ_s p_s·L·m_s  +  λ·L·max_s m_s
```

with risk-aversion weight *λ* ∈ [0, 1].

### 4.6 Baseline policy

The baseline is **not** the raw planned CSV schedule alone. The code constructs a **realistic unoptimised** policy:

- Forward pass respecting FS precedence from planned starts
- High-impact tasks (`impact ≥ 3`) randomly assigned to peak blocks B2/B3 mix (~60% peak bias)

This represents **status-quo** behaviour: work proceeds in logic order without traffic-aware block optimisation.

---

## 5. Solution methodology

### 5.1 Implementation stack

| Component | Tool |
|-----------|------|
| MIP modelling | PuLP 2.x |
| Solver | CBC (`PULP_CBC_CMD`) |
| Simulation | NumPy vectorised loops |
| Statistics | SciPy (`gaussian_kde`) |
| Visualisation | Matplotlib (Agg backend) |

### 5.2 Solution variants

| Variant | α | β | γ | λ | Purpose |
|---------|---|---|---|---|---------|
| Baseline | — | — | — | — | Heuristic peak-biased blocks |
| Optimised | 1.0 | 0.30 | 0.50 | 0.0 | Primary balanced solution |
| Robust | 1.0 | 0.25 | 0.50 | 0.35 | Scenario-hardened labour cost |

### 5.3 Sensitivity experiments

| Sweep | Range | Output |
|-------|-------|--------|
| γ (traffic weight) | 0.0 → 2.0 (9 points) | Peak task count, average DM |
| λ (risk aversion) | 0.0 → 1.0 (9 points) | P90 duration/cost, overrun rate |
| α (cost weight) | 0.2 → 2.5 (8 points) | Makespan vs peak usage |

### 5.4 Pareto and risk curves (`tradeoff_study.py`)

1. **Cost–duration:** solve with makespan caps 42…57 days; trace efficient frontier.
2. **Cost–traffic:** force fraction of tasks in {B2,B4} from 0% to 60%.
3. **Risk–cost:** for each λ, record E[cost], P90, P95, P99 from 3,000 MC runs.

### 5.5 Computational effort

- Single MIP: < 60 s (time limit), 8% optimality gap
- Full pipeline with plots: few minutes on a laptop
- 10,000 MC runs × 3 variants: dominant runtime (~minutes)

---

## 6. Uncertainty and simulation

### 6.1 Random inputs

Defined in `uncertainty_distributions.json` and mirrored in code:

| Factor | Distribution | Parameters |
|--------|--------------|------------|
| Global duration shock | Triangular | 1.0, 1.12, 1.45 |
| Peak traffic multiplier | Uniform | [1.30, 1.80] |
| Off-peak multiplier | Uniform | [1.00, 1.20] |
| Volume noise | Normal | μ=1, σ=0.08 (clipped) |
| Material delay | Bernoulli(0.25) + Geometric | Affects subset of tasks |
| Equipment breakdown | Bernoulli(0.05) | + cost ~ N(13000, 2500) |
| Weather | Bernoulli(0.08) | + delay triangular 0.5–3 d |
| Absenteeism | Beta(1.5, 10) | Scales effective crew |
| Cascade factor | Triangular | 1.0, 1.2, 1.6 |

### 6.2 Cascade logic

When the realised duration of task *i* exceeds its plan, each successor *j* may absorb amplified delay:

```
d_j ← max( d_j ,  d̄_j·κ + 0.4·(d_i − d̄_i) )
```

with *κ* drawn each Monte Carlo iteration. Depth-first propagation follows the FS adjacency list `SUCC`.

### 6.3 Outputs per iteration

- **Project duration** = max<sub>*t*</sub> (*s*<sub>*t*</sub> + *d*<sup>actual</sup><sub>*t*</sub>)  
- **Total cost** = stochastic labour + breakdown cost + delay penalty × max(0, duration − *C̄*<sub>max</sub>)  
- **Traffic index** = Σ<sub>*t*</sub> *I*<sub>*t*</sub> · DM<sub>*t*</sub> · *d*<sup>actual</sup><sub>*t*</sub>  
- **Overrun flag** = 1 if simulated cost exceeds 105% of planned deterministic cost  

### 6.4 Emergence phenomena (observed)

1. **Bimodal baseline duration** — mixture of “normal” and “cascade” regimes (**Figure 5** — cascade / regime plot).  
2. **Cost–duration coupling** — upper tail of duration aligns with cost spikes.  
3. **Risk category dominance** — equipment and traffic categories drive observed delays in `risk_clean.csv`.

---

## 7. Experimental design

### 7.1 Data split

| Role | Path |
|------|------|
| **Primary (model inputs)** | `Baseline reference datasets/Dependency2/` |
| **Source workbook** | `Baseline reference datasets/Infrastructure_Project_Dataset.xlsx` (consolidated case data; scripts read CSV/JSON from Dependency2 only) |
| **Optional prose draft** | `editROBUST OPTIMISATION OF INFRASTRUCTURE SCHEDULING.docx` (not used by code) |

Older repository layouts kept duplicate datasets under `Redundant/` or `files (2)/`; **this checkout does not include those folders**—Dependency2 is the single canonical input tree.

### 7.2 Randomness control

- `np.random.seed(42)` in both scripts
- MC sample size: **10,000** (main model), **3,000–5,000** (inner λ sweep)

### 7.3 Evaluation metrics

| Metric | Definition |
|--------|------------|
| Mean / P90 duration | Days to project completion |
| Mean / P90 cost | USD total simulated cost |
| Overrun probability | P(cost > 105% plan) |
| Traffic impact | Simulated disruption score |
| CVaR | Expected cost given cost ≥ percentile threshold |

---

## 8. Results

### Key result figures (embedded) {#key-result-figures-embedded}

The plots below are produced by `optimisation_model.py` (figures 1–2) and `tradeoff_study.py` (figures 3–5). **Regenerate** them with both scripts after clone; paths are relative to the repository root.

#### Figure 1 — Monte Carlo distributions and headline KPIs

![Figure 1: Histograms of simulated cost and duration, plus bar comparison of mean KPIs across baseline, optimised, and robust schedules.](Model and solutions/plots/01_distributions.png)

**Caption.** Ten-thousand Monte Carlo draws per schedule variant show tighter cost dispersion for the optimised and robust policies versus the baseline, consistent with the **−8.8%** mean cost and **−26.6%** mean traffic impact in `results_summary.csv`. Duration distributions overlap because the scalar objective prioritises cost and traffic over aggressive makespan reduction.

#### Figure 2 — Baseline vs optimised Gantt-style schedule comparison

![Figure 2: Task placement over time for baseline and optimised block assignments.](Model and solutions/plots/03_gantt.png)

**Caption.** The optimised schedule concentrates high–road-impact work in **off-peak** blocks (B1/B3) where delay multipliers are lower, reducing the traffic proxy while preserving feasibility of precedence and resource caps.

#### Figure 3 — Cost–duration and cost–traffic Pareto-style sweeps

![Figure 3: Deterministic trade-off curves from makespan caps and forced peak-block fractions.](O:P with pareto fronts/plots_tradeoff/06_tradeoff_curves.png)

**Caption.** The panels summarise how **tightening** the makespan cap or **forcing** more peak-block work shifts deterministic labour cost against traffic exposure. The chosen operating point sits on the **low traffic / moderate cost** region once γ weights congestion heavily enough.

#### Figure 4 — Tail risk and cost percentiles

![Figure 4: Tail cost and duration percentiles (P50–P99) and spread across variants.](O:P with pareto fronts/plots_tradeoff/08_tail_risk.png)

**Caption.** Optimised and robust schedules **compress** the upper tail of simulated cost versus baseline (e.g. **−3.1%** at P90 in `tail_risk_summary.csv`), which matters for contingency budgeting more than the mean alone.

#### Figure 5 — Robustness parameter λ vs cost statistics

![Figure 5: Risk–cost curve as the scenario-blend weight λ varies.](O:P with pareto fronts/plots_tradeoff/09_lambda_tradeoff.png)

**Caption.** Sweeping λ traces how much **deterministic** labour premia buy incremental improvements in simulated **P90 / P95** cost. The default robust setting (λ ≈ 0.35) lies in a flat region: modest extra protection without large mean-cost sacrifice.

#### Optional montage — thumbnails of all nine script outputs

<div style="text-align:center;font-size:7pt;">
<table style="width:100%;border:none;"><tr>
<td style="border:none;width:20%;"><img src="Model and solutions/plots/01_distributions.png" width="110"/><br/>01</td>
<td style="border:none;width:20%;"><img src="Model and solutions/plots/02_sensitivity.png" width="110"/><br/>02</td>
<td style="border:none;width:20%;"><img src="Model and solutions/plots/03_gantt.png" width="110"/><br/>03</td>
<td style="border:none;width:20%;"><img src="Model and solutions/plots/04_actionable.png" width="110"/><br/>04</td>
<td style="border:none;width:20%;"><img src="Model and solutions/plots/05_cascade.png" width="110"/><br/>05</td>
</tr><tr>
<td style="border:none;"><img src="O:P with pareto fronts/plots_tradeoff/06_tradeoff_curves.png" width="110"/><br/>06</td>
<td style="border:none;"><img src="O:P with pareto fronts/plots_tradeoff/07_stochastic_study.png" width="110"/><br/>07</td>
<td style="border:none;"><img src="O:P with pareto fronts/plots_tradeoff/08_tail_risk.png" width="110"/><br/>08</td>
<td style="border:none;"><img src="O:P with pareto fronts/plots_tradeoff/09_lambda_tradeoff.png" width="110"/><br/>09</td>
<td style="border:none;"></td>
</tr></table>
<p><em>Montage of outputs 01–09 after a full local run (sensitivity, stochastic deep-dive, and cascade panels included).</em></p>
</div>

### 8.1 Summary table (three variants)

*From `results_summary.csv`.*

| Metric | Baseline | Optimised | Robust |
|--------|----------|-----------|--------|
| Mean duration (d) | 52.38 | 52.39 | 52.38 |
| P90 duration (d) | 53.03 | 53.03 | 53.04 |
| Mean cost (USD) | 1,354,343 | 1,235,768 | 1,233,570 |
| P90 cost (USD) | 1,623,075 | 1,467,441 | 1,462,819 |
| Overrun probability | 0.981 | 0.997 | 0.996 |
| Mean traffic impact | 592.0 | 434.6 | 433.8 |

*Note: High overrun rates across all variants reflect strict 5% threshold against planned cost under broad stochastic shocks; use relative comparison and tail percentiles for ranking.*

### 8.2 Tail risk (cost)

*From `tail_risk_summary.csv`.*

| Percentile | Baseline ($) | Optimised ($) | Δ |
|------------|--------------|---------------|---|
| P50 | 1,237,230 | 1,215,608 | −1.7% |
| P75 | 1,368,612 | 1,337,865 | −2.2% |
| P90 | 1,507,409 | 1,460,035 | −3.1% |
| P95 | 1,617,748 | 1,564,386 | −3.3% |
| P99 | 1,825,345 | 1,796,349 | −1.6% |
| Std dev | 192,106 | 184,373 | −4.0% |

### 8.3 Trade-offs

**Cost vs duration (Pareto):** tightening makespan caps below the free optimum increases deterministic labour cost monotonically; the optimised solution sits on the **low-cost / moderate-duration** region.

**Cost vs traffic:** forcing more peak-block assignments lowers deterministic cost (exploiting shorter calendar windows) but raises traffic index — the **optimised point** favours off-peak blocks at higher γ.

**λ sweep:** robust premium in deterministic cost is small at λ=0.35; P90 cost and duration stabilise relative to the risk-neutral optimum (**Figure 5**).

### 8.4 Actionable task-level changes

`actionable_decisions.csv` records for each task:

- `base_block` → `opt_block`
- Start day shift (`start_shift`)
- DM saving (days equivalent) and `cost_saving_usd`

High-impact tasks (e.g. **T07–T08** milling, **T13** drainage) move from **B2 (peak AM)** to **B1/B3**, yielding large DM savings (2+ days equivalent) and five-figure cost deltas per task.

### 8.5 Managerial recommendations

| ID | Recommendation | Rationale |
|----|----------------|-----------|
| D1 | Move impact≥3 tasks to B3 (midday) where feasible | ≈−26.6% mean traffic score; lower DM |
| D2 | Insert 2-day float before T13 | Absorbs breakdown cascades on T07→T10→T13 chain |
| D3 | Stagger specialist crews weeks 3–5 | Avoids binding 12-unit pool |
| D4 | Monitor remaining peak assignments | Tasks still on B2/B4 need live traffic coordination |

---

## 9. Reproducibility

### 9.1 Environment

```text
Python >= 3.10
numpy, pandas, matplotlib, scipy, pulp
```

### 9.2 Path configuration

Both scripts resolve paths **relative to the repository root** using `pathlib` (`REPO = Path(__file__).resolve().parents[1]`). **`DATA`** points at `Baseline reference datasets/Dependency2/`. **`OUT`** is the script’s own directory; **`PLOTS`** is `OUT / "plots"` (main model) or `OUT / "plots_tradeoff"` (trade-off study). Those plot directories are listed in `.gitignore`; **clone → `pip install -r requirements.txt` → run both scripts** to regenerate CSVs and PNGs locally.

To override (e.g. alternate dataset), edit the `REPO`, `DATA`, `OUT`, or `PLOTS` assignments at the top of:

- `Model and solutions/optimisation_model.py`
- `O:P with pareto fronts/tradeoff_study.py`

### 9.3 Run order

1. `python "Model and solutions/optimisation_model.py"`
2. `python "O:P with pareto fronts/tradeoff_study.py"`

### 9.4 Expected artefacts

See `README.md` § [Outputs and figures](README.md#outputs-and-figures) for CSV and figure filenames. The repository tracks **`sensitivity_alpha.csv`** in `Model and solutions/` when the full main script has been run at least once.

### 9.5 Validation checks

| Check | Expected |
|-------|----------|
| Task count | 24 tasks loaded |
| FS edges | ~30+ feasible arcs |
| MIP status | `Optimal` or feasible within gap |
| MC means | Stable across two consecutive runs with seed 42 |

---

## 10. Discussion and recommendations

### 10.1 Strengths

- **Joint construction–traffic decision** in one solver
- **Transparent task-level outputs** for site implementation
- **Full uncertainty layer** with cascades, not just sensitivity factors
- **Explicit robustness knob** (λ) for risk-averse owners

### 10.2 Limitations

| Limitation | Impact |
|------------|--------|
| Aggregated traffic proxy | Cannot resolve detour queues or spatial spillback |
| Fixed planned durations in MIP | Duration risk only in MC, not in robust constraints |
| Single project scope | No multi-contractor game or adjacent works |
| High overrun metric | Threshold may be pessimistic; interpret ordinally |

### 10.3 Relation to other narrative documents (Word, slides, older HTML)

Some **thesis or executive-summary drafts** (including older styled HTML walkthroughs no longer shipped in this repository) may cite **aspirational KPIs** (e.g. larger percentage cost or delay reductions) from an earlier calibration or storytelling pass.

The **committed CSV outputs** in this repository are treated as the **authoritative numerical results** for analysis and reporting: `Model and solutions/results_summary.csv`, `Model and solutions/actionable_decisions.csv`, and the CSVs under `O:P with pareto fronts/` (summarised in §8). When finalising `*.docx` submissions, align tables and narrative claims to those files and to **Figures 1–5** (and the montage), which are generated from the same code and random seed as the CSVs.

### 10.4 Future work

1. Integrate **OSMnx** network delays into γ term  
2. **SimPy** activity-level simulation with resource contention  
3. **NSGA-II** for true tri-objective Pareto without scalarisation  
4. **Rolling-horizon** re-optimisation with realised progress  
5. **Dashboard** (Plotly/Dash) for λ and γ exploration by stakeholders  

---

## 11. Conclusions

Co-optimising infrastructure task starts and time-of-day blocks under traffic-sensitive delay multipliers produces **measurable system-level gains**: on the latest committed CSVs, roughly **8.8%** lower mean cost, **9.6%** lower P90 cost, and **~27%** lower mean traffic disruption versus baseline, with **negligible change** in mean simulated duration under the current scalar weights. Monte Carlo analysis exposes **cascade-driven tail risk** that deterministic scheduling hides. The robust formulation (λ ≈ 0.35) offers **tail cost protection** without materially sacrificing mean performance.

The project delivers a **reproducible optimisation pipeline**, **decision tables**, and **visual evidence** suitable for systems engineering coursework, smart-infrastructure portfolios, and further research coupling GIS traffic models with field scheduling tools.

---

## Appendix A: File inventory

### A.0 Repository root (documentation and optional prose)

| File | Role |
|------|------|
| `README.md` | Entry point: tree, run instructions, output locations |
| `REPORT.md` | This technical report (Markdown source) |
| `Final_Report.pdf` | PDF export of `REPORT.md` (regenerate with `scripts/build_final_report_pdf.py`) |
| `scripts/build_final_report_pdf.py` | Optional: rebuild `Final_Report.pdf` (`pip install markdown xhtml2pdf`) |
| `requirements.txt` | Runtime dependencies for the two models |
| `LICENSE` | MIT licence |
| `editROBUST OPTIMISATION OF INFRASTRUCTURE SCHEDULING.docx` | Optional editable report / thesis draft (not executed by Python) |

### A.1 Code

| File | Lines (approx.) | Function |
|------|-----------------|----------|
| `Model and solutions/optimisation_model.py` | 720 | Main MIP + MC + plots 01–05 |
| `O:P with pareto fronts/tradeoff_study.py` | 650 | Pareto + tail risk + plots 06–09 |

### A.2 Data (`Baseline reference datasets/`)

| File | Role |
|------|------|
| `Infrastructure_Project_Dataset.xlsx` | Consolidated case-study workbook (reference) |

**`Baseline reference datasets/Dependency2/`** — all files below are read by the Python scripts:

| File | Records (approx.) | Role |
|------|-------------------|------|
| `tasks_clean.csv` | 24 tasks | Durations, resources, road impact, DM per block |
| `dependencies_clean.csv` | 33 edges | FS / SS precedence |
| `cost_parameters_resources.csv` | 9 types | Labour unit rates, overtime, idle fractions |
| `cost_parameters_events.csv` | 10 types | Event penalties and disruption rates |
| `equipment_pool.csv` | 5 types | Shared equipment caps |
| `resource_limits.csv` | 72 rows | Weekly crew caps |
| `resource_usage_daily.csv` | — | Daily resource draw (supporting data) |
| `resources_clean.csv` | — | Resource master list |
| `risk_clean.csv` | 22 events | Risk narrative / attribution for plots |
| `traffic_schedule_multipliers.csv` | 388 rows | Day × block dynamic DM and delay index |
| `daily_schedule_grid.csv` | 97 rows | Fine schedule grid reference |
| `time_blocks.csv` | 4 rows | Block definitions |
| `uncertainty_distributions.json` | — | Monte Carlo factor specification |
| `cost_function_spec.json` | — | Machine-readable cost decomposition |

### A.3 Results CSV (committed in-repo)

| File | Location | Description |
|------|----------|-------------|
| `results_summary.csv` | `Model and solutions/` | Aggregate KPIs × 3 variants |
| `actionable_decisions.csv` | `Model and solutions/` | Per-task diffs |
| `sensitivity_lambda.csv` | `Model and solutions/` | λ sweep |
| `sensitivity_gamma.csv` | `Model and solutions/` | γ sweep |
| `sensitivity_alpha.csv` | `Model and solutions/` | α sweep (written by `optimisation_model.py` when the full pipeline runs) |
| `pareto_cost_duration.csv` | `O:P with pareto fronts/` | Makespan Pareto |
| `pareto_cost_traffic.csv` | `O:P with pareto fronts/` | Traffic Pareto |
| `risk_cost_curve.csv` | `O:P with pareto fronts/` | λ vs cost percentiles |
| `tail_risk_summary.csv` | `O:P with pareto fronts/` | Tail table |

### A.4 Figures

| ID | Filename | Section |
|----|----------|---------|
| 01 | 01_distributions.png | MC distributions |
| 02 | 02_sensitivity.png | Weight sweeps |
| 03 | 03_gantt.png | Schedule comparison |
| 04 | 04_actionable.png | Task deltas |
| 05 | 05_cascade.png | Emergence |
| 06 | 06_tradeoff_curves.png | Pareto triple |
| 07 | 07_stochastic_study.png | Stochastic deep dive |
| 08 | 08_tail_risk.png | Percentiles / CVaR |
| 09 | 09_lambda_tradeoff.png | Robust premium |

Figures **01–05** are written to **`Model and solutions/plots/`** and **06–09** to **`O:P with pareto fronts/plots_tradeoff/`** when you run the scripts. Those directories are **gitignored** in this repository; regenerate them locally after clone. CSV outputs remain next to each script and **are** tracked in git.

**Cross-reference (report vs file IDs):** **Figure 1** = `01_distributions.png`; **Figure 2** = `03_gantt.png`; **Figure 3** = `06_tradeoff_curves.png`; **Figure 4** = `08_tail_risk.png`; **Figure 5** = `09_lambda_tradeoff.png`. **`Final_Report.pdf`** embeds these five plus a thumbnail montage of **01–09** when the PNG folders are present at build time.

---

## Appendix B: Parameter tables

### B.1 Time blocks

| Block | Peak | Hours | DM role |
|-------|------|-------|---------|
| B1 | No | 1 | Early, low congestion |
| B2 | Yes | 2 | AM peak |
| B3 | No | 7 | Midday off-peak |
| B4 | Yes | 2 | PM peak |

### B.2 Robust scenarios

| Scenario | Cost mult. | Traffic mult. | Probability |
|----------|------------|---------------|-------------|
| S1 | 1.00 | 1.00 | 0.50 |
| S2 | 1.18 | 1.35 | 0.25 |
| S3 | 1.35 | 1.60 | 0.15 |
| S4 | 1.55 | 1.80 | 0.10 |

### B.3 Solver settings

| Parameter | Value |
|-----------|-------|
| Time limit | 45–60 s |
| Relative MIP gap | 8% |
| Horizon H | 40 (planned), H_MAX = 55 |

### B.4 Task taxonomy (summary)

| Phase | Task IDs | Examples |
|-------|----------|----------|
| Mobilisation | T01–T02 | Site setup, survey |
| Utilities / traffic | T03–T06 | Gas/fibre, diversions, lane closure |
| Milling | T07–T09 | Segments A/B/C |
| Subbase | T10–T12 | Earthworks |
| Drainage | T13–T15 | Specialist + crane |
| Paving | T16–T21 | Base + asphalt |
| Closeout | T22–T24 | Marking, barriers, QA |

---

*End of report — May 2026*
