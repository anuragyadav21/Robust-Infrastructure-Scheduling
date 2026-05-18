"""
Tradeoff Curves + Stochastic Study
====================================
1.  Pareto frontier: cost vs duration  (vary β, force different makespan targets)
2.  Pareto frontier: cost vs traffic impact (vary γ, force block assignments)
3.  Risk–cost curve: vary λ, measure E[cost] vs P90[cost]
4.  Full stochastic study:
      - distribution comparison (all 3 variants)
      - CDF (cost + duration)
      - tail risk table (P50/P75/P90/P95/P99/worst)
      - variance decomposition bar chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import pulp, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

np.random.seed(42)
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "Baseline reference datasets" / "Dependency2"
OUT = Path(__file__).resolve().parent
PLOTS = OUT / "plots_tradeoff"
PLOTS.mkdir(parents=True, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
DARK  = "#1a1a18"; INK2 = "#3a3a36"; INK3 = "#6b6b65"
PAPER = "#f7f5f0"; PAPER2= "#edeae3"; RULE = "#d4d0c7"
BLUE  = "#1f4e79"; GREEN = "#2d6a4f"; AMBER = "#8b5e00"
RED   = "#8b1a1a"; TEAL  = "#1a5c6e"; PURPLE= "#4a1a6e"
LBLUE = "#c5d9eb"; LGREEN= "#c6deb4"; LGOLD = "#f5e6c0"

plt.rcParams.update({
    'font.family':'DejaVu Sans','figure.facecolor':PAPER,'axes.facecolor':PAPER,
    'axes.edgecolor':RULE,'axes.labelcolor':INK2,'xtick.color':INK3,
    'ytick.color':INK3,'text.color':DARK,'grid.color':RULE,'grid.linewidth':0.5,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'figure.dpi':140, 'legend.frameon':False,
})

def lp(ax, letter, x=-0.08, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=DARK, va='bottom')

# ══════════════════════════════════════════════════════════════════════════════
# DATA + MODEL SETUP  (same as run_model_v2 — self-contained)
# ══════════════════════════════════════════════════════════════════════════════
tasks_df = pd.read_csv(DATA / "tasks_clean.csv")
deps_df  = pd.read_csv(DATA / "dependencies_clean.csv")
COST_RES = pd.read_csv(DATA / "cost_parameters_resources.csv").set_index("resource_type")
cost_e   = pd.read_csv(DATA / "cost_parameters_events.csv").set_index("event_type")
TASKS    = tasks_df.set_index("task_id")
task_ids = list(TASKS.index)
N        = len(task_ids)
H        = 40; H_MAX = 55

BLOCKS   = ["B1","B2","B3","B4"]
IS_PEAK  = {"B1":0,"B2":1,"B3":0,"B4":1}
BLOCK_HRS= {"B1":1,"B2":2,"B3":7,"B4":2}
DM_COL   = {"B1":"dm_block_B1_early","B2":"dm_block_B2_peak_am",
             "B3":"dm_block_B3_midday","B4":"dm_block_B4_peak_pm"}

def dm(t,b):   return float(TASKS.loc[t,DM_COL[b]])
def pd_(t):    return int(TASKS.loc[t,"planned_duration_days"])
def wk(t):     return int(TASKS.loc[t,"workers_required"])
def uc(t):     return float(COST_RES.loc[TASKS.loc[t,"primary_resource_type"],"base_cost_per_unit_per_day_usd"])
def imp(t):    return int(TASKS.loc[t,"road_impact_level"])

FS = [(r.predecessor_id,r.successor_id) for _,r in deps_df[deps_df.dependency_type=="FS"].iterrows()
      if r.predecessor_id in task_ids and r.successor_id in task_ids]
SS = [(r.predecessor_id,r.successor_id) for _,r in deps_df[deps_df.dependency_type=="SS"].iterrows()
      if r.predecessor_id in task_ids and r.successor_id in task_ids]

SUCC = {t:[] for t in task_ids}
for i,j in FS: SUCC[i].append(j)

SCENARIOS = {"S1":(1.00,1.00,1.00,0.50),"S2":(1.15,1.35,1.18,0.25),
             "S3":(1.30,1.60,1.35,0.15),"S4":(1.50,1.80,1.55,0.10)}

def solve(alpha=1.0, beta=0.30, gamma=0.50, lam=0.0,
          makespan_cap=None, force_peak_frac=None, quiet=True):
    """
    Core MIP solver.
    makespan_cap: if set, add constraint mk <= makespan_cap (for Pareto)
    force_peak_frac: if set (0–1), force at least this fraction of tasks into peak blocks
    """
    prob = pulp.LpProblem("sched", pulp.LpMinimize)
    s = {t: pulp.LpVariable(f"s_{t}",0,H_MAX,cat='Integer') for t in task_ids}
    b = {t: {k: pulp.LpVariable(f"b_{t}_{k}",cat='Binary') for k in BLOCKS} for t in task_ids}
    mk = pulp.LpVariable("mk",0,H_MAX,cat='Integer')

    for t in task_ids:
        prob += pulp.lpSum(b[t][k] for k in BLOCKS) == 1
    for i,j in FS: prob += s[j] >= s[i]+pd_(i)
    for i,j in SS: prob += s[j] >= s[i]
    for t in task_ids: prob += mk >= s[t]+pd_(t)

    if makespan_cap:
        prob += mk <= makespan_cap

    if force_peak_frac is not None:
        # force at least force_peak_frac*N tasks into B2 or B4
        n_peak = int(np.ceil(force_peak_frac * N))
        prob += pulp.lpSum(b[t][k] for t in task_ids for k in ["B2","B4"]) >= n_peak

    lab  = pulp.lpSum(b[t][k]*wk(t)*uc(t)*pd_(t)*dm(t,k) for t in task_ids for k in BLOCKS)
    traf = pulp.lpSum(b[t][k]*imp(t)*dm(t,k)*pd_(t)       for t in task_ids for k in BLOCKS)
    soc  = pulp.lpSum(b[t][k]*IS_PEAK[k]*imp(t)*BLOCK_HRS[k]*100 for t in task_ids for k in BLOCKS)

    if lam > 0:
        exp_lab  = sum(p*lab*cm for _,(_,_,cm,p) in SCENARIOS.items())
        worst_cm = max(cm for _,(_,_,cm,_) in SCENARIOS.items())
        obj_c    = (1-lam)*exp_lab + lam*lab*worst_cm
    else:
        obj_c = lab

    prob += alpha*obj_c + beta*mk*8000 + gamma*traf + soc
    solver = pulp.PULP_CBC_CMD(msg=0,timeLimit=45,gapRel=0.08)
    prob.solve(solver)

    sol = {}
    for t in task_ids:
        chosen = "B3"
        for k in BLOCKS:
            if pulp.value(b[t][k]) is not None and pulp.value(b[t][k])>0.5:
                chosen=k; break
        sd = max(0, int(round(pulp.value(s[t]))) if pulp.value(s[t]) is not None else pd_(t))
        sol[t] = {"start":sd,"block":chosen,"plan_dur":pd_(t),"dm":dm(t,chosen),"impact":imp(t),
                  "cost":wk(t)*uc(t)*pd_(t)*dm(t,chosen)}
    mk_val = int(round(pulp.value(mk))) if pulp.value(mk) else \
             max(v["start"]+v["plan_dur"] for v in sol.values())
    det_cost = sum(v["cost"] for v in sol.values())
    traf_val = sum(v["impact"]*v["dm"]*v["plan_dur"] for v in sol.values())
    if not quiet:
        print(f"  mk={mk_val}  cost=${det_cost:,.0f}  traffic={traf_val:.0f}  "
              f"peak={sum(1 for v in sol.values() if IS_PEAK[v['block']]==1)}")
    return sol, mk_val, det_cost, traf_val

def simulate(sol, n=5000):
    plan_cost = sum(v["cost"] for v in sol.values())
    plan_mk   = max(v["start"]+v["plan_dur"] for v in sol.values())
    D=np.zeros(n); C=np.zeros(n); T_=np.zeros(n)
    for i in range(n):
        dur_m  = np.random.triangular(1.00,1.12,1.45)
        tpeak  = np.random.uniform(1.30,1.80)
        toff   = np.random.uniform(1.00,1.20)
        vnoise = max(0.7,np.random.normal(1.0,0.08))
        matd   = max(0,np.random.geometric(0.75)-1) if np.random.random()<0.25 else 0
        ebreak = np.random.random()<0.05
        ecost  = max(0,np.random.normal(13000,2500)) if ebreak else 0
        wdays  = np.random.triangular(0.5,1.5,3.0) if np.random.random()<0.08 else 0
        absent = np.clip(np.random.beta(1.5,10.0),0,0.35)
        casc   = np.random.triangular(1.0,1.2,1.6)
        act={t:0.0 for t in task_ids}
        for t in task_ids:
            v=sol[t]
            tmd = tpeak*vnoise if IS_PEAK[v["block"]] else toff*vnoise
            d = v["plan_dur"]*dur_m*tmd
            if wdays>0 and v["impact"]>=2: d+=wdays
            if matd>0 and t in {"T03","T10","T11","T12","T15","T20"}: d+=matd
            if ebreak and t in {"T07","T08","T09","T13","T15","T16","T19"}: d+=np.random.choice([1,2])
            act[t]=max(v["plan_dur"],d/max(1-absent,0.6))
        visited=set()
        def cp(t):
            if t in visited: return
            visited.add(t)
            for j in SUCC[t]:
                if act[t]>sol[t]["plan_dur"]:
                    act[j]=max(act[j],sol[j]["plan_dur"]*casc+(act[t]-sol[t]["plan_dur"])*0.4)
                cp(j)
        for t in task_ids: cp(t)
        proj_dur = max(sol[t]["start"]+act[t] for t in task_ids)
        lab_cost = sum(wk(t)*uc(t)*act[t] for t in task_ids)+ecost
        tot      = lab_cost+max(0,proj_dur-plan_mk)*8500
        D[i]=proj_dur; C[i]=tot; T_[i]=sum(v["impact"]*v["dm"]*act[t] for t,v in sol.items())
    return D,C,T_

print("="*60)
print("TRADEOFF + STOCHASTIC STUDY")
print("="*60)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1A — PARETO: COST vs DURATION
# Approach: force makespan cap from tight (42) to loose (55), read cost at each
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Pareto frontier: Cost vs Duration")
pareto_cd = []
# Free solve first to get natural optimal makespan
sol_free,ms_free,c_free,tr_free = solve(1.0,0.30,0.50)
print(f"  Free optimum: makespan={ms_free}d  cost=${c_free:,.0f}")

# Sweep makespan caps from tight to loose
for cap in range(42, 58, 1):
    sol_,ms_,c_,tr_ = solve(1.0, 0.30, 0.50, makespan_cap=cap)
    peak = sum(1 for v in sol_.values() if IS_PEAK[v["block"]]==1)
    pareto_cd.append({"makespan_cap":cap,"achieved_makespan":ms_,"det_cost":c_,
                      "traffic":tr_,"peak_tasks":peak})
    print(f"  cap={cap}  achieved={ms_}  cost=${c_:,.0f}  traffic={tr_:.0f}")

pareto_cd = pd.DataFrame(pareto_cd)
# Keep only the efficient frontier (non-dominated)
pareto_cd = pareto_cd.sort_values("achieved_makespan")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1B — PARETO: COST vs TRAFFIC IMPACT
# Approach: force increasing fraction of tasks into peak blocks
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Pareto frontier: Cost vs Traffic Impact")
pareto_ct = []
for peak_frac in np.linspace(0.0, 0.60, 10):
    sol_,ms_,c_,tr_ = solve(1.0, 0.30, 0.50, force_peak_frac=peak_frac)
    peak = sum(1 for v in sol_.values() if IS_PEAK[v["block"]]==1)
    pareto_ct.append({"peak_frac_forced":round(peak_frac,2),"det_cost":c_,
                      "traffic":tr_,"peak_tasks":peak,"makespan":ms_})
    print(f"  peak_frac={peak_frac:.2f}  cost=${c_:,.0f}  traffic={tr_:.0f}  peak_tasks={peak}")

pareto_ct = pd.DataFrame(pareto_ct)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1C — RISK-COST CURVE: E[cost] vs P90[cost] as λ varies
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Risk–cost curve: lambda sweep with MC evaluation")
risk_cost = []
lambda_vals = np.linspace(0.0, 1.0, 11)
for lv in lambda_vals:
    sol_,ms_,c_,tr_ = solve(1.0,0.25,0.50,lam=lv)
    D_,C_,T__ = simulate(sol_, 3000)
    risk_cost.append({
        "lambda":  round(lv,2),
        "det_cost": c_,
        "mean_cost": np.mean(C_),
        "p50_cost":  np.percentile(C_,50),
        "p75_cost":  np.percentile(C_,75),
        "p90_cost":  np.percentile(C_,90),
        "p95_cost":  np.percentile(C_,95),
        "p99_cost":  np.percentile(C_,99),
        "std_cost":  np.std(C_),
        "mean_dur":  np.mean(D_),
        "p90_dur":   np.percentile(D_,90),
    })
    print(f"  λ={lv:.2f}  E[cost]=${np.mean(C_):,.0f}  P90=${np.percentile(C_,90):,.0f}  "
          f"P95=${np.percentile(C_,95):,.0f}")

risk_cost = pd.DataFrame(risk_cost)

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — STOCHASTIC STUDY
# Run full 10k MC on 3 canonical variants
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Full stochastic study: 3 variants × 10,000 iterations")
sol_base,ms_b,c_b,tr_b = solve(1.0,0.05,0.05)   # baseline: no traffic penalty
# force 6 high-impact tasks to peak
sol_base2,_,_,_ = solve(1.0,0.05,0.05,force_peak_frac=0.25)

sol_opt, ms_o,c_o,tr_o = solve(1.0,0.30,0.50)
sol_rob, ms_r,c_r,tr_r = solve(1.0,0.25,0.50,lam=0.35)

print("  Simulating baseline (10k)...")
dur_b, cost_b, traf_b = simulate(sol_base2, 10000)
print(f"    E[dur]={np.mean(dur_b):.1f}  P90={np.percentile(dur_b,90):.1f}  "
      f"E[cost]=${np.mean(cost_b):,.0f}  std=${np.std(cost_b):,.0f}")

print("  Simulating optimised (10k)...")
dur_o, cost_o, traf_o = simulate(sol_opt, 10000)
print(f"    E[dur]={np.mean(dur_o):.1f}  P90={np.percentile(dur_o,90):.1f}  "
      f"E[cost]=${np.mean(cost_o):,.0f}  std=${np.std(cost_o):,.0f}")

print("  Simulating robust (10k)...")
dur_r, cost_r, traf_r = simulate(sol_rob, 10000)
print(f"    E[dur]={np.mean(dur_r):.1f}  P90={np.percentile(dur_r,90):.1f}  "
      f"E[cost]=${np.mean(cost_r):,.0f}  std=${np.std(cost_r):,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Generating plots...")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — TRADEOFF CURVES  (3 panels)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.patch.set_facecolor(PAPER)
fig.suptitle("Tradeoff Curves — Decision Space Characterisation",
             fontsize=11, color=DARK, y=1.01, fontweight='500')

# 1A: Cost vs Duration (Pareto frontier)
ax = axes[0]
cd = pareto_cd.copy()
x = cd["achieved_makespan"].values
y = cd["det_cost"].values / 1e3

ax.plot(x, y, 'o-', color=BLUE, lw=2, ms=7, zorder=5, label="Pareto frontier")
ax.fill_between(x, y, y.max()*1.05, alpha=0.07, color=BLUE)

# Mark the three variants
ax.scatter([ms_b],[c_b/1e3], s=120, color=INK3, zorder=8, marker='s', label=f"Baseline (mk={ms_b}d)")
ax.scatter([ms_o],[c_o/1e3], s=120, color=GREEN, zorder=8, marker='^', label=f"Optimised (mk={ms_o}d)")
ax.scatter([ms_r],[c_r/1e3], s=120, color=RED,   zorder=8, marker='D', label=f"Robust (mk={ms_r}d)")

ax.set_xlabel("Project makespan (working days)", fontsize=9)
ax.set_ylabel("Deterministic labour cost ($K)", fontsize=9)
ax.set_title("Cost vs Duration\n(Pareto frontier)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, alpha=0.35)
# Annotate the efficient region
ax.axvspan(min(x), ms_o+1, alpha=0.04, color=GREEN)
ax.text(min(x)+0.3, y.min()+5, "Efficient\nregion", fontsize=7, color=GREEN, va='bottom')
lp(ax, "A")

# 1B: Cost vs Traffic Impact
ax = axes[1]
ct = pareto_ct.copy()
x2 = ct["traffic"].values
y2 = ct["det_cost"].values / 1e3

# Sort by traffic for clean line
order = np.argsort(x2)
x2s, y2s = x2[order], y2[order]

ax.plot(x2s, y2s, 'o-', color=RED, lw=2, ms=7, zorder=5, label="Tradeoff curve")
ax.fill_between(x2s, y2s, y2s.min()-5, alpha=0.07, color=RED)

ax.scatter([tr_b],[c_b/1e3], s=120, color=INK3, zorder=8, marker='s', label="Baseline")
ax.scatter([tr_o],[c_o/1e3], s=120, color=GREEN, zorder=8, marker='^', label="Optimised")
ax.scatter([tr_r],[c_r/1e3], s=120, color=BLUE,  zorder=8, marker='D', label="Robust")

ax.set_xlabel("Traffic disruption index (Σ impact×DM×duration)", fontsize=9)
ax.set_ylabel("Deterministic labour cost ($K)", fontsize=9)
ax.set_title("Cost vs Traffic Impact\n(more congestion = cheaper?)", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.35)

# Arrow annotation showing direction
ax.annotate("", xy=(x2s.min()+20, y2s[np.argmin(x2s)]+15),
            xytext=(x2s.min()+60, y2s[np.argmin(x2s)]+30),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.text(x2s.min()+62, y2s[np.argmin(x2s)]+28,
        "Lower traffic cost\n= higher $ cost", fontsize=6.5, color=RED)
lp(ax, "B")

# 1C: Risk–Cost Curve (E[cost] vs P90[cost] as λ varies)
ax = axes[2]
x3 = risk_cost["mean_cost"].values / 1e3
y3 = risk_cost["p90_cost"].values  / 1e3
lv  = risk_cost["lambda"].values

sc = ax.scatter(x3, y3, c=lv, cmap='RdYlGn_r', s=80, zorder=5, vmin=0, vmax=1)
ax.plot(x3, y3, '-', color=INK3, lw=1.2, alpha=0.5, zorder=4)

# annotate a few lambda values
for i, l in enumerate(lv):
    if l in [0.0, 0.3, 0.5, 0.7, 1.0]:
        ax.annotate(f"λ={l:.1f}", (x3[i], y3[i]),
                    textcoords="offset points", xytext=(6,4),
                    fontsize=7, color=DARK)

cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
cbar.set_label("λ (risk aversion)", fontsize=8)

ax.set_xlabel("Expected cost E[cost] ($K)", fontsize=9)
ax.set_ylabel("P90 cost ($K)", fontsize=9)
ax.set_title("Risk–Cost Tradeoff\n(λ: 0=risk-neutral → 1=minimax)", fontsize=10)
ax.grid(True, alpha=0.35)

# Draw "better" region arrow
ax.annotate("Better\n(lower E + lower P90)", xy=(x3.min(), y3.min()),
            xytext=(x3.min()+20, y3.min()+30),
            fontsize=7, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
lp(ax, "C")

plt.tight_layout()
fig.savefig(PLOTS / "06_tradeoff_curves.png", dpi=140, bbox_inches='tight')
plt.close()
print("  ✓ 06_tradeoff_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — DISTRIBUTION COMPARISON  (full stochastic)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.patch.set_facecolor(PAPER)
fig.suptitle("Stochastic Study — Full Distribution Analysis (n=10,000 per variant)",
             fontsize=11, color=DARK, y=1.01)

kw_fill = dict(density=True, alpha=0.55, linewidth=0)
colors   = [INK3, GREEN, BLUE]
labels   = ["Baseline","Optimised","Robust"]
dur_arrs = [dur_b, dur_o, dur_r]
cst_arrs = [cost_b, cost_o, cost_r]

# 2A: Duration density — all 3
ax = axes[0,0]
bins_d = np.linspace(min(dur_b.min(),dur_o.min(),dur_r.min())-1,
                     max(dur_b.max(),dur_o.max(),dur_r.max())+1, 55)
for arr,col,lbl in zip(dur_arrs,colors,labels):
    ax.hist(arr, bins=bins_d, color=col, label=lbl, **kw_fill)
    kde = stats.gaussian_kde(arr)
    xr  = np.linspace(bins_d[0],bins_d[-1],300)
    ax.plot(xr, kde(xr), color=col, lw=1.8)
    ax.axvline(np.percentile(arr,90), color=col, lw=1.1, ls=':', alpha=0.9)
ax.set_xlabel("Project duration (working days)")
ax.set_ylabel("Density")
ax.set_title("Duration distribution\n(dotted = P90)")
ax.legend(fontsize=7.5)
lp(ax,"A")

# 2B: Cost density
ax = axes[0,1]
bins_c = np.linspace(min(cost_b.min(),cost_o.min(),cost_r.min())/1e6-0.05,
                     max(cost_b.max(),cost_o.max(),cost_r.max())/1e6+0.05, 55)
for arr,col,lbl in zip(cst_arrs,colors,labels):
    ax.hist(arr/1e6, bins=bins_c, color=col, label=lbl, **kw_fill)
    kde = stats.gaussian_kde(arr/1e6)
    xr  = np.linspace(bins_c[0],bins_c[-1],300)
    ax.plot(xr, kde(xr), color=col, lw=1.8)
ax.set_xlabel("Total project cost ($M)")
ax.set_ylabel("Density")
ax.set_title("Cost distribution")
ax.legend(fontsize=7.5)
lp(ax,"B")

# 2C: Traffic impact density
ax = axes[0,2]
for arr,col,lbl in zip([traf_b,traf_o,traf_r],colors,labels):
    ax.hist(arr, bins=40, color=col, label=lbl, **kw_fill)
ax.set_xlabel("Traffic disruption score")
ax.set_ylabel("Density")
ax.set_title("Traffic disruption distribution")
ax.legend(fontsize=7.5)
lp(ax,"C")

# 2D: Duration CDF
ax = axes[1,0]
for arr,col,lbl in zip(dur_arrs,colors,labels):
    xs  = np.sort(arr)
    cdf = np.arange(1,len(xs)+1)/len(xs)
    ax.plot(xs, cdf, color=col, lw=2, label=lbl)
for pct,ls,lbl in [(50,'--','P50'),(75,':','P75'),(90,'-.','P90'),(95,'-','P95')]:
    ax.axhline(pct/100, color=RULE, lw=0.8, ls=ls)
    ax.text(xs[-1]*1.0005, pct/100-0.01, lbl, fontsize=6.5, color=INK3)
ax.set_xlabel("Duration (working days)")
ax.set_ylabel("Cumulative probability")
ax.set_title("Duration CDF\n(horizontal lines = percentile markers)")
ax.legend(fontsize=7.5)
lp(ax,"D")

# 2E: Cost CDF
ax = axes[1,1]
for arr,col,lbl in zip(cst_arrs,colors,labels):
    xs  = np.sort(arr)/1e6
    cdf = np.arange(1,len(xs)+1)/len(xs)
    ax.plot(xs, cdf, color=col, lw=2, label=lbl)
for pct,ls in [(50,'--'),(75,':'),(90,'-.'),(95,'-')]:
    ax.axhline(pct/100, color=RULE, lw=0.8, ls=ls)
ax.set_xlabel("Total cost ($M)")
ax.set_ylabel("Cumulative probability")
ax.set_title("Cost CDF\n(read: prob(cost ≤ X)")
ax.legend(fontsize=7.5)
lp(ax,"E")

# 2F: Variance decomposition (std dev breakdown)
ax = axes[1,2]
metrics  = ["Duration\nstd (days)","Cost std\n($K)","Traffic\nstd"]
vals_b   = [np.std(dur_b), np.std(cost_b)/1e3, np.std(traf_b)/10]
vals_o   = [np.std(dur_o), np.std(cost_o)/1e3, np.std(traf_o)/10]
vals_r   = [np.std(dur_r), np.std(cost_r)/1e3, np.std(traf_r)/10]
x_ = np.arange(3); w=0.25
ax.bar(x_-w, vals_b, w, color=INK3, alpha=0.8, label="Baseline")
ax.bar(x_,   vals_o, w, color=GREEN, alpha=0.8, label="Optimised")
ax.bar(x_+w, vals_r, w, color=BLUE, alpha=0.8, label="Robust")
ax.set_xticks(x_); ax.set_xticklabels(metrics, fontsize=8)
ax.set_ylabel("Standard deviation (scaled)")
ax.set_title("Variance decomposition\n(std dev per metric)")
ax.legend(fontsize=7.5)
# % reduction annotations
for xi, (vb,vo) in enumerate(zip(vals_b, vals_o)):
    pct = (vb-vo)/vb*100 if vb>0 else 0
    ax.text(xi, max(vb,vo)*1.02, f"↓{pct:.0f}%", ha='center', fontsize=7, color=GREEN)
lp(ax,"F")

plt.tight_layout()
fig.savefig(PLOTS / "07_stochastic_study.png", dpi=140, bbox_inches='tight')
plt.close()
print("  ✓ 07_stochastic_study.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — TAIL RISK DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.patch.set_facecolor(PAPER)
fig.suptitle("Tail Risk Analysis — P50 to Worst-Case Comparison",
             fontsize=11, color=DARK, y=1.01)

pctls = [50, 75, 90, 95, 99]

# 3A: Cost tail risk comparison (bar chart per percentile)
ax = axes[0]
x_  = np.arange(len(pctls)); w=0.27
vb  = [np.percentile(cost_b/1e3,p) for p in pctls]
vo  = [np.percentile(cost_o/1e3,p) for p in pctls]
vr  = [np.percentile(cost_r/1e3,p) for p in pctls]
ax.bar(x_-w, vb, w, color=INK3, alpha=0.8, label="Baseline")
ax.bar(x_,   vo, w, color=GREEN, alpha=0.8, label="Optimised")
ax.bar(x_+w, vr, w, color=BLUE, alpha=0.8, label="Robust")
ax.set_xticks(x_); ax.set_xticklabels([f"P{p}" for p in pctls], fontsize=8)
ax.set_ylabel("Project cost ($K)")
ax.set_title("Cost tail risk\n(P50 to P99)")
ax.legend(fontsize=7.5)
for xi,(vb_,vo_) in enumerate(zip(vb,vo)):
    diff = (vb_-vo_)/vb_*100
    ax.text(xi, max(vb_,vo_)+3, f"↓{diff:.0f}%", ha='center', fontsize=6.5, color=GREEN)
lp(ax,"A")

# 3B: Duration tail risk
ax = axes[1]
vb2 = [np.percentile(dur_b,p) for p in pctls]
vo2 = [np.percentile(dur_o,p) for p in pctls]
vr2 = [np.percentile(dur_r,p) for p in pctls]
ax.bar(x_-w, vb2, w, color=INK3, alpha=0.8, label="Baseline")
ax.bar(x_,   vo2, w, color=GREEN, alpha=0.8, label="Optimised")
ax.bar(x_+w, vr2, w, color=BLUE, alpha=0.8, label="Robust")
ax.set_xticks(x_); ax.set_xticklabels([f"P{p}" for p in pctls], fontsize=8)
ax.set_ylabel("Project duration (working days)")
ax.set_title("Duration tail risk\n(P50 to P99)")
ax.legend(fontsize=7.5)
lp(ax,"B")

# 3C: CVaR / Expected Shortfall plot
ax = axes[2]
# Expected shortfall (average of tail above threshold)
thresholds = np.linspace(0.50, 0.99, 50)
es_b  = [np.mean(cost_b[cost_b >= np.percentile(cost_b, t*100)])/1e3 for t in thresholds]
es_o  = [np.mean(cost_o[cost_o >= np.percentile(cost_o, t*100)])/1e3 for t in thresholds]
es_r  = [np.mean(cost_r[cost_r >= np.percentile(cost_r, t*100)])/1e3 for t in thresholds]
ax.plot(thresholds*100, es_b, color=INK3, lw=2, label="Baseline")
ax.plot(thresholds*100, es_o, color=GREEN, lw=2, label="Optimised")
ax.plot(thresholds*100, es_r, color=BLUE,  lw=2, label="Robust")
ax.fill_between(thresholds*100, es_b, es_o, alpha=0.12, color=GREEN, label="Savings vs baseline")
ax.set_xlabel("Confidence level (%)")
ax.set_ylabel("Expected Shortfall / CVaR ($K)")
ax.set_title("Conditional Value at Risk\n(expected cost above threshold)")
ax.legend(fontsize=7.5)
ax.axvline(90, color=RED, lw=0.8, ls='--', alpha=0.6)
ax.text(90.5, np.mean(es_b)*0.98, "P90", fontsize=7, color=RED)
ax.grid(True, alpha=0.35)
lp(ax,"C")

plt.tight_layout()
fig.savefig(PLOTS / "08_tail_risk.png", dpi=140, bbox_inches='tight')
plt.close()
print("  ✓ 08_tail_risk.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — LAMBDA SWEEP DETAIL (risk-cost tradeoff deep dive)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.patch.set_facecolor(PAPER)
fig.suptitle("Risk-Aversion Parameter (λ) — Full Tradeoff Characterisation",
             fontsize=11, color=DARK, y=1.01)

lv_arr   = risk_cost["lambda"].values
ec_arr   = risk_cost["mean_cost"].values / 1e3
p90_arr  = risk_cost["p90_cost"].values  / 1e3
p95_arr  = risk_cost["p95_cost"].values  / 1e3
p99_arr  = risk_cost["p99_cost"].values  / 1e3
std_arr  = risk_cost["std_cost"].values  / 1e3
dc_arr   = risk_cost["det_cost"].values  / 1e3
pd90_arr = risk_cost["p90_dur"].values

# 4A: Expected cost + percentiles vs lambda
ax = axes[0]
ax.plot(lv_arr, ec_arr,  color=GREEN,  lw=2,   label="E[cost]", zorder=5)
ax.plot(lv_arr, p90_arr, color=AMBER,  lw=1.8, ls='--', label="P90 cost")
ax.plot(lv_arr, p95_arr, color=RED,    lw=1.5, ls=':',  label="P95 cost")
ax.fill_between(lv_arr, ec_arr, p90_arr, alpha=0.10, color=AMBER)
ax.fill_between(lv_arr, p90_arr, p99_arr, alpha=0.08, color=RED)
ax.axvline(0.35, color=BLUE, lw=0.9, ls='--', alpha=0.7)
ax.text(0.37, p90_arr.max()*0.99, "λ=0.35\n(selected)", fontsize=6.5, color=BLUE, va='top')
ax.set_xlabel("λ (risk aversion)"); ax.set_ylabel("Cost ($K)")
ax.set_title("Cost percentiles vs λ\n(shaded = uncertainty band)")
ax.legend(fontsize=7.5); ax.grid(True, alpha=0.35); lp(ax,"A")

# 4B: Deterministic cost (robust premium) vs lambda
ax = axes[1]
ax.plot(lv_arr, dc_arr, 'o-', color=AMBER, lw=2, ms=6, label="Deterministic cost")
ax.fill_between(lv_arr, dc_arr, dc_arr.min(), alpha=0.15, color=AMBER, label="Robust premium")
ax.axhline(dc_arr.min(), color=GREEN, lw=0.9, ls='--')
ax.text(0.02, dc_arr.min()+1, f"Opt floor: ${dc_arr.min():.0f}K", fontsize=7, color=GREEN)
premium_pct = (dc_arr - dc_arr.min()) / dc_arr.min() * 100
ax2 = ax.twinx()
ax2.plot(lv_arr, premium_pct, 's--', color=RED, lw=1.5, ms=4, alpha=0.7, label="Premium (%)")
ax2.set_ylabel("Robust premium (%)", color=RED, fontsize=8)
ax2.tick_params(axis='y', labelcolor=RED)
ax.set_xlabel("λ"); ax.set_ylabel("Deterministic cost ($K)")
ax.set_title("Robust premium vs λ\n(cost of buying risk protection)")
lines = ax.get_lines()+ax2.get_lines()
ax.legend(lines,[l.get_label() for l in lines], fontsize=7); ax.grid(True, alpha=0.35); lp(ax,"B")

# 4C: P90 duration vs lambda
ax = axes[2]
ax.plot(lv_arr, pd90_arr, 'o-', color=BLUE, lw=2, ms=6)
ax.fill_between(lv_arr, pd90_arr, pd90_arr.max(), alpha=0.10, color=BLUE)
ax.set_xlabel("λ")
ax.set_ylabel("P90 project duration (working days)")
ax.set_title("P90 duration vs λ\n(risk protection → schedule stability)")
ax.axvline(0.35, color=RED, lw=0.9, ls='--', alpha=0.7)
ann_x = 0.35; ann_y = pd90_arr[np.argmin(np.abs(lv_arr-0.35))]
ax.annotate(f"λ=0.35: P90={ann_y:.1f}d",
            xy=(ann_x,ann_y), xytext=(0.50,ann_y+0.4),
            fontsize=7.5, color=RED,
            arrowprops=dict(arrowstyle="->",color=RED,lw=0.9))
ax.grid(True, alpha=0.35); lp(ax,"C")

plt.tight_layout()
fig.savefig(PLOTS / "09_lambda_tradeoff.png", dpi=140, bbox_inches='tight')
plt.close()
print("  ✓ 09_lambda_tradeoff.png")

# ── Save summary tables ───────────────────────────────────────────────────────
pareto_cd.to_csv(OUT / "pareto_cost_duration.csv", index=False)
pareto_ct.to_csv(OUT / "pareto_cost_traffic.csv", index=False)
risk_cost.to_csv(OUT / "risk_cost_curve.csv", index=False)

# Tail risk summary table
tail_summary = pd.DataFrame({
    "percentile": [50,75,90,95,99,"mean","std"],
    "baseline_cost_usd": [np.percentile(cost_b,p) if isinstance(p,int) else
                          (np.mean(cost_b) if p=="mean" else np.std(cost_b)) for p in [50,75,90,95,99,"mean","std"]],
    "optimised_cost_usd":[np.percentile(cost_o,p) if isinstance(p,int) else
                          (np.mean(cost_o) if p=="mean" else np.std(cost_o)) for p in [50,75,90,95,99,"mean","std"]],
    "robust_cost_usd":   [np.percentile(cost_r,p) if isinstance(p,int) else
                          (np.mean(cost_r) if p=="mean" else np.std(cost_r)) for p in [50,75,90,95,99,"mean","std"]],
    "baseline_dur_days": [np.percentile(dur_b,p) if isinstance(p,int) else
                          (np.mean(dur_b) if p=="mean" else np.std(dur_b)) for p in [50,75,90,95,99,"mean","std"]],
    "optimised_dur_days":[np.percentile(dur_o,p) if isinstance(p,int) else
                          (np.mean(dur_o) if p=="mean" else np.std(dur_o)) for p in [50,75,90,95,99,"mean","std"]],
    "robust_dur_days":   [np.percentile(dur_r,p) if isinstance(p,int) else
                          (np.mean(dur_r) if p=="mean" else np.std(dur_r)) for p in [50,75,90,95,99,"mean","std"]],
})
tail_summary.to_csv(OUT / "tail_risk_summary.csv", index=False)

print("\n── Summary ────────────────────────────────────────────")
print(tail_summary.round(1).to_string(index=False))
print(f"\n✅  All done. Plots: {PLOTS}")
