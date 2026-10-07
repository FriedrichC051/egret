# EGRET: Effect-Grounded Recomputation for Embodied Tasks

*Rethinking Skills for Embodied AI: A Verifiable Skill Framework and Benchmark for LLM-Synthesized Libraries.*

This repository contains the code, skill pool, benchmark, and precomputed artifacts to
reproduce all quantitative results of the paper.

The project has three components:

1. **A verifiable skill representation** — skills are symbolic cards with typed parameters,
   preconditions, and effects; a composite card nests an *ordered* sub-skill sequence that
   expands down to atomic skills, so what a skill truly achieves can be **recomputed from its
   structure** instead of read off its declaration.
2. **A controlled diagnostic benchmark on ALFRED / AI2-THOR** — 211 evaluation instructions
   over a pool of **511 skills** (13 atomic + 125 gold composite + 373 structurally perturbed
   candidates), with effect-grounded ground truth and success adjudicated by real execution.
3. **EGRET**, a lightweight neuro-symbolic router that selects by recomputed effects, with two
   small learnable heads (~28K parameters total) and **no LLM API calls at inference**.

Headline numbers (mean over three runs, 211 queries per run): EGRET reaches **84.7% SR /
93.6% GC**, 3.0pt below the **87.7%** ceiling measured by executing the gold skills directly.

---

## 1. Repository structure

```
egret/
├── README.md                     # this file
├── requirements.txt              # routing / offline evaluation / training
├── requirements-sim.txt          # AI2-THOR simulation (WSL/Linux)
└── embodied_skill/
    ├── skills/                   # skill framework + benchmark pool construction
    │   ├── code/
    │   │   ├── skill_effects.py         # effect-signature algebra (goal_signature, combine_sequential)
    │   │   ├── atomic_bodies.py         # 13 atomic skills → AI2-THOR action sequences
    │   │   ├── composite_bodies.py      # composite skill execution bodies
    │   │   ├── synth_composite.py       # LLM synthesis of composite skills & distractors
    │   │   ├── validate_skills.py       # validate + merge → pool_clean.json
    │   │   └── build_tiers.py           # distractor-density tiers (low/mid/high)
    │   └── datasets/             # the released skill pool (see §3)
    ├── data/                     # ALFRED task distillation
    │   ├── code/load_tasks.py           # ALFRED trajectories → task instructions
    │   └── scripts/fetch_alfred_lite.sh # download the ALFRED-lite archive
    ├── gt/                       # effect-grounded ground truth
    │   ├── code/derive_gt.py            # tasks → required_effects + gold skill sets
    │   └── datasets/*.jsonl             # gt_kitchen{,_train,_eval}.jsonl
    ├── router/                   # the router + all baselines
    │   └── code/
    │       ├── common.py                # pool loading, effect recomputation, goal parsing
    │       ├── retrievers.py            # BM25 + dense BGE retrieval
    │       ├── pipelines.py             # SkillFlow: dense → rerank → LLM
    │       ├── llm_baselines.py         # LLM-Full; shared OpenAI-compatible LLM call
    │       ├── lota_b4.py               # LoTa-Bench: local LM w/ constrained decoding
    │       ├── domain_m1.py             # EGRET router (coverage selection over recomputed effects)
    │       ├── goal_parser_llm.py       # LLM few-shot goal parser (ablation)
    │       ├── l0_head.py               # trainable Goal Parser head
    │       └── l4_head.py               # trainable tie-breaking preference scorer
    ├── eval/                     # evaluation drivers + precomputed outputs
    │   ├── code/
    │   │   ├── run_pilot.py             # main driver: routing → offline metrics → selections.json
    │   │   ├── metrics.py               # Sel-Hit / Sel-FullCov (effect-grounded)
    │   │   └── run_confusion_curve.py   # density-tier sweep (supplementary tooling, see §3)
    │   └── datasets/             # precomputed outputs + LLM caches (see §3)
    └── exec/                     # real-execution simulation (AI2-THOR)
        ├── code/
        │   ├── run_sim_from_selections.py # replay selected skills → SR / GC
        │   └── run_calibration.py         # success checker + gold/naive/empty calibration
        └── scripts/run_sim_methods.sh
```

The code resolves all paths relative to `embodied_skill/`, so the repository can live in any
directory. Run commands from the `embodied_skill/` folder unless noted otherwise.

The two learnable weights (`l0_head.pt`, `l4_head.pt`) are **not** shipped; regenerate them with
the training scripts in §5.2 (a few minutes on a single GPU). They are saved under
`router/models/`, which is created on demand.

---

## 2. Installation

The pipeline has two sides that can be set up independently:

- **Routing / offline evaluation / training** — pure Python, runs on Windows or
  Linux, CPU or a single small GPU.
- **Simulation (SR / GC)** — AI2-THOR, which needs a Linux environment with a GPU and an
  OpenGL display. We ran it under WSL (Windows Subsystem for Linux) with WSLg GPU rendering.
  The `ai2thor` package downloads its scene build automatically on first use.

```bash
# Side A: routing / evaluation (Python 3.10 recommended)
python -m venv .venv && source .venv/bin/activate      # or conda create -n egret python=3.10
pip install -r requirements.txt

# Side B: simulation (in the Linux/WSL environment)
pip install -r requirements-sim.txt
```

**API key (only for LLM-based methods B0, B3, and the LLM goal-parser ablation).**
These call an OpenAI-compatible chat endpoint. Provide a key via environment variable:

```bash
export OPENAI_API_KEY=<your-key>       # or MOONSHOT_API_KEY
```

The default endpoint/model are set in `router/code/llm_baselines.py`
(`KIMI_BASE`, `kimi-k2.7-code-highspeed`) and can be overridden there. LLM calls are cached to
disk (`eval/datasets/*_cache.json`), so re-runs on the shipped queries incur **no API cost**.
The non-LLM methods (BM25, Dense, LoTa-Bench, and the rule/head variants of EGRET) require
**no API key**.

---

## 3. Data included in this repository

**Skill pool** (`embodied_skill/skills/datasets/`):

| File | Contents |
|---|---|
| `atomic_skills.json` | 13 atomic skills (typed params, effects, AI2-THOR bodies) |
| `composite_skills_seed.json` | 5 hand-authored gold composite skills |
| `composite_skills_generated.json` | LLM-synthesized composite skills |
| `confusing_variants_generated.json` | structurally perturbed candidates: reorder / missing / lookalike (with `variant_of`, `why_wrong`) |
| `pool_clean.json` | validated pool = 125 gold composite + 373 perturbed candidates |
| `pool_tier_{low,mid,high}.json`, `tiers_manifest.json` | distractor-density tiers (×1 / ×3 / ×6 over a shared 34-task base) |

Loading `atomic_skills.json` + `pool_clean.json` yields the **511-skill routing pool**.

**Tasks & ground truth** (`embodied_skill/data/`, `embodied_skill/gt/`):

| File | Contents |
|---|---|
| `gt/datasets/gt_kitchen_eval.jsonl` | **211 evaluation queries** (ALFRED valid seen+unseen) with `required_effects` and gold skill sets |
| `gt/datasets/gt_kitchen_train.jsonl` | 2607 training queries (ALFRED train split) for the two heads |
| `gt/datasets/gt_kitchen.jsonl` | full 2818-query set |

Ground truth is **effect-grounded**: correctness is judged by whether a selected skill set's
*recomputed* effect signatures cover `required_effects`, not by skill-id match. This is the
single criterion shared by every method (see `router/code/common.py`).

> The intermediate `tasks_kitchen*.jsonl` files are **not** shipped (they embed absolute
> download paths). Regenerate them from raw ALFRED with `data/code/load_tasks.py` if needed;
> the shipped ground-truth files are sufficient for the full routing/evaluation pipeline.

**Precomputed artifacts** (`embodied_skill/eval/datasets/`) let you inspect paper numbers
without re-running anything:

| File | Backs |
|---|---|
| `selections.json` | per-query routing outputs of every method, including the information-matched `-Card` baselines |
| `sim_fix2_r{1,2,3}.json` | main table, three runs (SR / GC per query per method) |
| `sim_arch_fix_r{1,2,3}.json` + `selections_arch.json` | leave-one-out ablation, three runs |
| `sim_lota_r{1,2,3}.json` | LoTa-Bench, three runs |
| `sim_gen_ax{2,3}_r{1,2,3}.json` + `selections_gen_ax{2,3}.json` | the two held-out generalization settings |
| `confusion_llm_cache.json`, `goal_llm_cache.json` | LLM response caches (re-runs cost nothing) |

> `sim_results.json` and `confusion_curve.json` are earlier runs kept for history. They predate
> the two adjudication safeguards described in §5.4 and are **not** the source of any number in
> the paper; use the `sim_*_r{1,2,3}.json` files above.

---

## 4. Quick start (offline, no GPU / no API / no ALFRED download)

Reproduce the offline retrieval and effect-coverage metrics for the training-free methods
straight from the shipped pool and ground truth:

```bash
cd embodied_skill
python eval/code/run_pilot.py --n -1 --seed 0 \
    --gt gt_kitchen_eval.jsonl \
    --methods bm25 dense m1
```

`m1` is the rule-based variant of the router (no learned weights, no API). This prints
Sel-Hit / Sel-FullCov and writes/updates `eval/datasets/selections.json`.

---

## 5. Full reproduction

### 5.0 (Optional) Regenerate tasks and ground truth from ALFRED

```bash
# in the Linux/WSL environment
bash embodied_skill/data/scripts/fetch_alfred_lite.sh          # downloads ALFRED-lite to ~/alfred_data
python embodied_skill/data/code/load_tasks.py                  # → data/datasets/tasks_kitchen*.jsonl
python embodied_skill/gt/code/derive_gt.py --tasks tasks_kitchen_eval.jsonl  --out gt_kitchen_eval.jsonl
python embodied_skill/gt/code/derive_gt.py --tasks tasks_kitchen_train.jsonl --out gt_kitchen_train.jsonl
```

### 5.1 (Optional) Regenerate the skill pool

```bash
cd embodied_skill
# synthesize composite skills and perturbed candidates (needs OPENAI_API_KEY)
python skills/code/synth_composite.py --mode composite --n 6 --run
python skills/code/synth_composite.py --mode variant --base-limit 8 --run
# validate + merge into the routing pool
python skills/code/validate_skills.py --extra composite_skills_generated.json confusing_variants_generated.json
# build the density tiers
python skills/code/build_tiers.py --tiers 1 3 6 --seed 0
```

### 5.2 Train the two EGRET heads

```bash
cd embodied_skill
python router/code/l0_head.py                # → router/models/l0_head.pt
python router/code/l4_head.py --l0 head      # → router/models/l4_head.pt
```

### 5.3 Routing + offline metrics for all methods

```bash
cd embodied_skill
python eval/code/run_pilot.py --n -1 --seed 0 --gt gt_kitchen_eval.jsonl \
    --methods bm25c densec b0c b3c b4 m1 m1h m1h4
# b4 (LoTa-Bench) uses a local model; the paper configuration:
#   export B4_MODEL=Qwen/Qwen2.5-3B
```

Method flags: `m1` (rule goal parser + coverage selection), `m1h` (trained goal-parser head),
`m1h4` (the full router = head + tie-breaking scorer + coverage selection), `b4` (LoTa-Bench),
`gold` (oracle ceiling).

**Information matching.** Every baseline reported in the paper reads the *same per-skill
fields* as the router — name, description, symbolic preconditions, and the atomic-expanded
sub-skill sequence. Those are the `-Card` variants: `bm25c`, `densec`, `b0c`, `b3c`
(`BM25-Card`, `Dense-Card`, `LLM-Full-Card`, `SkillFlow-Card`). The plain `bm25`, `dense`,
`b0`, `b3` flags keep the earlier name-and-description-only inputs; they are retained for
reference and are **not** what the paper reports. The one exception is LoTa-Bench, whose
constrained-decoding interface scores skill *names* as continuations and therefore cannot
consume card structure.

`run_pilot.py` flushes `selections.json` after every query for the API-backed methods, so an
interrupted run can be resumed with `--only-missing` instead of restarting.

### 5.4 Real-execution simulation (SR / GC) in AI2-THOR

```bash
# in the Linux/WSL environment with a GPU display (ai2thor downloads its build on first run)
cd embodied_skill
DISPLAY=:0 GALLIUM_DRIVER=d3d12 python exec/code/run_sim_from_selections.py \
    --selections eval/datasets/selections.json --out eval/datasets/sim_run1.json
# or replay one method at a time:
bash exec/scripts/run_sim_methods.sh "M1-Head+L4" B0-LLM-Full-Card B3-SkillFlow-Card
```

`DISPLAY` / `GALLIUM_DRIVER=d3d12` are the WSLg GPU-OpenGL settings we used; adjust or drop
them for a native Linux display.

**Two safeguards are required for adjudication to be stable**, and we recommend both to anyone
reusing this harness:

- *Containment re-check.* AI2-THOR reports success for a forced placement that in fact lands
  the object beside an appliance rather than inside it; the interpreter therefore re-checks
  containment through `receptacleObjectIds` after each placement
  (`skills/code/atomic_bodies.py:_ensure_contained`). Without it an appliance is toggled on an
  empty cavity and the effect is lost silently.
- *Periodic controller rebuild.* A run's success rate decays with position in a long sequence,
  independent of load. `--restart-every N` (default 25) rebuilds the controller every N
  executed queries.

Results are flushed after every query through a temp file plus `os.replace`, so a long run can
be resumed with `--resume` and an interruption never truncates the output. We report the mean
over three runs with 95% confidence intervals; round-to-round spread for a fixed configuration
is at most 2.9pt.

### 5.5 Density-tier sweep (supplementary)

`eval/code/run_confusion_curve.py` sweeps the `pool_tier_{low,mid,high}` pools. It is retained
as tooling, but **no number in the paper comes from it**: once every baseline reads the same
per-skill fields, success on this axis is flat, and an audit of the perturbed candidates
explains why — a substantial share of them reach a terminal state equivalent to their gold base
while keeping a legal action chain, so adding more of them does not make a query harder.

---

## 6. Hardware and models

- **Hardware.** All experiments run on a single consumer GPU with **8 GB VRAM**; no multi-GPU,
  no reinforcement learning, and no models larger than 3B are required.
- **Encoders / models.** Dense retrieval and the Goal Parser head use `bge-small-en-v1.5`;
  SkillFlow reranking uses `bge-reranker-base`; LoTa-Bench uses `Qwen2.5-3B`; LLM-Full and
  SkillFlow selection use an OpenAI-compatible chat model. Hugging Face models download on
  first use.
- **Trainable parameters.** EGRET learns only two tiny heads on top of a frozen encoder
  (~28K parameters total): a multi-label Goal Parser head and a tie-breaking preference
  scorer. Everything else — legality filtering, effect recomputation, coverage selection, and
  coverage repair — is symbolic, and inference makes no LLM API calls.
