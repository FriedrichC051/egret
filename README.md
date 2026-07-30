# ESCOVER: Effect-Grounded Composite-Skill Routing for Embodied Agents

*Rethinking Skills for Embodied AI: An Effect-Grounded Composite-Skill Framework and Benchmark.*

This repository contains the code, skill pool, benchmark, and precomputed artifacts to
reproduce all quantitative results of the paper. It is released for **anonymous peer review**
and contains no author- or institution-identifying information.

The project has three components:

1. **A composite-skill framework** — skills are symbolic cards with typed parameters,
   preconditions, and *effect signatures*; composite skills nest ordered sub-skills and are
   what make skill *selection* a combinatorial problem rather than a lookup.
2. **A routing benchmark on ALFRED / AI2-THOR** — 211 evaluation instructions over a pool of
   **511 skills** (13 atomic + 125 gold composite + 373 adversarial distractors), with
   effect-grounded ground truth and a confusion-density protocol.
3. **ESCOVER**, a neuro-symbolic router that grounds selection in *re-computed* skill effects
   and solves a set-cover objective, with two small (~28K-parameter) learnable heads.

---

## 1. Repository structure

```
escover/
├── README.md                     # this file
├── requirements.txt              # routing / offline eval / training / plotting
├── requirements-sim.txt          # AI2-THOR simulation (WSL/Linux)
└── embodied_skill/
    ├── skills/                   # skill framework + benchmark pool construction
    │   ├── code/                 # skill semantics & synthesis
    │   │   ├── skill_effects.py         # effect-signature algebra (goal_signature, combine_sequential)
    │   │   ├── atomic_bodies.py         # 13 atomic skills → AI2-THOR action sequences
    │   │   ├── composite_bodies.py      # composite skill execution bodies
    │   │   ├── synth_composite.py       # LLM synthesis of composite skills & distractors
    │   │   ├── validate_skills.py       # validate + merge → pool_clean.json
    │   │   ├── build_tiers.py           # confusion-density tiers (low/mid/high)
    │   │   └── smoke_*.py, api_check.py # connectivity / execution smoke tests
    │   ├── datasets/             # the released skill pool (see §3)
    │   └── scripts/              # batch synthesis helper
    ├── data/                     # ALFRED task distillation
    │   ├── code/load_tasks.py           # ALFRED trajectories → task instructions
    │   └── scripts/fetch_alfred_lite.sh # download the ALFRED-lite archive
    ├── gt/                       # effect-grounded ground truth
    │   ├── code/derive_gt.py            # tasks → required_effects + gold skill sets
    │   └── datasets/*.jsonl             # gt_kitchen{,_train,_eval}.jsonl
    ├── router/                   # ESCOVER + all baselines
    │   ├── code/
    │   │   ├── common.py                # pool loading, effect re-computation, goal parsing
    │   │   ├── retrievers.py            # BM25 (B1) + dense BGE (B2)
    │   │   ├── pipelines.py             # SkillFlow (B3): dense → rerank → LLM
    │   │   ├── llm_baselines.py         # LLM-Full (B0); shared OpenAI-compatible LLM call
    │   │   ├── lota_b4.py               # LoTa-Bench (B4): local LM w/ constrained decoding
    │   │   ├── domain_m1.py             # ESCOVER router (set-cover over re-computed effects)
    │   │   ├── goal_parser_llm.py       # LLM few-shot goal parser (ablation)
    │   │   ├── l0_head.py               # trainable Goal Parser head (L0)
    │   │   └── l4_head.py               # trainable effect-grounded tie-break head (L3)
    │   └── models/               # trained weights land here (see models/README.md)
    ├── eval/                     # evaluation, curves, ablations
    │   ├── code/
    │   │   ├── run_pilot.py             # main driver: routing → offline metrics → selections.json
    │   │   ├── metrics.py               # Sel-Hit / Sel-FullCov (effect-grounded)
    │   │   ├── run_confusion_curve.py   # confusion-density curve
    │   │   ├── plot_confusion_curve.py  # curve figure
    │   │   ├── _confusion_ablation.py   # per-layer anti-confusion attribution
    │   │   ├── _arch_ablation.py        # leave-one-component-out ablation
    │   │   └── wp8_breakdown.py         # per-subset (single/composite, seen/unseen) breakdown
    │   └── datasets/             # precomputed outputs, LLM caches, figures (see §3)
    ├── exec/                     # real-execution simulation (AI2-THOR)
    │   ├── code/
    │   │   ├── run_sim_from_selections.py # replay selected skills → SR / GC
    │   │   └── run_calibration.py         # success checker + gold/naive/empty calibration
    │   └── scripts/run_sim_methods.sh
    └── env/                      # simulator setup & smoke tests
        ├── code/                 # smoke_thor_gl.py, thor_url_linux64.py
        └── scripts/              # wsl_fetch_linux64.sh, wsl_smoke_gl.sh, wsl_check_gl.sh
```

The code resolves all paths relative to `embodied_skill/`, so the repository can live in any
directory. Run commands from the `embodied_skill/` folder unless noted otherwise.

---

## 2. Installation

The pipeline has two sides that can be set up independently:

- **Routing / offline evaluation / training / plotting** — pure Python, runs on Windows or
  Linux, CPU or a single small GPU.
- **Simulation (SR / GC)** — AI2-THOR, which needs a Linux environment with a GPU and an
  OpenGL display. We ran it under WSL (Windows Subsystem for Linux) with WSLg GPU rendering.

```bash
# Side A: routing / evaluation (Python 3.10 recommended)
python -m venv .venv && source .venv/bin/activate      # or conda create -n escover python=3.10
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
The non-LLM methods (BM25, Dense, LoTa-Bench, and the rule/head variants of ESCOVER) require
**no API key**.

---

## 3. Data included in this repository

**Skill pool** (`embodied_skill/skills/datasets/`):

| File | Contents |
|---|---|
| `atomic_skills.json` | 13 atomic skills (typed params, effects, AI2-THOR bodies) |
| `composite_skills_seed.json` | 5 hand-authored gold composite skills |
| `composite_skills_generated.json` | LLM-synthesized composite skills |
| `confusing_variants_generated.json` | adversarial distractors: reorder / missing / lookalike (with `variant_of`, `why_wrong`) |
| `pool_clean.json` | validated pool = 125 gold composite + 373 distractors |
| `pool_tier_{low,mid,high}.json`, `tiers_manifest.json` | confusion-density tiers (×1 / ×3 / ×6 distractors over a shared 34-task base) |

Loading `atomic_skills.json` + `pool_clean.json` yields the **511-skill routing pool**.

**Tasks & ground truth** (`embodied_skill/data/`, `embodied_skill/gt/`):

| File | Contents |
|---|---|
| `gt/datasets/gt_kitchen_eval.jsonl` | **211 evaluation queries** (ALFRED valid seen+unseen) with `required_effects` and gold skill sets |
| `gt/datasets/gt_kitchen_train.jsonl` | 2607 training queries (ALFRED train split) for the L0/L3 heads |
| `gt/datasets/gt_kitchen.jsonl` | full 2818-query set |

Ground truth is **effect-grounded**: correctness is judged by whether a selected skill set's
*re-computed* effect signatures cover `required_effects`, not by skill-id match. This is the
single criterion shared by every method (see `router/code/common.py`).

> The intermediate `tasks_kitchen*.jsonl` files are **not** shipped (they embed absolute
> download paths). Regenerate them from raw ALFRED with `data/code/load_tasks.py` if needed;
> the shipped ground-truth files are sufficient for the full routing/evaluation pipeline.

**Precomputed artifacts** (`embodied_skill/eval/datasets/`) let you inspect paper numbers
without re-running anything: `selections.json` (per-query routing outputs of all methods),
`sim_results.json` (SR/GC), `confusion_curve.json`, `confusion_ablation.json`,
`wp8_breakdown.json`, LLM caches, and the figures used in the paper.

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
# synthesize composite skills and confusable variants (needs OPENAI_API_KEY)
bash skills/scripts/run_synth_batches.sh composite 6 4
bash skills/scripts/run_synth_batches.sh variant 0 16 2 4
# validate + merge into the routing pool
python skills/code/validate_skills.py --extra composite_skills_generated.json confusing_variants_generated.json
# build the confusion-density tiers
python skills/code/build_tiers.py --tiers 1 3 6 --seed 0
```

### 5.2 Train the two ESCOVER heads

```bash
cd embodied_skill
python router/code/l0_head.py                # → router/models/l0_head.pt
python router/code/l4_head.py --l0 head      # → router/models/l4_head.pt
```

### 5.3 Routing + offline metrics for all methods

```bash
cd embodied_skill
python eval/code/run_pilot.py --n -1 --seed 0 --gt gt_kitchen_eval.jsonl \
    --methods bm25 dense b0 b3 b4 m1 m1h m1h4
# b4 (LoTa-Bench) uses a local model; the paper configuration:
#   export B4_MODEL=Qwen/Qwen2.5-3B
```

### 5.4 Real-execution simulation (SR / GC) in AI2-THOR

```bash
# in the Linux/WSL environment with a GPU display
bash embodied_skill/env/scripts/wsl_fetch_linux64.sh      # fetch the AI2-THOR build (once)
DISPLAY=:0 GALLIUM_DRIVER=d3d12 python embodied_skill/env/code/smoke_thor_gl.py   # verify GPU OpenGL

cd embodied_skill
DISPLAY=:0 GALLIUM_DRIVER=d3d12 python exec/code/run_sim_from_selections.py \
    --selections eval/datasets/selections.json --merge
# or replay one method at a time:
bash exec/scripts/run_sim_methods.sh "M1-Head+L4" B0-LLM-Full B3-SkillFlow
```

### 5.5 Confusion-density curve, ablations, breakdown

```bash
cd embodied_skill
python eval/code/run_confusion_curve.py --methods m1h4 bm25 dense b0 b3 b4 --merge
python eval/code/plot_confusion_curve.py          # → confusion_curve.{png,pdf}
python eval/code/_confusion_ablation.py           # per-layer attribution figures
python eval/code/_arch_ablation.py                # leave-one-component-out table
python eval/code/wp8_breakdown.py                 # per-subset SR/GC breakdown
```

---

## 6. Method names and main results

The CLI method flags map to the paper as follows:

| CLI flag | Paper name | Description |
|---|---|---|
| `gold` | Oracle | ground-truth skill set (ceiling) |
| `bm25` | BM25 | Okapi BM25 sparse retrieval |
| `dense`| Dense | `bge-small-en-v1.5` dense retrieval |
| `b3`   | SkillFlow | dense → `bge-reranker-base` → LLM selection |
| `b0`   | LLM-Full | whole pool in the prompt, LLM selects |
| `b4`   | LoTa-Bench | local `Qwen2.5-3B` with logprob-constrained decoding |
| `m1`   | M1-Rule | rule goal parser + set-cover (ablation) |
| `m1f`  | LLM goal parser | few-shot LLM goal parser (ablation) |
| `m1h`  | M1-Head | trained Goal Parser head |
| `m1h4` | **ESCOVER** | Goal Parser head + effect-grounded tie-break + set-cover |

**Main result — real-execution Success Rate (SR) and Goal Condition (GC), n = 211:**

| Method | SR | GC |
|---|---:|---:|
| Oracle (gold) | 88.6 | 95.3 |
| **ESCOVER (`m1h4`)** | **82.0** | **92.5** |
| M1-Head (`m1h`) | 80.6 | 91.5 |
| M1-Rule (`m1`) | 78.7 | 90.5 |
| LLM-Full (`b0`) | 72.5 | 85.2 |
| SkillFlow (`b3`) | 65.9 | 79.9 |
| Dense (`dense`) | 31.8 | 61.3 |
| LoTa-Bench (`b4`) | 29.4 | 56.2 |
| BM25 (`bm25`) | 7.6 | 37.8 |

(These are exactly the numbers in `eval/datasets/sim_results.json`.)

---

## 7. Hardware and models

- **Hardware.** All experiments run on a single consumer GPU with **8 GB VRAM**; no multi-GPU,
  no reinforcement learning, and no models larger than 3B are required.
- **Encoders / models.** Dense retrieval and the Goal Parser head use `bge-small-en-v1.5`;
  SkillFlow reranking uses `bge-reranker-base`; LoTa-Bench uses `Qwen2.5-3B`; LLM-Full and
  SkillFlow selection use an OpenAI-compatible chat model. Hugging Face models download on
  first use.
- **Trainable parameters.** ESCOVER learns only two tiny heads on top of a frozen encoder
  (~28K parameters total): a multi-label Goal Parser head and an effect-grounded tie-break
  head. Everything else is symbolic.

---

## 8. Notes for reviewers

- The offline pipeline (§4) is fully deterministic and needs neither a GPU nor network access.
- SR/GC numbers come from real execution in AI2-THOR; simulation is stochastic in wall-clock
  time but the reported metrics are stable across runs.
- We report SR/GC on a decoupled-perception protocol (ground-truth object states are provided),
  so these numbers are **not** directly comparable to the ALFRED leaderboard; see the paper for
  the discussion of comparability.

## License

Released for anonymous peer review. A permissive open-source license will accompany the
public release.
