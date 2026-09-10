# Team Sigmoid: Early Fault Detection & Risk-Based Maintenance Prioritization for Wind Turbines

**Track 2 (Applied/Domain Research), DLE-AI-202, Cohort I 2026**
Team: Ziyad Muradov, Ismayil Yusifli, Emin Huseynli, Kamal Soltanaliyev

## What this project does

Trains a GRU-based sequence model on multi-farm SCADA (turbine sensor) data
to predict fault risk at every 10-minute timestep, using a physically
meaningful, farm-agnostic feature representation so the same trained model
can be evaluated on turbines and farms it was never trained on. The
project uses **Farm C as the source domain** and evaluates zero-shot
transfer to **Farms A and B** through a strict ten-feature physical
representation.

The authoritative feature mapping is `configs/physical_sensor_mapping.yaml`
and the full server workflow (audit -> split -> export -> validate ->
train) is documented in `SERVER_RUNBOOK.md`.

## Dataset

CARE-to-Compare (Gück & Roelofs, 2024), Fraunhofer IWES / Zenodo
(DOI: 10.5281/zenodo.10958775), licensed CC BY-SA 4.0. 36 turbines across
3 farms (Farm A onshore Portugal, Farms B/C offshore Germany),
10-minute-resolution SCADA data.

**No raw CARE data, generated NumPy/Parquet arrays, model checkpoints, or
Kaggle caches belong in this repository** - see `.gitignore`. Data is
downloaded and processed on the compute environment per `SERVER_RUNBOOK.md`.

## Repository structure

```
configs/          physical_sensor_mapping.yaml - the cross-farm feature contract
src/data/         acquisition, quality audit, sequence export (Ismayil)
src/baseline/     classical baseline models - LogReg, HistGradientBoosting (Ismayil)
src/model/        GRU model, training loop, evaluation (Ziyad)
src/eval/         lead-time, calibration, cost-sensitive metrics, plots,
                  cross-farm evaluation, deployment note (Emin)
src/prioritization/  risk-prioritization formula, AccessCost, criticality (Kamal)
docs/             physical feature contract and other technical documentation
artifacts/        computed results (baseline metrics, distribution-shift
                  analysis, label-shift analysis, turbine-model-shift analysis)
figures/eval/     generated evaluation figures (reliability, lead-time,
                  per-turbine breakdown)
report/           IEEE-format paper (report.tex)
presentation/     slide deck
tests/            unit and integration tests
run_all.sh        one-command pipeline entry point (see below)
PIVOT_NOTE.md     record of the Farm A -> Farm C pivot and why
FEATURE_PIVOT_NOTE.md   record of the raw-sensor -> physical-feature pivot
SERVER_RUNBOOK.md       exact commands to run the full pipeline on the
                        shared compute environment
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA-specific PyTorch build; if training on a
machine with a different CUDA version, reinstall torch for that machine's
driver before running (see `requirements.txt`'s top comment).

## How to get the data

Data is not included in this repository. Follow `SERVER_RUNBOOK.md` to
download the raw CARE-to-Compare data (via `scripts/download_care_farm.py`,
Kaggle credentials required), audit it, generate the frozen train/val/test
split, and export the physical-feature sequences.

## Reproducing the headline results

```bash
bash run_all.sh
```

This runs, in order: the Farm C baseline, GRU training, GRU evaluation,
per-timestep prediction export, and the cross-farm generalization check.
Evaluation figures (`src/eval/metrics.py`, `plots.py`) and the
prioritization pipeline currently require running their functions directly
(see the scripts' own docstrings for the exact input format) rather than
a single CLI flag - this is documented explicitly, not hidden, in
`run_all.sh`'s own output.

## Key findings (see `report/report.tex` for full detail)

- A GRU trained on Farm C reaches validation ROC-AUC 0.57-0.67 across six
  tested configurations, but held-out test ROC-AUC of only 0.40-0.48 -
  consistent with (and no worse than) classical baselines trained on the
  same data, indicating the limitation is structural to the dataset's
  turbine diversity rather than specific to any one model.
- Cross-farm distribution-shift analysis quantifies real, farm-specific
  physical differences (e.g., a distinct gearbox ratio per farm,
  compression factors up to 66x on some features under a frozen scaler),
  documented in `artifacts/distribution_shift.md`,
  `artifacts/label_shift.md`, and `artifacts/turbine_model_shift.md`.

## AI-assistance disclosure

Team members used AI assistance (Claude) for code review, debugging
support, and drafting portions of documentation and the paper from
decisions made during development. All architecture, methodology, and
result interpretation decisions were made and are understood by the
authors independently of this assistance. See `report/report.tex`'s
"Tools and Acknowledgements" section and `contribution_report.pdf` for
full detail.
