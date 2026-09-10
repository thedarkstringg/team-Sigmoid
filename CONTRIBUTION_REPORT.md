
# Contribution Report — Team Sigmoid (DLE-AI-202)

**Project:** Early Fault Detection & Risk-Based Maintenance Prioritization
for Wind Turbines Using Multi-Farm SCADA Data

**Team:** Ziyad Muradov, Ismayil Yusifli, Emin Huseynli, Kamal Soltanaliyev

Each member's contribution is described below, cross-checked against git
history (`git shortlog -sn --all`, `git log --author=...`) and repository
commit content. Git-identity issues found during the project are disclosed
explicitly rather than silently corrected or hidden.

---

## Contribution Table

| Member                 | Code                                                                                                                                                                                                                                                                                        | Report Sections                  | Experiments                                                                                                                                                                | Slides | Other                                                                                                                                                                                                  |     |     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --- | --- |
| **Ziyad Muradov**      | GRU model design, training loop, evaluation script (`src/model/`); `run_all.sh`; `requirements.txt` and repo scaffold; diagnosed and fixed a CSV-delimiter bug, a `grid_power_factor` threshold unit-mismatch bug, and a scaler-precision issue in the Farm C data pipeline                 | Method                           | Trained and tuned the GRU across 6 configurations (learning rate, dropout, feature count, architecture) on Farm A and Farm C; ran the 3-way baseline comparison            | —      |                                                                                                                                                                                                        |     |     |
| **Ismayil Yusifli**    | Data acquisition, cleaning, and quality audit for Farm A and Farm C (`src/data/`); physical feature engineering and cross-farm sensor mapping (`configs/physical_sensor_mapping.yaml`); baseline models (`src/baseline/`) for both farms; Farm A/B re-export using the frozen Farm-C scaler | Experimental Setup, Related Work | Built and ran Logistic Regression and HistGradientBoosting baselines on Farm A and Farm C                                                                                  | —      | Documented the physical-feature domain-generalization contract (`FEATURE_PIVOT_NOTE.md`, `SERVER_RUNBOOK.md`)                                                                                          |     |     |
| **Emin Huseynli**      | Evaluation suite (`src/eval/metrics.py`, `plots.py`): lead-time metric, per-turbine/per-farm breakdown, cost-sensitive metric, calibration analysis; deployment note and Dockerfile; found and fixed a hardcoded input-size bug in the deployment benchmark                                 | Results                          | Cross-farm Task 15 (results table) and Task 16 (distribution-shift analysis, compression-factor method); independently re-verified Tasks 17/18 with his own implementation | +      | Identified a Farm-A/11-feature checkpoint mismatch before it reached the final report .Completed cross-farm Tasks 17 (label shift) and 18 (turbine-model shift) analysis, originally delegated to Emin |     |     |
| **Kamal Soltanaliyev** | Prioritization layer (`src/prioritization/`): core formula, degradation-rate calculation, prediction aggregation, criticality scoring, AccessCost model; full pipeline integration and tests; cross-farm generalization experiment execution and CLI                                        | Discussion & Limitations         | Ran the Farm-C-to-Farm-A/B cross-farm generalization experiment                                                                                                            | —      | Delegated Tasks 17-18 to Emin;                                                                                                                                                                         |     |     |

---

## Disclosed Git-Identity Issues

The team worked from a shared JupyterLab machine for parts of the project.
Multiple times, a person's local git identity was not reset before another
person committed, causing commits to be attributed to the wrong team
member. Each case below was found via `git shortlog -sn --all` and
`git log --author=...`, cross-checked against commit content and branch
names.

1. **Ziyad's commits split across two identities:** `Ziyad Muradov` (10
   commits) and `TheDarkString` (15 commits) — a local git config change
   partway through the project. Combined: 25 commits, all verified by
   content to be Ziyad's work.

2. **Emin's commits split across two identities:** `Emin Huseynli` and
   `ehuseynli96-ops` — same root cause. Combined, correctly reflects
   Emin's evaluation-suite work.

3. **One commit authored by Ziyad appeared under Emin's identity**
   (the Farm-C `grid_power_factor` threshold fix) due to the shared
   machine still having Emin's git config active. Disclosed here; not
   rewritten, since real work from multiple team members had already
   been built on top of it by the time it was found.

4. **4 commits authored by Emin appeared under Ismayil's identity**
   (Tasks 15-16 — results table and distribution-shift analysis). Found,
   verified by content (branch names, task numbering matching Emin's
   convention), and corrected via an interactive rebase before further
   work was built on top, since they sat near the tip of `main` at the
   time. Now correctly attributed to Emin Huseynli.

5. A small number of Ismayil's earlier commits show wording patterns
   similar to Emin's known task areas; this was investigated but left
   uncorrected, since rewriting would have required rewriting a large
   portion of subsequent shared history. Flagged here for transparency
   rather than silently left unmentioned.

---

## Sign-off

By signing below, each member confirms this report accurately reflects
their contribution to the project.

**Ziyad Muradov**
Signature: Ziyad Muradov Date: 9 September, 2026

**Ismayil Yusifli**
Signature: Ismayil Yusifli Date: 9 September, 2026

**Emin Huseynli**
Signature: Emin Huseynli Date: 9 September, 2026

**Kamal Soltanaliyev**
Signature: Kamal Soltanaliyev Date:  9 September, 2026
