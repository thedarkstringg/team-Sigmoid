# Label Shift Analysis (Task 17)

Compares fault-label characteristics across farms - distinct from Task 16's sensor-feature shift analysis.

| Farm | Split | Positive rate | # fault events | Median event duration (timesteps) |
|---|---|---:|---:|---:|
| A | test | 0.0561 | 10 | 3120.0 |
| B | test | 0.0208 | 6 | 4631.5 |
| C | test | 0.0365 | 4 | 3043.0 |

Ratio of highest to lowest positive rate across farms: **2.70x**


## Label derivation methodology (documented, not computed)

- **Farm A:** EDP fault logbook: explicit onset timestamp per event, curated by the site operator from SCADA alarms and maintenance records.
- **Farm B:** Operator-provided operating-mode codes plus service reports; onset inferred from mode-code transitions, not a single curated timestamp (per Gueck & Roelofs 2024).
- **Farm C:** Operator-provided operating-mode codes plus service reports; same derivation method as Farm B, distinct from Farm A's logbook-based labels (per Gueck & Roelofs 2024).
