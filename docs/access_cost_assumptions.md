# Access Cost Model Assumptions

## 1. Purpose

`AccessCost` represents the relative burden of physically reaching a wind
turbine and carrying out maintenance work on it, not an exact monetary
maintenance cost in USD/EUR/AZN. It is one input to the project's
risk-based maintenance prioritization formula
(`src/prioritization/core.py`):

```
PriorityScore =
    w1 * FaultRisk
  + w2 * DegradationRate
  - w3 * AccessCost
  + w4 * Criticality
```

`AccessCost` enters with a negative sign (`- w3 * AccessCost`): all else
equal, a turbine that is harder or costlier to access should be prioritized
lower than an equally at-risk turbine that is easy to reach, because the
maintenance crew's limited time and transport capacity is a real constraint
on how many interventions can actually be scheduled.

Access burden matters for prioritization because two turbines with similar
technical fault risk (`FaultRisk`, `DegradationRate`) do not necessarily
warrant the same maintenance priority. A turbine that is difficult, slow, or
expensive to reach may need to be scheduled differently (e.g. bundled with
other nearby interventions, or deferred until a suitable weather/vessel
window) than an easily accessible one with the same fault indicators.
`AccessCost` gives the prioritization layer a way to express that
difference explicitly and transparently, instead of ignoring it.

## 2. Why distinguish onshore and offshore?

The operations-and-maintenance (O&M) literature consistently describes
offshore wind maintenance as facing additional accessibility and logistics
constraints that onshore maintenance does not face in the same way,
including:

- **Distance from shore/port** - offshore turbines must be reached by boat
  or helicopter, and travel time/cost generally grows with distance from the
  operations base.
- **Vessel-based access** - technician transfer depends on the availability
  and suitability of crew transfer vessels or similar specialized assets.
- **Specialized logistics** - offshore campaigns typically require more
  coordination (vessel scheduling, port logistics, spare-parts staging) than
  a straightforward onshore site visit.
- **Personnel transfer** - safely transferring technicians from vessel to
  turbine is itself a constrained, weather-sensitive operation.
- **Weather and sea-state limitations** - maintenance access can be blocked
  or delayed by wind, wave height, and other sea-state conditions.
- **Waiting/downtime** - when access is denied by weather, the turbine may
  remain faulted or down for longer, extending effective downtime.
- **Specialized vessels/equipment** - some offshore repairs require
  higher-capability vessels or lifting equipment that are less available
  and more costly to mobilize than onshore road/crane access.
- **Potentially higher O&M burden** - the combination of the above factors
  is widely reported as increasing the practical burden of offshore O&M
  relative to onshore.

**Important caveat:** this project does **not** claim that every offshore
turbine has exactly the same access cost, nor that offshore access always
carries one fixed, universal monetary multiplier. Actual offshore access
burden varies substantially by distance to port, vessel strategy, weather
climate, and farm-specific logistics. The `AccessCost` model implemented
here is a **relative access-burden abstraction**: it distinguishes
"onshore" from "offshore" as a first-order approximation and exposes
configurable components so the abstraction can later be refined or
calibrated, rather than asserting a precise, universally-true cost ratio.

## 3. Model structure

This section documents the model exactly as implemented in
[`src/prioritization/access_cost.py`](../src/prioritization/access_cost.py).

```
raw =
    base_access_cost
    + transport_cost
    + logistics_cost

if farm_type == "offshore":
    raw *= offshore_premium

AccessCost =
    raw * multiplier
```

Component meaning:

- **`base_access_cost`** - a baseline access-burden component applied
  regardless of farm type (e.g. the generic overhead of dispatching a
  maintenance visit at all).
- **`transport_cost`** - a component representing the travel/transport
  burden of reaching the turbine (e.g. distance-related travel effort).
- **`logistics_cost`** - a component representing additional logistics
  burden around the visit (e.g. scheduling, equipment/parts staging,
  coordination overhead).
- **`offshore_premium`** - a multiplicative factor applied to the summed
  `raw` cost **only** when `farm_type == "offshore"`, representing the
  additional burden discussed in Section 2. It is not applied for
  `"onshore"` (equivalent to a premium of `1.0`).
- **`multiplier`** - an optional, additional scaling factor applied to the
  final result, letting a caller rescale the whole `AccessCost` index (e.g.
  to weight it relative to other prioritization inputs) without changing
  the underlying component values.

The `farm_type` input is validated to be exactly `"onshore"` or
`"offshore"` (case-insensitive, whitespace-tolerant), and the function
supports both a single turbine (`str` in, `float` out) and a vectorized
batch of turbines (1D array-like of strings in, `numpy.ndarray` out).

## 4. Default parameters

The current implementation defines these defaults in `AccessCostConfig`:

| Parameter | Default value |
| --- | --- |
| `base_access_cost` | `1.0` |
| `transport_cost` | `0.0` |
| `logistics_cost` | `0.0` |
| `offshore_premium` | `1.5` |
| `multiplier` | `1.0` |

These are **illustrative model parameters, not empirical cost estimates.**
In particular, the default `offshore_premium = 1.5` is not presented as a
measured, industry-wide offshore/onshore cost ratio - it is a configurable
placeholder chosen so the model is testable and interpretable (offshore
access costs more than onshore access, all else equal) before any
farm-specific evidence is incorporated.

Every parameter is intentionally configurable because real access costs
depend on factors this module does not observe, such as: farm location,
distance to port, vessel strategy, weather conditions, turbine type, and
the specific maintenance task being scheduled. Calibrating these parameters
to real conditions is out of scope for this task.

## 5. Literature rationale

| Assumption | Rationale | Source |
| --- | --- | --- |
| A. Offshore accessibility is more constrained than onshore. | Offshore O&M reviews describe turbine access as dependent on vessel/helicopter transfer rather than direct road/crane access available onshore. | [Offshore wind turbine operations and maintenance: A state-of-the-art review](https://www.sciencedirect.com/science/article/pii/S1364032121001805) |
| B. Weather/sea-state can delay offshore maintenance. | Access-probability studies model maintenance access explicitly as a function of weather and sea-state windows, showing delays are a routine part of offshore access planning. | [Accessing offshore wind turbines for maintenance: calculating access probabilities, expected delays and the associated costs using a probabilistic approach](https://strathprints.strath.ac.uk/43668/) |
| C. Vessel and logistics requirements contribute to offshore maintenance cost/burden. | O&M concept studies for near/far offshore wind farms discuss vessel strategy and logistics planning as central cost/organizational drivers distinct from onshore O&M. | [O&M Concepts for Near and Far Offshore Wind Farms](https://publications.ecn.nl/ECN-E--16-055) |
| D. Distance from shore/port affects accessibility and cost. | O&M and safety-focused frameworks for offshore maintenance discuss distance-to-port and transfer logistics as factors shaping maintenance planning and safety/access considerations. | [A Safety-Aware Framework for Offshore Wind Turbine Maintenance](https://doi.org/10.1002/WE.70065) |
| E. Access burden is therefore appropriate as a prioritization feature. | Since accessibility materially affects when/how maintenance can be carried out, a prioritization scheme that only considers technical fault risk and ignores access burden would be an incomplete decision-support signal. | Synthesis of A-D above |

Sources used (full references):

1. "Offshore wind turbine operations and maintenance: A state-of-the-art review", Renewable and Sustainable Energy Reviews, ScienceDirect. <https://www.sciencedirect.com/science/article/pii/S1364032121001805>
2. Dewan, A. & Asgarpour, M., "O&M Concepts for Near and Far Offshore Wind Farms", ECN publication, 2016. <https://publications.ecn.nl/ECN-E--16-055>
3. "Accessing offshore wind turbines for maintenance: calculating access probabilities, expected delays and the associated costs using a probabilistic approach", University of Strathclyde repository. <https://strathprints.strath.ac.uk/43668/>
4. "A Safety-Aware Framework for Offshore Wind Turbine Maintenance", Wind Energy, 2025. <https://doi.org/10.1002/WE.70065>

No page numbers, direct quotations, additional DOIs, author lists, or
numerical findings from these sources are claimed beyond the general,
qualitative rationale summarized above.

## 6. Interpretation and limitations

- `AccessCost` is **not** a direct estimate of USD/EUR/AZN maintenance cost.
- It is a **relative access-burden / index variable** intended for use
  inside the `PriorityScore` formula, not as a standalone financial figure.
- The current default parameters (Section 4) are **illustrative**, chosen
  for interpretability and testability of the model, not measured from
  real operations data.
- Empirical calibration of this model would require farm-specific
  information such as distance to port, vessel availability, weather
  windows, maintenance strategy, and actual O&M cost data - none of which
  is incorporated in this task.
- A single `offshore_premium` factor cannot capture all offshore
  variability (e.g. near-shore vs. far-shore farms, differing vessel
  strategies, differing weather climates); it is a first-order
  simplification, not a claim of uniform offshore cost.
- Onshore farms can also have substantial access costs depending on
  terrain, remoteness, and road infrastructure; "onshore" is not a
  synonym for "always cheap to access" in reality, even though the current
  default configuration treats it as the lower-cost baseline case.
- The model intentionally prioritizes interpretability and reproducibility
  over pretending to have precise monetary estimates, consistent with its
  role as one configurable, transparent input to a research-stage
  prioritization layer.

## 7. Reproducibility

- Implementation: [`src/prioritization/access_cost.py`](../src/prioritization/access_cost.py)
- Tests: [`tests/test_access_cost.py`](../tests/test_access_cost.py)

The model is deterministic (no randomness) and its default parameters and
formula are documented above exactly as implemented; production code and
tests were not modified as part of this documentation task.
