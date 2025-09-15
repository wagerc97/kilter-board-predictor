---
license: mit
language:
- en
size_categories:
- 100K<n<1M
---

# Kilter Board L1‑S10 (clean) — train / val / test

Filtered problems for **layout 1, size 10, sets {1, 20}**.  
Rows are keyed by `(uuid, angle)`; splits are **80 / 10 / 10 %** on UUIDs.

## Files
| table in `kilter_splits.sqlite` | content |
| --- | --- |
| `kilter_train`, `kilter_val`, `kilter_test` | filtered climbs |
| `difficulty_grades`, `placements`, `placement_roles`, `holes` | reference look‑ups |

## Filters
* `ascensionist_count > 0`
* `difficulty_numeric ≤ 30.5` (~ V13)
* `num_holds ≥ 3` and `≤ 50`

Data source: **BoardLib** repo + Kilter app.