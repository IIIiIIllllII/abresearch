# Phase 2 summary

Judge: Qwen3-8B bf16, rubric v2. Hand audit 52/60 exact (refusal 20/20, evasive 16/20, compliance 16/20); comply-vs-not 55/60. Old string-match judge agreed with hand 'real compliance' only 38/60.

## Headline table (n=48 unless noted)

| question | answer | evidence |
|---|---|---|
| Is the L10 one-way dissociation specific to r_a? | **Yes** | D_ra: content kept 0.96, destr comply 0.77 vs A 0.35. 0/15 controls clear A's CI; max control 0.56 (ctr_math_vs_poetry) |
| Does it appear in 2507 or 1.7B? | **2507 partly, 1.7B no** | 2507 L20 holds at n=48 (content kept 0.92, destr comply 0.85, +0.27); L14 fails at n=48. 1.7B: 0 flagged layers |
| Early decision vs late expression layer in all models? | **No, 2 of 4** | Qwen3.5-4B and 2507 shift to evasive at the late layer; 1.7B and 8B comply at both |

## Per-model early/late B_rc (content_harm, n=48)

| model | early L | late L | early real-compl | late real-compl | late evasive | cos(r_c early, late) |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 10 | 24 | 0.92 | 0.23 | 0.71 | 0.21 |
| Qwen3-4B-2507 | 17 | 24 | 0.06 | 0.00 | 0.73 | 0.34 |
| Qwen3-1.7B | 12 | 18 | 0.94 | 0.88 | 0.10 | 0.26 |
| Qwen3-8B | 16 | 19 | 0.88 | 1.00 | 0.00 | 0.43 |

## Caveats

- Action-caution floor: 1.7B/8B comply with 0.88-0.81 of destructive requests already in A, so 'preserved caution' is untestable there.
- 3 of 5 benign-contrast controls move destr comply to 0.50-0.56 (A=0.35, CIs overlap): r_a is specific, but L10 action caution is not robust.
- Scan screening used n=24; only 2507 L20 survived n=48. Treat single-layer scan hits as noise until rerun.
- content_harm is mlabonne/harmful_behaviors only (HarmBench/AdvBench gated).
- r_a / D_ra direction vectors are NOT in the tarball (per your rule); recompute locally from the shipped activations/ with `python -m abr.directions`.
