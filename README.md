# SliceShuffle: Cranio-Caudal Sequence Reconstruction

| | |
| --- | --- |
| Final rank | 1st |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 3 |
| Last submission | 2026-06-13 |

## Problem statement

### Overview

This is a **Computer Vision / Medical Imaging** challenge that hides the temporal-axis structure inside a 3D volume and asks the solver to reconstruct it from a shuffled, partially-observed projection. Each test "volume" is **one subject's axial T1-weighted MRI brain scan trimmed to a contiguous window of 48 axial slices** along the cranio-caudal axis (rank 0 = most caudal slice in the window, rank 47 = most cranial). Before the solver sees a volume, **16 axial positions are dropped** and the remaining **32 slices are shuffled into a random presentation order** — index 0 in the shuffled stream is not necessarily caudal, and consecutive `s00.png`, `s01.png`, … files are not anatomically adjacent. The solver gets only the 32 PNGs and must output two structured predictions per volume:

1. `**pred_rank**` — for every visible slice, an integer in `{0, …, 31}` giving its predicted cranio-caudal rank among the 32 visible slices (0 = most caudal of the 32 visible, 31 = most cranial of the 32 visible). Within one volume the 32 `pred_rank` values must form a permutation of `{0, …, 31}`; the grader does not enforce this, but ties / out-of-range values directly hurt the Kendall-τ term.
2. `**pred_missing_mask**` — a 49-character string `"M" + 48 chars in {"0", "1"}`. The literal `"M"` prefix is a non-digit guard that prevents CSV round-trips from coercing the column to a numeric dtype (without the prefix, pandas auto-infers a 48-digit numeric column to `float64` and round-trips lose the bits). Bit `j` of the 48-character body is `1` if and only if the solver believes original axial position `j` of the underlying 48-slice window was one of the 16 dropped positions. The mask must be **identical on every row of the same `volume_id**`; the grader uses the first row's value (rows sorted by `presented_index`), so disagreement among rows just discards information.

Two independent failure modes are penalised: getting the cranio-caudal **ordering** of the visible slices wrong, and getting the **localisation of the gaps** wrong. The two are not redundant — a model that nails the order can still totally fail to localise the gaps, because gap localisation requires reasoning about *what is anatomically missing between two visible slices*, which is a different operation from ranking.

A small fraction of training labels carry irreducible noise on both the rank labels and the missing-position mask, capping the achievable training accuracy below 1.0. Test labels are clean.

The 48-slice axial window per subject is sampled from inside each subject's brain z-range, but the **exact start and end z-coordinates and the overall span of the window are different per subject** (each subject's brain has a different overall axial extent). The window is also **kept narrower than the full brain z-range** so adjacent visible slices have similar anatomy and pairwise ranking is genuinely hard. The solver therefore cannot assume that `rank 0` corresponds to a constant anatomical landmark — only that within one volume, rank increases monotonically from caudal to cranial. The 16 missing positions follow a non-uniform distribution that is biased away from the centre of the 48-slice window; solvers that assume uniformly-random gaps will systematically mis-localise them.

Each visible slice has been processed by an irreversible per-slice pixel-domain transform: an **independent 50%-probability left-right (horizontal) mirror flip**, a small in-plane rotation, gamma jitter, a brightness shift, additive Gaussian noise, and a JPEG round-trip, then PNG save. Because the horizontal flip is decided **per slice**, the left-right orientation is *not* consistent down a volume — adjacent slices may be mirrored relative to one another, so left-right anatomical continuity is not a reliable ordering cue and models should be designed to be invariant to (or to reason jointly despite) per-slice mirroring. All transform parameters are private and applied independently per slice, so a "is slice A darker / sharper than slice B" shortcut is not reliable. The same family of transforms is applied identically to train and test so the pixel-statistic distributions match across splits, and so that pixel-level matching against external collections is not feasible.

### Evaluation

For every test `volume_id`, the grader computes:

```
rank_score(volume) = max(0, kendall_tau_b(pred_rank, true_rank)) ** 2
mask_score(volume) = macro_f1(pred_missing_mask, true_missing_mask) ** 2
S(volume)          = 0.60 * rank_score(volume) + 0.40 * mask_score(volume)

Final              = mean over test volumes of S(volume), clipped to [0, 1]
```

Both per-volume terms are **squared** so that the score compresses aggressively at the high end: a Kendall τ of 0.9 contributes 0.81 to the rank term (instead of 0.9), and a macro-F1 of 0.7 contributes 0.49 to the mask term. The squaring widens the relative gap between an "almost there" agent and a perfect submission, leaving more room for genuine human-vs-agent skill differentiation.

- `kendall_tau_b` is the standard tau-b coefficient over the 32 `(presented_index, pred_rank)` tuples of one volume, which lies in `[-1, +1]`. `max(0, τ) ** 2` puts it in `[0, 1]` for any positive correlation, with anti-correlation collapsing to 0; a uniformly random permutation has `E[τ] = 0` and therefore expected `rank_score = 0` — random guessing earns no free leaderboard floor.
- `macro_f1` averages the F1 of class `"0"` and class `"1"` over the 48 binary cells of `pred_missing_mask` vs `true_missing_mask`, and is then squared. The grader takes the **first row's** `pred_missing_mask` per `volume_id` (rows sorted by `presented_index`); per-row variation within a volume is silently discarded.

**Higher is better.** Minimum: 0.0, Maximum: 1.0.

A row that is missing entirely from the submission contributes `pred_rank = 0` and `pred_missing_mask = "M" + "0" * 48` (the 49-character all-zeros mask) for that `(volume_id, presented_index)` pair, so missing rows pull the volume score down rather than zeroing the whole submission. The grader returns `0.0` if `submission` lacks any of the required columns, contains duplicate `(volume_id, presented_index)` pairs, or otherwise raises an exception during parsing — robust submissions must obey the schema.

`pred_rank` values out of range are clipped to `[0, 31]`; `pred_missing_mask` values are sanitised to a 48-character `"0"`/`"1"` string (non-binary chars are stripped, over-long strings are truncated, short ones are right-padded with `"0"`). Submitting illegible / corrupt strings therefore degrades the score smoothly rather than zeroing the whole submission.

### Dataset

- `public/train/<volume_id>/` — one folder per training volume, each containing 32 PNG files `s00.png` … `s31.png` of size 192×192 grayscale.
- `public/test/<volume_id>/` — one folder per test volume, each containing 32 PNG files `s00.png` … `s31.png` of size 192×192 grayscale.
- `public/train.csv` — one row per `(volume_id, presented_index)` pair on training volumes, with the cranio-caudal rank label and the 48-bit missing-position mask. Both labels carry the seeded train-side noise described in the **Overview**.
- `public/test.csv` — one row per `(volume_id, presented_index)` pair on test volumes, with no labels.
- `public/sample_submission.csv` — one row per `(volume_id, presented_index)` pair on test volumes, in the exact submission format. The shipped values are a deliberately weak placeholder (`pred_rank = presented_index` and a constant `pred_missing_mask`); participants must overwrite both columns to score meaningfully.

Volume counts are seed-dependent and printed at the end of `prepare.py` (typical: ~480 train volumes and ~100 test volumes; one volume = 32 PNGs + 32 CSV rows).

### File overview

Files shipped to participants: `public/train/<volume_id>/sNN.png` (192×192 grayscale axial slices, 32 per training volume), `public/test/<volume_id>/sNN.png` (192×192 grayscale axial slices, 32 per test volume), `public/train.csv` (labels for training volumes), `public/test.csv` (ids only, no labels), and `public/sample_submission.csv` (submission template with the full 5-column shape).

### Feature Details

The columns differ between the training CSV, the test CSV, and the submission CSV. They are listed separately so there is no ambiguity about which columns belong to which file.

**Training data columns (`public/train.csv`) — 5 columns:**

Columns (in order): `row_id` (int, unique row identifier), `volume_id` (int, volume identifier with 4-digit zero pad), `presented_index` (int, slice index in the stream, 0–31), `true_rank` (int, cranio-caudal rank of the slice, 0–31), `true_missing_mask` (string, `"M"` + 48 `"0"`/`"1"` chars = 49 chars total).

**Test metadata columns (`public/test.csv`) — 3 columns:**

Columns (in order): `row_id` (int, unique row identifier), `volume_id` (int, matches a folder under `public/test/`), `presented_index` (int, slice index in the stream, 0–31). No label columns are present.

**Submission columns (`public/sample_submission.csv` and your final submission) — 5 columns:**

Columns (in order): `row_id` (int, same set as `public/test.csv`), `volume_id` (int, from `public/test.csv`), `presented_index` (int, from `public/test.csv`, 0–31), `pred_rank` (int, 0–31, predicted rank for this slice), `pred_missing_mask` (string, `"M"` + 48 `"0"`/`"1"` chars, identical across all rows of a volume).

### Submission

Submit a CSV file with a header row and **exactly one row per `(volume_id, presented_index)` pair in `test.csv**`. The header must contain these **5 columns in this order**: `row_id`, `volume_id`, `presented_index`, `pred_rank`, `pred_missing_mask`.

**Requirements:**

- Header row plus exactly one row per `(volume_id, presented_index)` pair appearing in `test.csv` — `row_id` must equal exactly the set in `test.csv`.
- `pred_rank` must be an integer in `[0, 31]`. Out-of-range values are clipped, but ideally each volume's 32 `pred_rank` values form a permutation of `{0, …, 31}` because Kendall-τ rewards correctly-ordered pairs.
- `pred_missing_mask` must be a 49-character string `"M" + 48 chars in {"0", "1"}`. The literal `"M"` prefix is required so that pandas does not coerce the column to a numeric dtype on CSV read (a 48-digit numeric column would round-trip through `float64` and lose information). The grader is permissive: any non-binary characters are stripped, and the result is right-padded or truncated to 48 binary entries — but a correctly-formed `"M..."` string of length 49 always scores best.
- `pred_missing_mask` should be the **same** on every row sharing the same `volume_id`. The grader takes the first row's value (rows sorted by `presented_index`); per-row variation within a volume is silently discarded.
- Missing rows are filled in with `pred_rank = 0` and `pred_missing_mask = "M" + "0" * 48`, which lowers the score for the affected volumes rather than zeroing the whole submission.
- The grader returns `0.0` only if any of the five required columns are missing from the submission.

**Example of a correctly formatted submission file (illustrative only):**

The three rows below come from one volume `volume_id = 0` with three of its 32 visible slices. The full submission has 32 rows for this `volume_id` (and 32 rows per other test `volume_id`). The `pred_missing_mask` is the same 49-character `"M..."` string on every row of the same `volume_id`.

```
row_id,volume_id,presented_index,pred_rank,pred_missing_mask
0,0,0,5,M111110000100001000010000100001000010000100001111
1,0,1,17,M111110000100001000010000100001000010000100001111
2,0,2,11,M111110000100001000010000100001000010000100001111
```

### What Not To Do

Using any of the approaches below is grounds for solution rejection on review, regardless of leaderboard score:

- **Reverse-image-search or pixel-hash any visible PNG** against any public 3D MRI archive in order to recover the original subject identity, original axial index, or DICOM header. The 32 visible slices and the 48-bit mask must be recovered from the provided PNGs alone.
- **Hosted / closed-source APIs** at any stage of training or inference (OpenAI, Anthropic, Google, Cohere, Mistral-API, xAI, etc.), including any distillation / pseudo-labelling from such teachers. Only open-weights models and self-trained pipelines are permitted.
- **Externally pretrained 3D-MRI sequence models** whose training set overlaps with the upstream collection these slices were drawn from. Generic ImageNet / DINOv2 / CLIP / natural-image backbones are fine; fine-tuned brain-MRI sequencers that already encode the target axial coordinate are not.
- **Filename / order side-channels.** Do not assume that the lexicographic order of `s00.png` … `s31.png` matches any anatomical order, and do not exploit file-system metadata, file mtimes, or any signal outside the PNG pixels and the CSVs.
- **Probability / format hacks that game the grader.** Submitting non-permutation `pred_rank` values to deliberately exploit Kendall-τ tie-handling, padding `pred_missing_mask` with non-binary characters, duplicate `(volume_id, presented_index)` pairs, or per-row-varying masks for the same `volume_id` to try to game the "first row wins" rule.
- **Grader / platform exploitation.** Hard-coded answer dictionaries, filesystem probes for `private/answers.csv`, attempts to read hidden splits, or any channel that is not `public/train/`, `public/train.csv`, `public/test/`, and `public/test.csv`.
- **Ensembles mixing allowed and prohibited components.** An ensemble is allowed only if every component is itself trained (or used zero-shot) within the rules above. One prohibited component contaminates the whole ensemble.

---
