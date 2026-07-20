# Contributing to HFP

Thank you for your interest in HFP. Please read this before opening a pull request.

## Contributor License Agreement (required)

HFP is published under AGPL-3.0 and is additionally offered under commercial
licenses by the copyright holder (dual licensing). To keep that possible, the
project can only accept contributions whose licensing status is unambiguous.

**Every contribution requires agreeing to the [Contributor License Agreement](CLA.md).**
In short: you keep the copyright to your contribution, but you grant the
project maintainer a perpetual, irrevocable, sublicensable license to use and
relicense it — including under commercial terms. Contributions without a CLA
agreement cannot be merged, regardless of quality.

How to agree: include the line below in the description of your first pull
request, and add a `Signed-off-by: Your Name <email>` line to your commits
(`git commit -s`):

> I have read the CLA (CLA.md) and I agree to its terms for this and all my
> future contributions to this repository.

## Ground rules

- **Honest claims only.** This project deliberately keeps its claims calibrated.
  PRs that add performance or capability claims must include the experiment
  (command, seed, output) that backs them. Single-seed results are labeled as
  preliminary. "Physics-inspired" naming is welcome only when the README can
  state precisely what the mechanism does in ML terms.
- **Baseline stays clean.** New mechanisms ship behind a flag, default off,
  so `decay_mode="exp"` / defaults remain an untouched baseline.
- **License headers.** Run `python add_license_headers.py` after adding any
  `.py` file (CI-style check: `python add_license_headers.py --check`).

## Before you open a PR

```bash
python smoke_test.py                          # must pass in full
python review_scripts/verify_claims.py        # correctness suite (torch)
python add_license_headers.py --check         # license headers present
```

If your change touches the recurrence (`hfp_bulk_state.py`), also confirm
chunk-consistency: full-sequence forward must equal state-carried chunked
forward for every `decay_mode` × `key_feature_map` combination (covered by
`verify_claims.py`).

## Scope notes

Bug reports and reproductions of the retention/recall experiments are the most
valuable contributions right now. Architectural additions are welcome but will
be held to the "flag off by default + honest ablation" standard above.
