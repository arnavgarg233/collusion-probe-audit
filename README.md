# An access-fair fusion test for white-box monitoring of deceptive language models

Code and result records for the paper "An access-fair fusion test for white-box monitoring of deceptive language models" (Information Fusion submission).

**Author**: Arnav Garg (ORCID [0009-0001-1915-8655](https://orcid.org/0009-0001-1915-8655)), Independent Researcher

## Abstract

Detectors that read a language model's internal activations are increasingly used to catch covert collusion and hidden capability in language-model agents, and they report an area under the receiver operating characteristic curve (AUROC) between 0.9 and 1.0. These figures are usually compared against a text monitor that does not see the full record, so it is unclear whether activations add evidence that an equally informed behavioral monitor lacks. We treat the question as one of decision-level fusion. A probe's value is the gain from fusing its score into a monitor that already reads the full transcript and behavior, estimated by cross-fitted stacking, calibrated against a group-aware permutation floor and reported at a fixed false-positive rate. Across fifteen settings this test reveals an information-access boundary: the activation view adds evidence only when the deceptive content never reaches the observable channel. When collusion is carried out through votes or trades, a behavioral monitor with the same access already catches it and the reported white-box edge disappears. A token-equalization attack shows that the remaining probe signal survives length equalization. When the content is withheld, the activation view can be decisive. A sandbagging model's hidden answer decodes from its activations at AUROC 0.99 while its visible answer is always wrong, and a code backdoor is flagged at a 1% false-positive rate where a text monitor flags none. The test provides a reporting standard for white-box monitors: the fusion gain over a full-access baseline at a fixed false-positive rate.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/). The replay runs offline on CPU and needs no model weights, GPU or API key.

```bash
uv sync --locked
```

## Reproducing the results

```bash
bash reproduce.sh          # a few minutes on a laptop
FULL=1 bash reproduce.sh   # adds the question-text baseline of Section 6 (about 9 more minutes)
```

This checks the 324 numbers listed in `results/tables/claims.json` against the stored records, runs the tests, redraws Figs. 1 to 4, recomputes the sandbagging decode (Table 12, Fig. 4) from the stored activations, and checks the file checksums. The entry points are `scripts/verify_claims.py`, `scripts/recompute_sandbagging.py`, `scripts/make_figures.py` and `scripts/verify_repository.py`.

## Code structure

- `src/collusion_probe_audit/audit.py` - the fusion test: cross-fitted stacking on a full-access baseline, the group-block permutation floor and the decision rule. `audit(X, y, groups, visible=..., texts=...)` runs it on any set of activations.
- `src/collusion_probe_audit/decoders.py`, `sandbagging.py` - the withheld-answer decoder and the recomputation of Section 6.
- `src/collusion_probe_audit/claims.py`, `derive.py`, `figures.py` - the claim checker and the figures.
- `configs/` - estimator, decoder and the fifteen settings of Section 5.
- `scripts/experiments/` - the scripts that produced the stored records. `reproduce.sh` does not run them; most need the external data in `data/manifest.json`, model weights and a GPU.

## Results

`results/tables/evidence/` holds the result records, out-of-fold scores, the GPT-5.5 judge scores and the activations of the password-locked Mistral-7B organism (`sandbagging/sandbag_quantization_transfer_rawacts.npz`). No dataset text, transcript or model weight is included.

`reproduce.sh` sets `PYTHONHASHSEED=0`, which reproduces the stored permutation floors exactly. The sandbagging decode recomputes identically on macOS 26.5. Apple's Accelerate library differs between macOS releases, and on the GitHub runners (macOS 14, 15 and 26.6) the near-chance layer 16 moves by up to one question and its bootstrap interval by up to 0.01, so that check allows 0.01. Every other layer is identical. The NARCBench technique reproduction is stored as summary values, since its per-run records were not kept.

## License

Released under the [MIT License](LICENSE). External data and models remain subject to their original terms.
