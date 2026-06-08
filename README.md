# collusion-probe-audit

**A confound audit of activation-based AI-deception detection** - when does reading a model's internal activations tell you something its visible behaviour does not?

Recent work claims multi-agent collusion is detectable from model activations *"even when text
outputs appear entirely normal."* This repository tests that claim with a single estimator - a
permutation-floored, cross-fitted **text-controlled delta**, an estimate of the conditional mutual
information `I(deception ; activations | visible behaviour)` - on the benchmark the claim was made
on, and contrasts it against a regime where capability is genuinely withheld from output.

It is a **controlled audit, not a detector**. The probes *do* separate colluders (0.9-1.0 AUROC);
what the audit corrects is the *interpretation*. Once the visible record the original evaluation
omitted (votes, bets, length, the full transcript) is controlled for, the activation advantage for
executed deception collapses to within its permutation floor - it is the overt behaviour, measured
a different way. White-box monitoring carries non-redundant signal only where the deceptive content
is *absent from the visible channel*.

| | |
|---|---|
| **Estimator** | `delta = AUROC(visible + activation) − AUROC(visible)`, pooled out-of-fold, cross-fitted |
| **Validation** | `GroupKFold` by scenario/run + a label-permutation floor on every delta |
| **Fairness** | the text baseline sees the *whole* transcript the activation encodes (full-vocab TF-IDF / a 32B transcript judge) |
| **Reach** | 16 settings - NARCBench (4 models × core/stego/transfer), Liars'-Bench, b2w, sleeper/backdoor, password-locked sandbagging |

## Headline results

**Collusion - confounded by visible behaviour** (controlled delta within the permutation floor):

| model / tier | activation | full-access text | controlled delta | perm floor | verdict |
|---|---|---|---|---|---|
| gpt-oss-20B core | 0.963 | 0.861 | **+0.102** | 0.219 | confounded |
| qwen3-32B core | 0.862 | 0.980 | **−0.119** | 0.120 | confounded |
| qwen3-32B transfer (non-saturated) | 0.669 | 0.921 | **−0.251** | 0.096 | confounded |
| gpt-oss-20B core, **vs 32B transcript judge** | 0.963 | **0.954** | **+0.021** | - | confounded |

Even against a strong 32B judge reading the full transcript, the activation adds ~0.02 AUROC. The
confound does not weaken with scale - the 70B model is the *most* confounded.

**Sandbagging - white-box wins** (the positive control): on a password-locked Mistral-7B the
activation decodes the *withheld* correct WMDP answer at **0.913** (robust 0.88-0.93 across late
layers) while the visible output is **0.000** and the question text alone is **0.228** (≈ chance).

**The regime map (16 settings): 12 confounded / 4 white-box-wins.** The boundary is *information
access*, not a hidden-vs-executed label - white-box helps only when the deceptive content is
withheld from the visible channel, and the same estimator says which case you are in.

## Repository layout

```
collusion-probe-audit/
├── src/
│   ├── common/        leak-aware stats (group-CV AUROC + permutation null), model I/O
│   ├── synthetic/     four-condition synthetic gate - the plumbing test
│   ├── narcbench/     core analysis - loader, technique reproduction, text-controlled delta
│   ├── audit/         collusion_audit.py - the reusable protocol (self-test included)
│   ├── breadth/       the 16-setting study (judge, sandbagging, sleeper, b2w, backdoor)
│   └── sandbagging/   capability-hiding: extraction + steering
├── scripts/           make_paper_figures.py  (multi-panel publication figures)
├── results/           tables/ (run-of-record metrics) + runpod/ (job scripts + result JSON)
├── figures/           paper figures
├── environment.yml    conda environment (PyTorch + MPS)
└── LICENSE
```

## Reproducing

```bash
conda env create -f environment.yml && conda activate collusion

# NARCBench audit (downloads the published activations slice; CPU/MPS):
python src/narcbench/narcbench_textctrl.py --model gpt_oss_20b --tier core --contrast emergent --n-perm 80

# the reusable protocol + its self-test:
python src/audit/collusion_audit.py --selftest

# regenerate the multi-panel publication figures:
python scripts/make_paper_figures.py
```

The breadth scripts under `src/breadth/` are self-contained RunPod / CUDA jobs.

## License

MIT - see [`LICENSE`](LICENSE).
