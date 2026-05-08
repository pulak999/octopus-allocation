# 2026-05 — reward ablation artifacts

Console captures and logs from reward-variant training and eval batches. Full training/eval outputs live under `output/` locally (gitignored).

| File | Contents |
|------|-----------|
| `ablation1.txt`, `ablation2.txt` | Ablation run consoles |
| `eval1.txt` | LVL01 eval batch console |
| `eval1_wave2_rerun.txt` | LVL01 wave-2 re-run after fix |
| `eval_batch2_AMS20_console.txt` | AMS20 parallel eval batch console |
| `eval2_AMS20_rerun.txt` | Sequential AMS20 re-eval (R1–R3, current) |
| `greedyrun.txt`, `greedyrun.txtclear` | Greedy baseline runs |
| `speedprobe_train.log` | Training speed probe |

Scripts that produced these live in `scripts/experiments/` (run from repo root: `bash scripts/experiments/eval_batch1_LVL01.sh`, etc.).
