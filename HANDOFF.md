# Wan2.2 Della Handoff

Last updated: 2026-06-11 15:15 PDT.

This is the short handoff for the next agent. The full chronological record is
in `WORKLOG.md`.

## Repository And Cluster

- Local repo: `/Users/lzha/code/Wan2.2`
- Della repo: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- Canonical branch: `main`
- Active implementation branch: `codex/droid-ddp-8gpu`
- Fork remote: `git@github.com:lihzha/Wan2.2.git`
- Upstream remote: `https://github.com/Wan-Video/Wan2.2.git`
- Latest canonical commit before the distributed-training patch:
  `f37022874c588817d4ed77d463e3d27745053df4`
- Distributed-training implementation commit:
  `dd6f1c829a968f00c47947d835a9e6ee1f36d127`
- The canonical Della checkout is intentionally left at the old commit while
  active single-GPU job `9541718` runs. Deploy DDP from an isolated Della
  worktree, not by mutating `/scratch/gpfs/AM43/lz3952/Wan2.2`.

Use Git for tracked code/config/script/docs. Do not deploy tracked source with
`rsync`. Use `rsync` only for logs, videos, checkpoints, datasets, and other
large artifacts.

Important SSH note: use normal `ssh della-gpu`. Do not force
`BatchMode=yes`; that caused false auth failures because the working config uses
keyboard-interactive auth and ControlMaster. At this handoff, a final live
refresh from this Codex process failed because `tigressgateway` auth was not
available, so the next agent should refresh SSH first.

## Scientific Goal

We are testing action-conditioned control of frozen Wan2.2 video generation.
The current best direction is the side adapter:

- frozen Wan backbone remains frozen;
- adapter injects action through trainable residual branches;
- training/eval guidance scale should match;
- realistic tests must use 25 diffusion steps and fresh initial noise.

Key conclusion so far: learning/predicting `z_init` directly from a large DDIM
sample set did not reveal an obvious easy low-rank structure. The practical
## Current Status - 2026-06-11 05:35 PDT

- A DDP implementation is in progress on branch `codex/droid-ddp-8gpu`.
  It adds torchrun/NCCL setup, rank-aware Wan device construction,
  `DistributedSampler`, DDP wrapping, rank-0-only validation/logging/checkpoint
  writes, and a new `run_action_conditioned_droid_dist.sh` launcher.
  Static checks passed locally:
  `git diff --check`,
  `bash -n run_action_conditioned_droid_dist.sh`,
  `bash -n run_action_conditioned_droid.sh`, and
  `/usr/bin/python3 -m py_compile scripts/train_action_conditioned_wan_droid.py`.
  Della worktree:
  `/scratch/gpfs/AM43/lz3952/worktrees/Wan2.2/codex-droid-ddp-8gpu`,
  detached at `f284b18340e1d111bcb30b31fd07a4ed8da0ecfc`, with symlinks for
  untracked runtime assets. Remote compile/import checks and manual focused
  action-conditioner tests passed `11/11`.
- DDP Slurm chain is queued from the isolated worktree:
  `9564218` `act-ddp2-smoke` -> `9564219` `act-ddp8-smoke` ->
  `9564220` `act-ddp8-10k`.
  The full 8-GPU run uses local batch `5`, global batch `40`, LR `5e-5`,
  10k optimizer steps, 25 diffusion steps, and full train/val manifests.
  Dependent jobs must be canceled/replaced if a smoke fails.

- Max-fit DROID batch size on the current single-H200 setup is `5`.
  A real optimizer-step profiler showed batch sizes `6`, `7`, and `8` OOM,
  while `5` passed with about 6.9 GiB free after the step.
- Full DROID train cache is complete:
  train `1,440,554/1,440,554`, val `14,636/14,636`.
  Final cache array `9540854` completed cleanly.
- Current-cache DROID job `9497852` completed cleanly at step `10000`.
  Final validation loss is `0.19281196547672153`.
- Step-10000 eval was submitted manually as job `9542335` because the periodic
  watcher was no longer alive. It completed cleanly.
  Local artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step10000_ep399_v0_s00004_s1000_1001/`
  Viz URL:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step10000_ep399_v0_s00004_s1000_1001`
- Step-10000 eval metrics:
  seed1000 latent MSE `0.22305864095687866` vs null `1.9798624515533447`;
  seed1001 latent MSE `0.2230650782585144` vs null `4.661829471588135`.
  Qualitatively, the table/object layout is preserved, but the robot/gripper
  still smears into a gray cloudy blur by mid/end frames.
- Full-cache batch-5 smoke job `9541649` completed cleanly.
- Premature full job `9541650` was canceled after launch logging only, due a
  waiter job-id parsing bug.
- Waiter guard fix is committed in `8c6cc5d`; latest main/worklog commit is
  `1809c29`, and the Della checkout is fast-forwarded there.
- Active full-cache batch-5 job:
  `9541718` `act-droid-win-10k`, node `della-i21g3`, run dir
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5`.
  It is healthy through at least step `1020`. First validation at step `1000`
  is `0.2770900116302073`; `ckpt_latest.pt` and `ckpt_best_val.pt` were written
  at step `1000` and are about `914M` each. Its 36-hour wall time is likely too
  short for all 10k steps at the observed `~13.6s/step`. `scontrol update` to
  48 hours was denied. Do not patch resume support until the user confirms the
  plan.
- Full-cache batch-5 step-1000 eval job `9551286` completed cleanly using
  `ckpt_step1000.pt`.
  Local artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step1000_ep399_v0_s00004_s1000_1001/`
  Viz URL:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step1000_ep399_v0_s00004_s1000_1001`
  Metrics: seed1000 latent MSE `0.29633891582489014` vs null
  `1.9798624515533447`; seed1001 latent MSE `0.2788466513156891` vs null
  `4.661829471588135`. Visually better than null but worse than the
  current-cache 10k eval, with stronger ghosting/object and robot smearing.
- Full-cache batch-5 step-2000 validation improved to
  `0.24543552426621318`. Step-2000 eval job `9557082` completed cleanly using
  `ckpt_step2000.pt`.
  Local artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step2000_ep399_v0_s00004_s1000_1001/`
  Viz URL:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step2000_ep399_v0_s00004_s1000_1001`
  Metrics: seed1000 latent MSE `0.27191445231437683` vs null
  `1.9798624515533447`; seed1001 latent MSE `0.2750750482082367` vs null
  `4.661829471588135`. Numerically improved from step 1000; visually the
  robot/gripper haze remains.

The current research direction is scaling side-adapter training with fresh
noise.

## Relevant Implementation

- Model wrapper: `models/action_conditioned_wan.py`
- Design docs:
  - `docs/action_conditioned_wan.md`
  - `docs/adaptor_design.md`
  - `docs/della_workflow.md`
- One-sample trainer: `scripts/train_action_conditioned_wan.py`
- DROID trainer: `scripts/train_action_conditioned_wan_droid.py`
- DROID Slurm wrapper: `run_action_conditioned_droid.sh`
- Overfit wrapper: `run_action_conditioned_overfit.sh`
- DROID cache planner/precompute:
  - `scripts/make_droid_window_plan.py`
  - `scripts/precompute_features_droid_plan.py`
  - `run_precompute_droid_window_plan_array.sh`
  - `scripts/submit_droid_train_cache_chunks.sh`
- Full-cache waiter:
  `scripts/wait_and_submit_action_droid_training.sh`

Tests for the adapter implementation are in `tests/test_action_conditioned_wan.py`.

## Last Live-Verified Cluster State

Last successful Della snapshot:
`2026-06-10 04:30:59 EDT` / `2026-06-10 01:30:59 PDT`.

Jobs:

- `9478714` `act-side-fresh25-10k`
  - partition/QOS: `ailab`
  - state: `RUNNING`
  - node: `della-i23g1`
  - run dir:
    `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5`
  - last fetched around step `4840/10000`, loss `0.0643683`
  - robust recent-25 loss stats were median `0.0624064`, clean mean `0.0625621`
  - occasional loss/grad spikes occurred but recovered immediately under grad
    clipping

- `9497851` `act-droid-cur-smoke`
  - partition/QOS: `ailab`
  - state: `PENDING`
  - reason: `QOSMaxJobsPerUserLimit`
  - output dir:
    `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_smoke`

- `9497852` `act-droid-cur-10k`
  - partition/QOS: `ailab`
  - state: `PENDING`
  - reason: `Dependency`
  - dependency: `afterok:9497851`
  - output dir:
    `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`

- `9497638` `droid-train-cache-s289-312`
  - partition/QOS: `gputest`
  - state at last check: array active; tasks `295-297` running and `298-312`
    pending on `JobArrayTaskLimit`

Cache counts at last live check:

- train: `417,594` windows
- val: `14,636` windows
- planned train target: `1,440,554`
- planned val target: `14,636`

Remote submitters last known active:

- cache submitter log:
  `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_121.log`
- full-cache training waiter log:
  `runs/droid_window_plans/action_droid_training_submitter_resume.log`

The current-cache DROID training snapshot manifest is:

`runs/droid_window_plans/train_current_cache_414672_20260610_042850.jsonl`

It contains `414,835` train names. It was created from train-cache directories
older than two minutes to avoid active cache writes. The matching summary is:

`runs/droid_window_plans/train_current_cache_414672_20260610_042850.summary.json`

## Local Artifacts

Fetched monitor artifacts are intentionally ignored by Git under `_cluster/`.
Current useful plots:

- `_cluster/loss_curves/side_fresh25_10k_loss_current.svg`
- `_cluster/loss_curves/droid_cache_progress_current.svg`
- `_cluster/loss_curves/current_monitor_summary.json`

Previous pilot random-eval videos:

`_cluster/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_pilot/videos_best_random_eval_s1000_1003/`

## Immediate Resume Commands

Start with normal SSH:

```bash
ssh della-gpu 'hostname && date'
```

Then inspect state:

```bash
ssh della-gpu 'cd /scratch/gpfs/AM43/lz3952/Wan2.2 && \
  git rev-parse HEAD && git status --short && \
  squeue -u $(whoami) -o "%.18i %.9P %.32j %.8T %.20M %.80R"'
```

Check cache counts:

```bash
ssh della-gpu 'cd /scratch/gpfs/AM43/lz3952/Wan2.2 && \
  printf "train=" && find data/droid_cache_windows_v0/train -mindepth 1 -maxdepth 1 -type d | wc -l && \
  printf "val=" && find data/droid_cache_windows_v0/val -mindepth 1 -maxdepth 1 -type d | wc -l'
```

Check overfit loss:

```bash
ssh della-gpu 'cd /scratch/gpfs/AM43/lz3952/Wan2.2 && \
  tail -20 runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5/train_log.csv'
```

Check current-cache smoke/full logs once jobs start:

```bash
ssh della-gpu 'cd /scratch/gpfs/AM43/lz3952/Wan2.2 && \
  tail -80 slurm_outputs/action-droid/out_act-droid-cur-smoke_9497851.log 2>/dev/null || true && \
  tail -80 slurm_outputs/action-droid/err_act-droid-cur-smoke_9497851.log 2>/dev/null || true && \
  tail -80 slurm_outputs/action-droid/out_act-droid-cur-10k_9497852.log 2>/dev/null || true && \
  tail -80 slurm_outputs/action-droid/err_act-droid-cur-10k_9497852.log 2>/dev/null || true'
```

Scan current errors:

```bash
ssh della-gpu 'cd /scratch/gpfs/AM43/lz3952/Wan2.2 && \
  grep -R "Traceback\\|RuntimeError\\|CUDA out of memory\\|failed\\|fail=[1-9]\\|planned cache shard incomplete" \
    -n slurm_outputs/droid-window-cache runs/droid_window_plans slurm_outputs/action-overfit slurm_outputs/action-droid 2>/dev/null | \
  grep -v "train_cache_chunk_submitter_gputest_1024.nohup.log" | tail -40 || true'
```

## Next Decisions

1. If `9478714` completed, inspect the final overfit curve and create/fetch
   random-noise eval videos. Record final video paths.
2. If `9497851` started, verify five smoke steps plus validation. If it fails,
   debug before allowing/keeping the full run. If it passes, monitor `9497852`.
3. If `9497851` remains pending on `QOSMaxJobsPerUserLimit`, decide whether to
   wait for overfit `9478714` to finish or cancel `9478714` to prioritize DROID
   dataset training.
4. Continue cache submitter monitoring. When the full planned cache completes,
   the existing waiter should submit the full-cache smoke and 10k runs.
5. Keep updating `WORKLOG.md` and commit/push any code or documentation changes.
