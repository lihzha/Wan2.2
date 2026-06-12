# Wan2.2 Della Handoff

Last updated: 2026-06-12 00:20 PDT.

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
direction is scaling side-adapter training with fresh noise.

## Current Status - 2026-06-11 20:26 PDT

- Active single-GPU full-cache batch-5 job `9541718` is still running on
  `della-i21g3` from the canonical Della checkout. Run dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5`.
- Step-4000 validation landed and improved to `0.21933504613116384`.
  `ckpt_latest.pt` was copied to `ckpt_step4000.pt`.
- Step-4000 eval job `9573559` completed cleanly (`COMPLETED 0:0`, elapsed
  `00:02:23`) with output dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step4000_ep399_v0_s00004_s1000_1001/`.
  Local artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step4000_ep399_v0_s00004_s1000_1001/`.
  Viz URL:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step4000_ep399_v0_s00004_s1000_1001`.
  Contact sheet:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step4000_ep399_v0_s00004_s1000_1001/droid_full_bs5_step4000_eval_contact_sheet.jpg`.
- Previous full-cache evals:
  - step 1000: seed1000 MSE `0.29633891582489014`, seed1001 MSE
    `0.2788466513156891`.
  - step 2000: seed1000 MSE `0.27191445231437683`, seed1001 MSE
    `0.2750750482082367`.
  - step 3000: seed1000 MSE `0.2574395537376404`, seed1001 MSE
    `0.26347997784614563`.
  - step 4000: seed1000 MSE `0.24452289938926697`, seed1001 MSE
    `0.24768376350402832`.
  - step 5000: seed1000 MSE `0.23440805077552795`, seed1001 MSE
    `0.23428887128829956`.
  Qualitatively, the scene remains much better than null, but the moving
  robot/gripper still turns gray/hazy/smeared after motion starts.
- Step-5000 eval job `9579605` completed cleanly (`COMPLETED 0:0`, elapsed
  `00:02:17`). Local artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step5000_ep399_v0_s00004_s1000_1001/`.
  Viz URL:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step5000_ep399_v0_s00004_s1000_1001`.
  Contact sheet:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step5000_ep399_v0_s00004_s1000_1001/droid_full_bs5_step5000_eval_contact_sheet.jpg`.
- Full 8-GPU DDP job `9565757` is dependency-released and pending on priority
  from isolated Della worktree
  `/scratch/gpfs/AM43/lz3952/worktrees/Wan2.2/codex-droid-ddp-8gpu-barrierfix`.
  Latest start estimate is `2026-06-12T14:49:54` Della time on
  `della-i20g2` with a 2-day time limit. Run dir:
  `runs/action_droid_dist_side_bn512h8_L0-29_fresh_25step_fullcache_10k_lr5e-5_bs5x8_ddp_7cb94a9`.
- The 8-GPU smoke job `9565756` passed: world size `8`, one optimizer step,
  validation/checkpoint writes, and no NCCL ambiguous-device barrier warning.
- Do not send a final answer while `9541718`, `9573559`, or `9565757` still
  need active monitoring, artifact inspection, or relaunch/debug handling.

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
  `9564218` `act-ddp2-smoke` completed cleanly (`COMPLETED 0:0`, elapsed
  `00:01:55`). It validated world size 2 DDP with one optimizer step and
  rank-0-only validation/checkpointing. The first 8-GPU smoke/full jobs
  (`9564219`, `9564220`) were canceled before start to patch a non-fatal NCCL
  barrier device warning. Replacement 8-GPU jobs should be launched from the
  new barrier-fix commit.
  Original chain was `9564218` -> `9564219` -> `9564220`.
  Replacement worktree:
  `/scratch/gpfs/AM43/lz3952/worktrees/Wan2.2/codex-droid-ddp-8gpu-barrierfix`
  at `7cb94a9671926d12f20b253d64ad37152795f577`.
  Replacement jobs: `9565756` `act-ddp8-smoke` ->
  `9565757` `act-ddp8-10k`. The 8-GPU smoke completed functionally:
  world size `8`, one optimizer step, val/checkpoint writes, no NCCL barrier
  warning after the barrier fix. The remaining grad-stride warning is
  performance-only. Full job `9565757` is dependency-released and pending with
  reason `(None)`; latest estimated start was `2026-06-12T06:31:24`.
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
- Full-cache batch-5 step-3000 validation improved again to
  `0.23995679058134556`. Step-3000 eval job `9566483` completed cleanly using
  `ckpt_step3000.pt`.
  Local artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step3000_ep399_v0_s00004_s1000_1001/`
  Viz URL:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step3000_ep399_v0_s00004_s1000_1001`
  Metrics: seed1000 latent MSE `0.2574395537376404` vs null
  `1.9798624515533447`; seed1001 latent MSE `0.26347997784614563` vs null
  `4.661829471588135`. Numerically improved from step 2000; qualitatively the
  scene layout remains much better than null, but the robot/gripper haze still
  remains after motion starts.

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
