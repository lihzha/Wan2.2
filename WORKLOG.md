# Wan2.2 Action Conditioning Worklog

This worklog follows `robotics-cluster-development-core`. It was created
retrospectively on 2026-06-09 from local code, fetched cluster artifacts under
`_cluster/`, and the active Della monitoring history. Della was down for
maintenance during creation, so final queue state after the last observed
snapshot is not live-verified here.

## Current Handoff State As Of 2026-06-10 13:41 PDT

Goal:
- Learn an action/I0-conditioned adapter for frozen Wan2.2 that can control
  generated videos without fine-tuning the backbone, with training/eval
  guidance scale aligned and with fresh initial diffusion noise.

Version control:
- local branch: `main`
- local/fork latest documented commit before handoff:
  `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff`
- Della remote checkout was last live-verified and deployed at the same commit
  at `2026-06-10 04:30:59 EDT` / `2026-06-10 01:30:59 PDT`.
- Final handoff refresh at `2026-06-10 13:41 PDT` could not live-verify Della
  because SSH auth failed from this Codex process:
  `tigressgateway` returned `Permission denied (publickey,keyboard-interactive)`
  and `della-gpu` returned `Connection closed by UNKNOWN port 65535`.

Current conclusions:
- Fixed-noise or 1-step overfit is too easy and does not diagnose the real
  25-step fresh-noise problem.
- Same-context, wrong-noise replay fails badly, confirming that the initial
  noise basin matters.
- Large-scale DDIM `z_init` samples do not show an obvious low-rank global
  geometric structure that is easy to predict directly.
- The side adapter is the strongest implemented adapter so far. It reuses Wan
  hidden states and timestep/noisy-latent/I0 processing and injects action
  through trainable residual branches.
- Side adapter can overfit one sample at 1 diffusion step very strongly. The
  active 25-step fresh-noise overfit run has occasional one-step loss/gradient
  spikes but immediately recovers; robust recent loss statistics remained near
  `0.062` in the last verified monitor window.
- The first DROID dataset training attempt using the full planned cache is
  waiting for cache completion via a remote waiter. A second DROID dataset
  training attempt using a fixed snapshot of the current partial cache has
  already been submitted and is waiting for the `ailab` slot.

Latest live-verified Della state:
- Last successful live check: `2026-06-10 04:30:59 EDT`
  / `2026-06-10 01:30:59 PDT`.
- Remote project path: `/scratch/gpfs/AM43/lz3952/Wan2.2`.
- Remote commit: `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff`.
- Overfit job `9478714`, `act-side-fresh25-10k`, was running on
  `della-i23g1` with elapsed `11:39:39`.
- Current-cache DROID smoke job `9497851`, `act-droid-cur-smoke`, was pending
  with reason `QOSMaxJobsPerUserLimit`.
- Current-cache DROID full job `9497852`, `act-droid-cur-10k`, was pending
  with reason `Dependency` on `afterok:9497851`.
- DROID train-cache array `9497638` for shards `289-312` was active on
  `gpu-test`; tasks `295-297` were running and `298-312` were pending on the
  array task limit.
- DROID window cache count was `417,594` train windows and `14,636` val
  windows.
- Current-cache DROID training snapshot manifest:
  `runs/droid_window_plans/train_current_cache_414672_20260610_042850.jsonl`
  with `414,835` train names.

Key artifact paths:
- Local fetched artifacts: `_cluster/`
- Local latest overfit/cache plots:
  - `_cluster/loss_curves/side_fresh25_10k_loss_current.svg`
  - `_cluster/loss_curves/droid_cache_progress_current.svg`
  - `_cluster/loss_curves/current_monitor_summary.json`
- Remote 25-step fresh-noise overfit:
  `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5`
- Remote current-cache DROID smoke:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_smoke`
- Remote current-cache DROID full:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`
- Remote cache submitter log:
  `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_121.log`
- Remote full-cache training waiter log:
  `runs/droid_window_plans/action_droid_training_submitter_resume.log`
- Random-eval videos for previous 25-step fresh-noise pilot:
  `_cluster/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_pilot/videos_best_random_eval_s1000_1003/`
- DROID planning scripts and trainers are in `scripts/` and Slurm wrappers in
  the repo root.

Immediate next checks for the next agent:
1. Refresh user SSH/auth first. Use normal `ssh della-gpu`, not
   `ssh -o BatchMode=yes della-gpu`, because the working config relies on
   keyboard-interactive auth and ControlMaster.
2. Check `squeue -u $(whoami)` and specifically jobs `9478714`, `9497851`,
   `9497852`, and the current `droid-train-cache-*` array.
3. If `9478714` completed, inspect its logs and metrics, then export/fetch
   random-noise eval videos for the best/latest checkpoint.
4. If `9497851` started, inspect its stdout/stderr and verify five smoke steps
   plus validation. If it passed, monitor `9497852`; if it failed, debug before
   allowing the full run.
5. Continue watching the train-cache submitter until the full planned cache is
   complete; then the existing waiter should submit the full-cache smoke and
   10k jobs.

## 2026-06-10 13:50 PDT - Resume After Handoff Pull

Goal:
- Resume the Della monitoring loop after pulling the new handoff commit and
  verify the active overfit/DROID/cache state before launching any new jobs.

Hypothesis:
- Della SSH/auth is now healthy, the 10k one-sample overfit likely completed
  cleanly, and the current-cache DROID run should be monitored before any
  further scale-up or export work.

Change:
- Pulled local `main` from `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff` to
  `4a28dc358074ffebff091df945df1104d1556d18`.
- Created local ignored `_cluster/` artifact folders and fetched small JSON,
  CSV, and Slurm log files for the completed overfit, current-cache smoke, and
  current-cache 10k run. Checkpoints and videos were not fetched.
- No code changes and no new Slurm jobs launched in this entry.

Version Control:
- branch: `main`
- base_commit: `4a28dc358074ffebff091df945df1104d1556d18`
- implementation_commit: pending worklog-only update
- push/pull: pulled local fork; Della checkout not pulled yet because the
  remote is actively running jobs and the new local commit is handoff/worklog
  documentation only.
- changed_files: `WORKLOG.md`; ignored fetched artifacts under `_cluster/`
- remote_commit/status: Della `/scratch/gpfs/AM43/lz3952/Wan2.2` remains at
  `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff` with untracked `6/` and
  `Wan2.2-TI2V-5B/`.

Command / Job:
- command: `git pull --ff-only`; `ssh della-gpu 'hostname && date'`; remote
  `squeue`, `sacct`, log tails, cache counts; `rsync` small run metadata/logs
  to `_cluster/`
- job_id: monitored existing jobs `9478714`, `9497851`, `9497852`,
  `9518561`
- run_dir:
  - `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5`
  - `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_smoke`
  - `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`
- logs:
  - `_cluster/slurm_outputs/action-overfit/out_act-side-fresh25-10k_9478714.log`
  - `_cluster/slurm_outputs/action-overfit/err_act-side-fresh25-10k_9478714.log`
  - `_cluster/slurm_outputs/action-droid/out_act-droid-cur-smoke_9497851.log`
  - `_cluster/slurm_outputs/action-droid/err_act-droid-cur-smoke_9497851.log`
  - `_cluster/slurm_outputs/action-droid/out_act-droid-cur-10k_9497852.log`
  - `_cluster/slurm_outputs/action-droid/err_act-droid-cur-10k_9497852.log`
- artifacts:
  - `_cluster/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5/{config.json,summary.json,train_log.csv}`
  - `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_smoke/{config.json,summary.json,train_log.csv,val_log.csv}`
  - `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/{config.json,train_log.csv,val_log.csv}`

Result:
- status: in_progress
- metrics/artifacts:
  - SSH succeeded: `della-gpu.princeton.edu`, `Wed Jun 10 04:46:37 PM EDT 2026`.
  - Overfit job `9478714` completed `0:0` after `23:32:23`; final log line
    reports `done best=0.040368 final=0.052908`.
  - Overfit CSV has `501` logged rows through step `10000`; logged-loss min
    `0.0405246`, last `0.0540686`, recent-25 mean `0.0538273`, recent-25
    median `0.0537409`. Summary best loss is `0.0403681`.
  - Smoke job `9497851` completed `0:0` and ran 5 train steps plus validation;
    summary best/latest val loss `4.7875304`.
  - Full current-cache DROID job `9497852` is running on `della-i23g3`; fetched
    logs were at step `3780`, with recent-25 train mean `0.230979`, recent-25
    median `0.218067`, and val losses `0.336866`, `0.265357`, `0.233172` at
    steps `1000`, `2000`, `3000`.
  - Cache array `9518561` is active on `gputest` shards `625-648`; live count
    was `894928/1440554` train windows and `14636/14636` val windows. Submitter
    log tail showed `886264/1440554` train at `2026-06-10 16:35:56` EDT.
  - Error scan only surfaced an older `9479259` DROID wrapper smoke OOM; no
    current `9497851`/`9497852` traceback was observed.
- key evidence: local `_cluster/` JSON/CSV/log files listed above, plus
  `sacct` showing `9478714` and `9497851` completed and `9497852` running.

Analysis:
- The 10k overfit improved substantially over the 100-step pilot and ended
  stably, but final fixed-seed eval is worse than the best train step, so
  random-noise video export is still required before judging qualitative
  control.
- The current-cache DROID run has passed the smoke gate and early validation is
  improving, making cancellation/debugging unnecessary at this point.
- The cache builder is making steady progress but the full planned cache is not
  complete, so the full-cache waiter should remain under observation.

Next:
- Submit the overfit random-noise export with `run_action_conditioned_export.sh`
  after user confirmation, then fetch and inspect MP4s plus `metrics.json`.
- Continue monitoring `9497852` through completion and inspect final train/val
  curves, summary/checkpoints, and any generated artifacts.
- Continue cache submitter monitoring until `1440554/1440554` train windows are
  present and the full-cache waiter submits its smoke/10k jobs.

## 2026-06-10 14:46 PDT - SSH Restored And DROID Monitor Resumed

Goal:
- Resume active monitoring after a transient SSH failure interrupted the Della
  loop.

Hypothesis:
- The failure was authentication/session infrastructure only; the active DROID
  10k training and cache precompute jobs likely continued on Della.

Change:
- Retried normal `ssh della-gpu` after the user fixed SSH.
- Refreshed queue state, remote git state, DROID train/val logs, Slurm stderr,
  cache submitter tail, and live cache counts.
- Fetched updated DROID 10k train log and action-droid Slurm logs into ignored
  local `_cluster/` paths.

Version Control:
- branch: `main`
- base_commit: `63752a13cc51d1d40e4a56a3af4693ceb5c0f6b4`
- implementation_commit: pending worklog-only update
- push/pull: local worklog commit `63752a1` was already pushed; Della remains
  at code commit `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff` because active
  jobs are running and the newer commits are monitoring documentation.
- changed_files: `WORKLOG.md`; ignored fetched artifacts under `_cluster/`
- remote_commit/status: `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff`, untracked
  `6/` and `Wan2.2-TI2V-5B/`.

Command / Job:
- command:
  `ssh della-gpu 'hostname && date && cd /scratch/gpfs/AM43/lz3952/Wan2.2 && ...'`;
  `rsync` DROID 10k CSV/log files; live cache count via `find ... | wc -l`
- job_id: monitored existing `9497852` and cache array `9520633`
- run_dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`
- logs:
  `_cluster/slurm_outputs/action-droid/out_act-droid-cur-10k_9497852.log`,
  `_cluster/slurm_outputs/action-droid/err_act-droid-cur-10k_9497852.log`
- artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/{train_log.csv,val_log.csv}`

Result:
- status: in_progress
- metrics/artifacts:
  - SSH succeeded at `Wed Jun 10 05:46:37 PM EDT 2026`.
  - `9497852` was still running on `della-i23g3` with elapsed `10:19:25`.
  - Latest fetched DROID train step was `4160`, loss `0.2305805`.
  - Recent 25 logged train losses: mean `0.251804`, median `0.225247`, max
    `0.604854`, min `0.115840`; recent 25 grad norm mean `5.19643`, max
    `31.1511`.
  - Validation losses: step `1000` `0.336866`, step `2000` `0.265357`,
    step `3000` `0.233172`, step `4000` `0.232768`.
  - Current stderr still only showed AMP deprecation warnings and checkpoint
    load progress; no traceback/OOM.
  - Cache array `9520633` for shards `649-672` was active. Live cache count:
    `934620/1440554` train windows and `14636/14636` val windows.
- key evidence: live SSH snapshot plus refreshed local `_cluster/` logs.

Analysis:
- The DROID run survived the SSH interruption and continues normally. The
  validation curve improved strongly through step 3000 and then plateaued
  between steps 3000 and 4000, so the next important signal is step 5000.
- The train curve remains noisy, with occasional spikes, but the median recent
  loss and grad norms do not indicate divergence.
- Cache generation is progressing steadily but is still short of the full
  planned train cache.

Next:
- Continue monitoring `9497852` until step 5000 validation and then reassess
  whether the plateau persists.
- Continue cache submitter monitoring until the full planned cache completes.
- Keep the overfit random-noise export pending until the user confirms launching
  that new Slurm job.

## 2026-06-10 16:54 PDT - Current-Cache DROID Step 5000 Validation

Goal:
- Check whether the current-cache DROID 10k run recovers from the apparent
  step-3000 to step-4000 validation plateau.

Hypothesis:
- If the plateau was only a temporary flat section, step-5000 validation should
  improve below the step-4000 value without train loss or gradients diverging.

Change:
- Continued read-only monitoring only.
- Fetched updated DROID 10k train/val CSVs and Slurm logs into ignored
  `_cluster/` paths.

Version Control:
- branch: `main`
- base_commit: `90d2b3de84b1b78f194802560758f191db265db3`
- implementation_commit: pending worklog-only update
- push/pull: local monitoring docs are pushed; Della still running active jobs
  on code commit `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff`.
- changed_files: `WORKLOG.md`; ignored fetched artifacts under `_cluster/`
- remote_commit/status: not re-pulled while active jobs are running.

Command / Job:
- command: `ssh della-gpu ... tail train_log.csv val_log.csv ...`; `rsync`
  DROID 10k CSV/log files
- job_id: monitored existing `9497852` and cache array `9526326`
- run_dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`
- logs:
  `_cluster/slurm_outputs/action-droid/out_act-droid-cur-10k_9497852.log`,
  `_cluster/slurm_outputs/action-droid/err_act-droid-cur-10k_9497852.log`
- artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/{train_log.csv,val_log.csv}`

Result:
- status: in_progress
- metrics/artifacts:
  - `9497852` was running on `della-i23g3` at `Wed Jun 10 07:54:45 PM EDT`
    with elapsed `12:27:33`.
  - Step `5000` train loss `0.201143`.
  - Validation losses: step `1000` `0.336866`, step `2000` `0.265357`,
    step `3000` `0.233172`, step `4000` `0.232768`, step `5000`
    `0.222377`.
  - Last 50 logged train losses: mean `0.238912`, median `0.223288`, min
    `0.126937`, max `0.553217`.
  - Last 50 grad norms: mean `9.33277`, max `153.893`. Recent isolated grad
    spikes did not coincide with loss divergence.
  - Current stderr still only showed AMP deprecation warnings and checkpoint
    load progress; no traceback/OOM.
  - Cache generation advanced to array `9526326` for shards `721-744`, with
    `721-723` running at the snapshot.
- key evidence: refreshed local train/val CSVs and Slurm logs under
  `_cluster/`.

Analysis:
- The validation plateau did not persist: step-5000 validation improved by
  about `0.01039` absolute from step 4000 and is the best validation so far.
- Train loss remains noisy due sample variability, with occasional gradient
  spikes, but there is no sustained upward trend, NaN, OOM, or traceback.
- The run should continue to the next validation checkpoint. Current-cache
  training is producing a real signal and should not be canceled.

Next:
- Continue monitoring `9497852` through step `6000` validation.
- Continue cache monitoring; full planned cache is still incomplete.
- After `9497852` completes, inspect final train/val curves and checkpoints.

## 2026-06-10 17:52 PDT - Launch Overfit And DROID Eval Exports

Goal:
- Export qualitative random-noise videos for the completed 10k overfit run and
  for the current DROID best-val checkpoint, then fetch and inspect the videos
  locally.

Hypothesis:
- The overfit checkpoint's low latent MSE should produce visibly closer samples
  than the null baseline under random eval noise.
- The DROID step-5000 `ckpt_best_val.pt` should show whether the improving
  validation loss corresponds to usable qualitative control on a held-out val
  window.

Change:
- User approved eval launches going forward without further per-job
  confirmation.
- Submitted two eval/export Slurm jobs on `ailab`; no code changes.
- Checked local `viz_open` availability; it is not currently on this shell
  `PATH`, so if it remains unavailable after fetch, inspect representative
  frames with the available local image viewer and report exact local video
  paths.

Version Control:
- branch: `main`
- base_commit: `29888ffb86994ea36ceeb0c3fa2173ac2e2ff9f2`
- implementation_commit: pending worklog-only update
- push/pull: no source deployment; remote active jobs remain on
  `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff`
- changed_files: `WORKLOG.md`
- remote_commit/status: not re-pulled while active jobs are running.

Command / Job:
- command:
  `sbatch --parsable --time=02:00:00 --job-name=act-overfit-eval-10k --export=ALL,RUN_DIR=runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5,CKPT_NAME=ckpt_best.pt,OUTPUT_DIR=runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5/videos_best_random_eval_s1000_1003,TRIPLETS_ROOT=data/droid_cache/train,INCLUDE_NULL=1,EVAL_NOISE_MODE=random,NUM_EVAL_NOISES=4,EVAL_SEED_START=1000 run_action_conditioned_export.sh`
- job_id: `9527755`
- run_dir:
  `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5`
- logs:
  `slurm_outputs/action-overfit/out_act-overfit-eval-10k_9527755.log`,
  `slurm_outputs/action-overfit/err_act-overfit-eval-10k_9527755.log`
- artifacts:
  `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5/videos_best_random_eval_s1000_1003/`

Command / Job:
- command:
  `sbatch --parsable --job-name=act-droid-eval-s5000 --output=slurm_outputs/action-droid/out_%x_%j.log --error=slurm_outputs/action-droid/err_%x_%j.log --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=180G --gres=gpu:1 --time=02:00:00 --account=am43 --partition=ailab --qos=ailab --wrap="<export_action_conditioned_wan_video.py ... --ckpt_path runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/ckpt_best_val.pt --triplets_root data/droid_cache_windows_v0/val --overfit_one ep399_v0_s00004 --eval_noise_mode random --eval_seed_start 1000 --num_eval_noises 2 --include_null>"`
- job_id: `9527756`
- run_dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`
- logs:
  `slurm_outputs/action-droid/out_act-droid-eval-s5000_9527756.log`,
  `slurm_outputs/action-droid/err_act-droid-eval-s5000_9527756.log`
- artifacts:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_bestval_step5000_ep399_v0_s00004_s1000_1001/`

Result:
- status: in_progress
- metrics/artifacts:
  - `9527755` and `9527756` submitted successfully.
  - Initial `squeue` showed both eval jobs pending on `ailab` while
    `9497852` continued running; no eval artifacts produced yet.
  - `9497852` was running at the launch check with elapsed `13:24:11`.
- key evidence: `squeue`/`sacct` snapshot after submission.

Analysis:
- Eval exports are queued behind the active training allocation as expected for
  the proven `ailab` execution surface. This avoids interfering with current
  training and still records the requested eval jobs.
- DROID export uses validation sample `ep399_v0_s00004`, which is present in
  the val window manifest and has a nontrivial action score.

Next:
- Monitor `9527755` and `9527756`; when they complete, fetch MP4s/metrics into
  `_cluster/`, inspect metrics and videos, and use `viz_open` if available or
  representative-frame inspection otherwise.
- Continue periodic DROID training eval launches at future validation
  checkpoints without asking for additional permission.

## 2026-06-10 18:00 PDT - Eval Videos Fetched And Inspected

Goal:
- Validate the overfit and DROID step-5000 eval videos, not just the eval
  Slurm exit codes.

Hypothesis:
- The overfit eval should be visually close to ground truth and much better
  than null-only generation. The DROID eval may be weaker but should still show
  a meaningful qualitative gap versus null if the step-5000 validation signal
  is real.

Change:
- Fetched MP4s, `metrics.json`, and eval Slurm logs for jobs `9527755` and
  `9527756` into ignored local `_cluster/` directories.
- Generated local contact-sheet JPEGs from first/middle/last video frames.
- `viz_open` was not available as a local shell command and no exposed tool
  matched it, so visual inspection used local contact sheets via the available
  image viewer.

Version Control:
- branch: `main`
- base_commit: `e8bc3f1beb57c1c48059f7e4afb8876eb2528b67`
- implementation_commit: pending worklog-only update
- push/pull: no source deployment; artifact fetch only
- changed_files: `WORKLOG.md`; ignored fetched artifacts under `_cluster/`
- remote_commit/status: not re-pulled while active jobs are running.

Command / Job:
- command: monitored `sacct`/logs for `9527755`, then `rsync` fetched
  `*.mp4`, `metrics.json`, and Slurm logs; generated contact sheet with
  `ffmpeg` frame extraction and PIL composition.
- job_id: `9527755`
- run_dir:
  `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5`
- logs:
  `_cluster/slurm_outputs/action-overfit/out_act-overfit-eval-10k_9527755.log`,
  `_cluster/slurm_outputs/action-overfit/err_act-overfit-eval-10k_9527755.log`
- artifacts:
  `_cluster/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5/videos_best_random_eval_s1000_1003/`
  including `ground_truth.mp4`, `sample_seed1000..1003.mp4`,
  `null_only_seed1000..1003.mp4`, `metrics.json`, and
  `overfit_eval_contact_sheet.jpg`.

Result:
- status: passed
- metrics/artifacts:
  - Job `9527755` completed `0:0` in `00:02:19`.
  - Overfit eval latent MSEs:
    - seed1000 `0.0416005` vs null `2.53899`
    - seed1001 `0.0405720` vs null `4.93637`
    - seed1002 `0.0413762` vs null `2.49327`
    - seed1003 `0.0404059` vs null `3.78863`
  - All MP4s are `320x192`, `33` frames, `16 fps`, duration `2.0625s`.
- key evidence: local `metrics.json`, MP4s, and contact sheet.

Analysis:
- The overfit videos are qualitatively strong: the action-conditioned samples
  remain close to the sink/bottle scene across random seeds, while null-only
  generations quickly collapse into severe color/geometric artifacts.
- This supports that the side adapter overfit checkpoint learned meaningful
  control for the one-sample case under fresh random noise.

Command / Job:
- command: monitored `sacct`/logs for `9527756`, then `rsync` fetched
  `*.mp4`, `metrics.json`, and Slurm logs; generated contact sheet with
  `ffmpeg` frame extraction and PIL composition.
- job_id: `9527756`
- run_dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`
- logs:
  `_cluster/slurm_outputs/action-droid/out_act-droid-eval-s5000_9527756.log`,
  `_cluster/slurm_outputs/action-droid/err_act-droid-eval-s5000_9527756.log`
- artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_bestval_step5000_ep399_v0_s00004_s1000_1001/`
  including `ground_truth.mp4`, `sample_seed1000..1001.mp4`,
  `null_only_seed1000..1001.mp4`, `metrics.json`, and
  `droid_step5000_eval_contact_sheet.jpg`.

Result:
- status: passed
- metrics/artifacts:
  - Job `9527756` completed `0:0` in `00:02:01`.
  - Held-out sample: `ep399_v0_s00004`.
  - DROID eval latent MSEs:
    - seed1000 `0.251466` vs null `1.97986`
    - seed1001 `0.252762` vs null `4.66183`
  - All MP4s are `320x192`, `33` frames, `16 fps`, duration `2.0625s`.
- key evidence: local `metrics.json`, MP4s, and contact sheet.

Analysis:
- The DROID step-5000 eval is meaningfully better than null, but still
  qualitatively imperfect. It preserves the table/object layout much better
  than null-only generation, but middle/late frames become cloudy and do not
  cleanly reproduce the robot arm motion.
- This is consistent with the current validation loss: the dataset run is
  learning a real signal, but it has not reached overfit-level control quality.

Next:
- Continue monitoring DROID training through step `6000` validation.
- Launch another DROID eval at the next validation checkpoint without asking
  again, then compare videos and latent MSE against the step-5000 eval.
- Continue cache monitoring; full train cache is still incomplete.

## 2026-06-10 19:32 PDT - DROID Step 6000 Eval And Periodic Watcher

Goal:
- Launch and validate the next periodic eval for the current-cache DROID run
  after step `6000`, then arm future evals without requiring per-job user
  confirmation.

Hypothesis:
- If the validation improvement at step `6000` is real, the current checkpoint
  should improve held-out random-noise latent MSE versus the step-5000 eval and
  remain clearly better than the null-context baseline.

Change:
- Added a remote login-side watcher for step `6000`; it waited for
  `val_log.csv` to contain the step-6000 row and for `ckpt_latest.pt` to update
  before submitting the eval.
- After the step-6000 eval completed and was inspected locally, launched a
  second remote watcher for future periodic evals at steps `7000`, `8000`,
  `9000`, and `10000`.
- No source code changes. Fetched eval videos, metrics, train/val logs, and
  Slurm logs into ignored local `_cluster/` paths. Opened fetched eval
  directories with local `viz-open`.

Version Control:
- branch: `main`
- base_commit: `e2f6cff6f04c72d0b94981d396386c46b086c671`
- implementation_commit: pending worklog-only update
- push/pull: no source deployment; remote active jobs remain on
  `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff`
- changed_files: `WORKLOG.md`; ignored fetched artifacts under `_cluster/`
- remote_commit/status: not re-pulled while active jobs are running.

Command / Job:
- command: remote watcher
  `runs/droid_window_plans/eval_watcher_droid_step6000_20260610_214040.log`
  submitted
  `sbatch --job-name=act-droid-eval-s6000 ... export_action_conditioned_wan_video.py --ckpt_path .../ckpt_latest.pt --triplets_root data/droid_cache_windows_v0/val --overfit_one ep399_v0_s00004 --eval_noise_mode random --eval_seed_start 1000 --num_eval_noises 2 --include_null`
- job_id: `9529887`
- run_dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`
- logs:
  `_cluster/slurm_outputs/action-droid/out_act-droid-eval-s6000_9529887.log`,
  `_cluster/slurm_outputs/action-droid/err_act-droid-eval-s6000_9529887.log`
- artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step6000_ep399_v0_s00004_s1000_1001/`
  including `ground_truth.mp4`, `sample_seed1000..1001.mp4`,
  `null_only_seed1000..1001.mp4`, `metrics.json`, and
  `droid_step6000_eval_contact_sheet.jpg`.

Result:
- status: passed, active monitoring continues
- metrics/artifacts:
  - Training job `9497852` remained running after step `6000`.
  - Validation improved to `0.20653978269547224` at step `6000`, from
    `0.2223765980452299` at step `5000`; this is the new best validation loss.
  - Step `6000` train loss was `0.243960`.
  - Job `9529887` completed `0:0` in `00:03:34`.
  - DROID step-6000 eval latent MSEs:
    - seed1000 `0.237017` vs null `1.97986`
    - seed1001 `0.237107` vs null `4.66183`
  - All step-6000 MP4s are `320x192`, `33` frames, `16 fps`, duration
    `2.0625s`.
  - `viz-open` URL:
    `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step6000_ep399_v0_s00004_s1000_1001`
  - Future periodic watcher PID `1953264` is running and waiting for step
    `7000`; log:
    `runs/droid_window_plans/periodic_eval_watcher_droid_current_20260610_223224.log`.
- key evidence: local `metrics.json`, fetched MP4s, Slurm logs, and contact
  sheet under `_cluster/`.

Analysis:
- The step-6000 eval is a measurable improvement over step 5000 on the same
  held-out sample and random seeds: latent MSE dropped from about `0.252` to
  about `0.237`.
- Visual quality also remains much better than null-only generation. The
  samples preserve the table/object layout, while null runs collapse into
  saturated geometric artifacts. The main remaining qualitative failure is that
  middle/late frames are still cloudy and robot arm motion is not cleanly
  reproduced.
- The current-cache DROID run is still learning useful signal and should
  continue to the next validation checkpoints.

Next:
- Monitor periodic eval watcher `1953264`; fetch, validate, and visualize each
  new eval output at steps `7000`, `8000`, `9000`, and `10000`.
- Continue monitoring DROID training job `9497852` and the cache submitter /
  full-cache waiter.

## 2026-06-10 21:57 PDT - DROID Step 7000 Eval

Goal:
- Validate the periodic step-7000 eval from the current-cache DROID training
  run and compare it against step `6000`.

Hypothesis:
- Even if step-7000 validation is not a new best, `ckpt_latest.pt` may still
  improve qualitative or held-out random-noise behavior on the fixed eval
  sample.

Change:
- Periodic watcher `1953264` detected step `7000`, waited for
  `ckpt_latest.pt` to update, submitted eval job `9534318`, and then moved on
  to wait for step `8000`.
- Fetched MP4s, `metrics.json`, updated train/val logs, and Slurm logs into
  ignored local `_cluster/` paths.
- Generated `droid_step7000_eval_contact_sheet.jpg` and opened the local eval
  directory with `viz-open`.

Version Control:
- branch: `main`
- base_commit: `6338aec4a92290c2e29d87e2f7c66f24e413301a`
- implementation_commit: pending worklog-only update
- push/pull: no source deployment; remote active jobs remain on
  `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff`
- changed_files: `WORKLOG.md`; ignored fetched artifacts under `_cluster/`
- remote_commit/status: not re-pulled while active jobs are running.

Command / Job:
- command: periodic watcher submitted
  `sbatch --job-name=act-droid-eval-s7000 ... export_action_conditioned_wan_video.py --ckpt_path .../ckpt_latest.pt --triplets_root data/droid_cache_windows_v0/val --overfit_one ep399_v0_s00004 --eval_noise_mode random --eval_seed_start 1000 --num_eval_noises 2 --include_null`
- job_id: `9534318`
- run_dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`
- logs:
  `_cluster/slurm_outputs/action-droid/out_act-droid-eval-s7000_9534318.log`,
  `_cluster/slurm_outputs/action-droid/err_act-droid-eval-s7000_9534318.log`
- artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step7000_ep399_v0_s00004_s1000_1001/`
  including `ground_truth.mp4`, `sample_seed1000..1001.mp4`,
  `null_only_seed1000..1001.mp4`, `metrics.json`, and
  `droid_step7000_eval_contact_sheet.jpg`.

Result:
- status: passed, active monitoring continues
- metrics/artifacts:
  - Training job `9497852` was still running and had reached step `7020` at the
    post-eval snapshot.
  - Validation at step `7000` was `0.2071253014728427`, slightly worse than the
    step-6000 best `0.20653978269547224` but still better than step `5000`.
  - Step `7000` train loss was `0.180847`.
  - Job `9534318` completed `0:0` in `00:02:05`.
  - DROID step-7000 eval latent MSEs:
    - seed1000 `0.231918` vs null `1.97986`
    - seed1001 `0.236279` vs null `4.66183`
  - All step-7000 MP4s are `320x192`, `33` frames, `16 fps`, duration
    `2.0625s`.
  - `viz-open` URL:
    `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step7000_ep399_v0_s00004_s1000_1001`
  - Periodic watcher `1953264` is alive and waiting for step `8000`.
  - Cache waiter latest count at the post-eval snapshot:
    `1166216/1440554` train and `14636/14636` val.
- key evidence: local `metrics.json`, fetched MP4s, Slurm logs, contact sheet,
  and remote watcher log.

Analysis:
- Step-7000 validation did not beat step 6000, but the held-out random-noise
  eval did improve slightly: seed1000 improved from `0.237017` to `0.231918`,
  and seed1001 improved from `0.237107` to `0.236279`.
- Visual quality remains similar to step 6000: samples preserve the tabletop
  scene and are far better than null-only outputs, but late frames remain cloudy
  and robot-arm motion is still not cleanly reproduced.
- Continuing training is still justified; there is no divergence, OOM, or
  traceback, and eval quality has not regressed.

Next:
- Continue monitoring training job `9497852` through step `8000`; periodic
  watcher `1953264` should submit the next eval automatically.
- Fetch, validate, visualize, and record the step-8000 eval once it completes.
- Continue cache/full-cache waiter monitoring.

## 2026-06-10 23:22 PDT - Cache Submitter Resume From Shard 865

Goal:
- Recover the DROID train-cache materialization after the previous chunk
  submitter stopped at shard range `841-864`.

Hypothesis:
- The chunk `841-864` actually completed successfully, and the failure was a
  Slurm accounting race where `sacct` still reported two array tasks as
  `RUNNING` after `squeue` emptied.

Change:
- Inspected `squeue`, `sacct`, `scontrol show job`, and per-task logs for
  array `9534921`.
- Confirmed shards `841-864` all wrote `done ok=... fail=0`; `scontrol` showed
  tasks `9534921_861` and `9534921_864` as `COMPLETED` with `ExitCode=0:0`
  even though `sacct` still had stale `RUNNING` rows.
- Launched a run-local robust cache submitter from shard `865` that accepts
  stale `sacct RUNNING` rows only when `scontrol` reports the task completed
  cleanly. No tracked source code changes.

Version Control:
- branch: `main`
- base_commit: `f4bb71c224b39441943ff5d28d4d0a13206bdddd`
- implementation_commit: pending worklog-only update
- push/pull: no source deployment; recovery submitter is a remote run-local
  orchestration process
- changed_files: `WORKLOG.md`
- remote_commit/status: tracked remote checkout not changed.

Command / Job:
- command: remote `nohup bash` robust submitter with `START_SHARD=865`,
  `TOTAL_SHARDS=1024`, `CHUNK_SIZE=24`, `CONCURRENCY=3`, `--qos=gpu-test`,
  `TIME_LIMIT=00:10:00`, and `run_precompute_droid_window_plan_array_anygpu.sh`
- submitter_pid: `3404408`
- submitter_log:
  `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_865_robust_20260611_022200.log`
- first resumed array job: `9536297`
- first resumed shard range: `865-888`

Result:
- status: in_progress
- metrics/artifacts:
  - Original array `9534921` for shards `841-864` completed according to
    `scontrol` and per-shard output logs.
  - Previous waiter count before recovery was `1216132/1440554` train windows
    and `14636/14636` val windows.
  - New robust submitter launched array `9536297` for shards `865-888`, pending
    initially on `gpu-test`.
- key evidence: remote `scontrol show job 9534921_861`,
  `scontrol show job 9534921_864`, shard output logs, and robust submitter log.

Analysis:
- The original submitter's strict `sacct` check is vulnerable to stale
  accounting immediately after array completion. Since all suspicious shard logs
  ended with `fail=0`, restarting from `865` avoids duplicate work and does not
  skip any planned shard.
- The robust submitter is intentionally run-local rather than a tracked source
  patch so active training/eval code is not mutated underneath running jobs.

Next:
- Monitor robust cache submitter `3404408` and array `9536297`; ensure it
  continues through remaining shard ranges `865-1023`.
- Keep the full-cache waiter running; it should only submit full-cache training
  after the train count reaches `1440554/1440554`.

## 2026-06-11 00:27 PDT - DROID Step 8000 Eval

Goal:
- Validate the periodic step-8000 eval and compare it to the prior DROID evals
  at steps `5000`, `6000`, and `7000`.

Hypothesis:
- If the new best validation at step `8000` reflects improved control, the
  fixed held-out random-noise eval should improve versus step `7000`.

Change:
- Periodic watcher `1953264` detected step `8000`, waited for
  `ckpt_latest.pt` to update, submitted eval job `9537162`, and then moved on
  to wait for step `9000`.
- Fetched MP4s, `metrics.json`, updated train/val logs, and Slurm logs into
  ignored local `_cluster/` paths.
- Generated `droid_step8000_eval_contact_sheet.jpg` and opened the local eval
  directory with `viz-open`.

Version Control:
- branch: `main`
- base_commit: `b0934fd526bd2d77c300f1bfd846e3a0378e7a72`
- implementation_commit: pending worklog-only update
- push/pull: no source deployment; remote active jobs remain on
  `79554b590d0579be0d2dbe94bd0e74dc1ea5f7ff`
- changed_files: `WORKLOG.md`; ignored fetched artifacts under `_cluster/`
- remote_commit/status: not re-pulled while active jobs are running.

Command / Job:
- command: periodic watcher submitted
  `sbatch --job-name=act-droid-eval-s8000 ... export_action_conditioned_wan_video.py --ckpt_path .../ckpt_latest.pt --triplets_root data/droid_cache_windows_v0/val --overfit_one ep399_v0_s00004 --eval_noise_mode random --eval_seed_start 1000 --num_eval_noises 2 --include_null`
- job_id: `9537162`
- run_dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`
- logs:
  `_cluster/slurm_outputs/action-droid/out_act-droid-eval-s8000_9537162.log`,
  `_cluster/slurm_outputs/action-droid/err_act-droid-eval-s8000_9537162.log`
- artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step8000_ep399_v0_s00004_s1000_1001/`
  including `ground_truth.mp4`, `sample_seed1000..1001.mp4`,
  `null_only_seed1000..1001.mp4`, `metrics.json`, and
  `droid_step8000_eval_contact_sheet.jpg`.

Result:
- status: passed, active monitoring continues
- metrics/artifacts:
  - Training job `9497852` was still running and had reached step `8020` at the
    post-eval snapshot.
  - Validation at step `8000` was `0.20188114559277892`, a new best compared
    with step `6000` `0.20653978269547224` and step `7000`
    `0.2071253014728427`.
  - Step `8000` train loss was `0.165461`.
  - Job `9537162` completed `0:0` in `00:02:11`.
  - DROID step-8000 eval latent MSEs:
    - seed1000 `0.224116` vs null `1.97986`
    - seed1001 `0.222946` vs null `4.66183`
  - All step-8000 MP4s are `320x192`, `33` frames, `16 fps`, duration
    `2.0625s`.
  - `viz-open` URL:
    `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step8000_ep399_v0_s00004_s1000_1001`
  - Periodic watcher `1953264` is alive and waiting for step `9000`.
  - Robust cache submitter `3404408` is alive on array `9536919`, shards
    `889-912`.
  - Cache waiter latest count at the post-eval snapshot:
    `1250432/1440554` train and `14636/14636` val.
- key evidence: local `metrics.json`, fetched MP4s, Slurm logs, contact sheet,
  and remote watcher/cache logs.

Analysis:
- Step-8000 is the best DROID checkpoint so far by both validation loss and
  fixed held-out random-noise eval MSE.
- Eval MSE improved materially from step 7000: seed1000 `0.231918 -> 0.224116`,
  seed1001 `0.236279 -> 0.222946`.
- Visual quality still has the same limitation: samples preserve the table and
  object layout and remain far better than null-only outputs, but middle/late
  frames are cloudy and robot-arm motion is not cleanly reconstructed.
- The current-cache run is still improving and should continue to steps `9000`
  and `10000`.

Next:
- Continue monitoring training job `9497852`; periodic watcher `1953264` should
  submit step-9000 and step-10000 evals automatically.
- Fetch, validate, visualize, and record the step-9000 eval once it completes.
- Continue robust cache/full-cache waiter monitoring.

## 2026-06-08 - Initial Della Development Loop

Goal:
- Establish local-development / Della-execution workflow for
  `/scratch/gpfs/AM43/lz3952/Wan2.2`.

Hypothesis:
- Local edits plus targeted `rsync`, Slurm wrapper launches, fetched logs, and
  iterative debugging are sufficient for Wan adapter experiments.

Change:
- Synced remote code to local `~/code/Wan2.2`, excluding large model/data/run
  outputs.
- Created/used Della helper workflow scripts such as `scripts/della_loop.sh`.

Command / Job:
- command: `rsync` remote scratch project to local with large-output excludes.
- job_id: n/a
- run_dir: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs: Della Slurm logs under `slurm_outputs/`
- artifacts: local `_cluster/`

Result:
- status: passed
- metrics/artifacts: code synced locally; later jobs/logs/artifacts fetched
  into `_cluster/`.
- key evidence: local repo contains synced scripts, docs, fetched summaries,
  loss logs, and videos.

Analysis:
- Della workflow is viable when cluster SSH is up. `della-gpu` access depends
  on `tigressgateway`; during maintenance, the gateway may be reachable while
  Della returns `No route to host`.

Next:
- Continue using local code as source of truth and Della scratch as run target.

## 2026-06-08 - Initial Inverted Noise Artifact Probe

Goal:
- Inspect existing inverted initial noise artifacts for same/different sample
  structure.

Hypothesis:
- If `z_init` basins are geometrically structured, same-sample and
  different-sample pair statistics or PCA should show separable structure.

Change:
- Added/fetched analysis for existing inversion artifacts.

Command / Job:
- command: `run_zinit_probe_droid.sh`
- job_id: `9454356`
- run_dir: remote `runs/zinit_probe_droid`
- logs: `_cluster/slurm_outputs/zinit-probe/out_log_zinit-probe_9454356.out`
- artifacts: `_cluster/zinit_probe_droid/analysis/summary.json`

Result:
- status: passed
- metrics/artifacts:
  - `n_artifacts=115`
  - main latent shape `48x9x12x20`: count `64`, std mean `0.8406`,
    excess kurtosis mean `0.4459`
  - larger shape `48x9x22x40`: count `51`, std mean `0.8079`
  - different-sample cosine mean `0.0100`, RMSE mean `1.165`
  - same-sample duplicates in this deterministic artifact set were identical
    with cosine ~`1.0`
- key evidence: `_cluster/zinit_probe_droid/analysis/summary.json`

Analysis:
- Existing deterministic artifacts did not reveal a useful predictive manifold;
  same-sample identity was expected for deterministic DDIM reuse and not the
  distribution needed for fresh-noise training.

Next:
- Run basin replay and stochastic / multi-sample inversion probes.

## 2026-06-08 - Noise Basin Replay

Goal:
- Test whether a context optimized for one inverted initial noise generalizes
  when the initial noise is fresh or comes from another sample.

Hypothesis:
- If the action/context adapter is learning a basin-specific control input,
  replaying with out-of-basin initial noise should cause much higher latent MSE.

Change:
- Ran basin replay with own `z_init`, fresh Gaussian seeds, and cross-sample
  inverted `z_init`.

Command / Job:
- command: `run_zinit_basin_replay.sh`
- job_id: `9454357`
- run_dir: remote `runs/zinit_basin_eval`
- logs: `_cluster/slurm_outputs/zinit-basin/out_log_zinit-basin_9454357.out`
- artifacts: `_cluster/zinit_basin_eval/basin_replay.csv`

Result:
- status: passed
- metrics/artifacts:
  - `ep0_v0` own latent MSE `0.01484`; fresh seed MSEs `2.68475`,
    `2.83207`; cross-sample MSEs roughly `0.96684` to `1.76353`.
  - `ep1_v0` own latent MSE `0.02237`; fresh seed MSEs `2.05348`,
    `3.09344`; cross-sample MSEs roughly `0.90785` to `0.95509`.
  - `ep2_v0` own latent MSE `0.02022`; fresh seed MSEs `3.34050`,
    `3.34334`; cross-sample MSEs roughly `0.82874` to `1.05127`.
- key evidence: `_cluster/zinit_basin_eval/basin_replay.csv`

Analysis:
- The result strongly supports the basin hypothesis: correct context plus wrong
  fresh noise is much worse than own inverted noise. Cross-sample basins can be
  somewhat less catastrophic than pure fresh Gaussian but remain far from own
  inversion.

Next:
- Inspect whether non-deterministic inversion creates a learnable
  same-sample distribution.

## 2026-06-08 - Stochastic Inversion Probe For One Sample

Goal:
- Run non-deterministic inversion for the same sample to mimic random sampling
  behavior and inspect endpoint distribution.

Hypothesis:
- Eta-noisy inversion might produce a same-sample cloud that reveals basin
  structure better than deterministic DDIM.

Change:
- Ran seeds `0..7`, etas `0.0`, `0.3`, `0.6`; optimized reconstructions for
  eta `0.6`.

Command / Job:
- command: `run_stochastic_inversion_droid.sh`
- job_id: `9456094`
- run_dir: remote `runs/stochastic_zinit_droid`
- logs: `_cluster/slurm_outputs/stoch-inv/out_log_stoch-inv_9456094.out`
- artifacts: `_cluster/stochastic_zinit_droid/summary.json`,
  endpoint/reconstruction CSVs.

Result:
- status: passed
- metrics/artifacts:
  - `episodes=[0]`, `seeds=0..7`, `etas=[0.0,0.3,0.6]`
  - `n_endpoints=24`, `n_reconstructions=8`
- key evidence: `_cluster/stochastic_zinit_droid/summary.json`

Analysis:
- This confirmed the pipeline worked, but one episode was too small to justify
  a strong geometry conclusion.

Next:
- Expand to multiple DROID samples and then a 10k DDIM dataset.

## 2026-06-08 - Multi-Sample Stochastic Z-Init Pilot

Goal:
- Compare same-sample stochastic endpoints against different-sample endpoints.

Hypothesis:
- Same-sample non-deterministic inversions should be closer to each other than
  different-sample endpoints if a basin distribution exists.

Change:
- Ran DROID pilot on episodes `0..3`, seeds `0..3`, eta `0.3`, then analyzed
  a retained subset.

Command / Job:
- command: `run_zinit_dataset_pilot_droid.sh`
- job_id: `9464818`
- run_dir: remote `runs/zinit_dataset_pilot_droid`
- logs: `_cluster/slurm_outputs/zinit-data/out_log_zinit-data_9464818.out`
- artifacts: `_cluster/zinit_dataset_pilot_droid/summary.json`,
  `_cluster/zinit_dataset_pilot_droid/analysis/summary.json`

Result:
- status: passed
- metrics/artifacts:
  - endpoint generation: `n_endpoints=16`, `n_reconstructions=16`
  - retained analysis records: `10`
  - eta `0.3` std mean `0.9649`
  - same-sample same-eta RMSE mean `0.7128`, cosine mean `0.7250`
  - different-sample RMSE mean `1.3007`, cosine mean `0.0892`
  - PCA on 10 records: PC1 cumulative `0.365`, PC5 cumulative `0.960`
- key evidence: `_cluster/zinit_dataset_pilot_droid/analysis/summary.json`

Analysis:
- Same-sample stochastic inversions were much closer than different-sample
  inversions. However, the sample count was very small; high PCA explained
  variance is not reliable at `n=10`.

Next:
- Run large deterministic DDIM `z_init` collection to test for global
  structure across many samples.

## 2026-06-08 - 10k DDIM Z-Init Dataset And Analysis

Goal:
- Generate `10,000` deterministic DDIM `z_init` samples for uniformly sampled
  DROID videos/frames and analyze whether global geometry is easy to predict.

Hypothesis:
- If there is a clear low-dimensional structure, PCA on thousands of DDIM
  endpoints should show meaningful concentration and statistics should reveal
  non-random structure.

Change:
- Added and ran 10k z-init generation/analysis scripts:
  `run_zinit_10k_droid_array.sh`, `run_zinit_10k_analyze.sh`.

Command / Job:
- command: `run_zinit_10k_droid_array.sh`; analysis via
  `run_zinit_10k_analyze.sh`
- job_id: array `9466112`; analysis `9466137`
- run_dir: remote `runs/zinit_10k_ddim_droid`
- logs: `_cluster/slurm_outputs/zinit10k/`, `_cluster/slurm_outputs/zinit10k-an/`
- artifacts: `_cluster/zinit_10k_ddim_droid/analysis/summary.json`

Result:
- status: passed
- metrics/artifacts:
  - `n_records=10000`; PCA subset `2000`
  - endpoint mean mean `-0.00070`, std `0.00416`
  - endpoint std mean `0.83423`, std `0.02309`, q05 `0.79825`,
    q50 `0.83507`, q95 `0.86771`
  - q0.001 mean `-2.835`; q0.999 mean `2.849`
  - PCA cumulative explained variance: PC1 `0.00261`, PC10 `0.01832`,
    PC50 `0.06155`, PC100 `0.10451`
- key evidence: `_cluster/zinit_10k_ddim_droid/analysis/summary.json`

Analysis:
- Large DDIM endpoints look high-dimensional; PCA does not show an obvious
  compact global structure. This weakens the plan of predicting low-level
  `z_init` structure directly from action/I0.

Next:
- Shift focus from explicit `z_init` prediction to scaling adapter training
  with fresh initial noise and 25 diffusion steps.

## 2026-06-08 - Action-Conditioned Wan Adapter Design

Goal:
- Redesign the adapter so it is noise/timestep/I0 aware without training the
  Wan backbone.

Hypothesis:
- Reusing frozen Wan representations is better than predicting context from
  action/I0 alone because Wan already processes noisy latent `z_t`, timestep
  `t`, and pinned I0.

Change:
- Implemented/extended `models/action_conditioned_wan.py` modes:
  - `side_adapter`: zero-initialized trainable residuals after selected frozen
    Wan transformer blocks; residuals cross-attend current hidden states to
    action tokens.
  - `pre_context`: context-free feature pass through patch/time embedding and
    first self-attention before cross-attention; predicts context tokens from
    pre-context features plus action.
  - `append_context` / `replace_context`: action tokens appended to or
    replacing text context for ablations.
- Added `scripts/train_action_conditioned_wan.py`, export support, and tests.

Command / Job:
- command: local/remote Python compile plus direct unit-test functions from
  `tests/test_action_conditioned_wan.py`
- job_id: n/a
- run_dir: local `/Users/lzha/code/Wan2.2`
- logs: terminal output; no pytest installed locally/remote
- artifacts: `models/action_conditioned_wan.py`,
  `docs/action_conditioned_wan.md`, `tests/test_action_conditioned_wan.py`

Result:
- status: passed
- metrics/artifacts:
  - `py_compile` and `bash -n` passed for modified scripts.
  - Direct unit-test functions passed remotely: `direct tests ok`.
- key evidence: wrapper and model tests in repo; no local pytest dependency.

Analysis:
- Side adapter is identity-preserving at initialization and uses Wan hidden
  states directly, making it the most plausible path for fresh-noise/25-step
  training. Pre-context avoids circularity but replaces text context and has a
  harder optimization target.

Next:
- Run one-sample overfits for pre-context and side adapter.

## 2026-06-08 - One-Step Fixed-Noise Pre-Context Overfit

Goal:
- Test whether pre-context can overfit one DROID sample at one diffusion step.

Hypothesis:
- If pre-context feature-to-context prediction is wired correctly, it should
  reduce one-step latent MSE, though likely slower than side adapter.

Change:
- Trained pre-context on `ep0_v0`, 1 diffusion step, fixed noise.

Command / Job:
- command: `run_action_conditioned_overfit.sh` with `MODE=pre_context`,
  `SAMPLING_STEPS=1`, `NOISE_MODE=fixed`
- job_id: representative fetched logs include `9475789`, `9475943`, `9477950`
  export
- run_dir: remote `runs/action_overfit_ep0_v0_prectx_1step_fixed_1k` and
  related pre-context runs
- logs: `_cluster/slurm_outputs/action-overfit/`
- artifacts: `_cluster/action_overfit_ep0_v0_prectx_1step_fixed_1k/`,
  videos under `videos_best/`

Result:
- status: passed
- metrics/artifacts:
  - `prectx_1step_fixed`: best `0.09873`, final eval `0.09863`
  - `prectx_1step_fixed_1k`: best `0.08103`, final eval `0.08147`
  - `prectx_constlr_1step_1k`: best `0.08664`, final eval `0.09065`
  - videos exported for the 1k fixed run.
- key evidence: summary JSONs and `videos_best` under `_cluster/`.

Analysis:
- Pre-context works but underfits relative to side adapter. It can learn a
  useful context, but the context bottleneck is likely too indirect for the
  immediate overfit objective.

Next:
- Prefer side adapter for the main 25-step/fresh-noise direction.

## 2026-06-08 - One-Step Fixed-Noise Side Adapter Overfit Sweep

Goal:
- Test whether side adapter can overfit one sample and compare layer coverage.

Hypothesis:
- Injecting action residuals into more Wan layers should improve one-step
  overfit because it gives the adapter more control while keeping the backbone
  frozen.

Change:
- Swept side adapter bottleneck/layers and training length on `ep0_v0`,
  1 diffusion step, fixed noise.

Command / Job:
- command: `run_action_conditioned_overfit.sh` with `MODE=side_adapter`,
  `SAMPLING_STEPS=1`, `NOISE_MODE=fixed`, varying layers/steps
- job_id: fetched logs include `9475704`, `9475749`, `9475790`, `9475944`,
  `9476002`, `9476003`, `9476105`, `9476106`, `9476149`, `9476540`
- run_dir: remote `runs/action_overfit_ep0_v0_side_*`
- logs: `_cluster/slurm_outputs/action-overfit/`
- artifacts: `_cluster/action_overfit_ep0_v0_side_*/summary.json`,
  `train_log.csv`, selected `videos_best/`

Result:
- status: passed
- metrics/artifacts:

| Attempt | Key setting | Best loss | Final eval | Decision |
| --- | --- | ---: | ---: | --- |
| side_bn512h8_1step_fixed | side, default layers, fixed | 0.04762 | 0.04752 | side beats pre-context |
| side_bn512h8_1step_fixed_1k | side, default layers, 1k | 0.04003 | 0.03997 | longer helps |
| side_bn1024h8_L24-29_1step_1k | late layers, hidden 1024 | 0.03986 | 0.04030 | wider late-only not enough |
| side_bn512h8_L15-29_1step_1k | layers 15-29 | 0.01890 | 0.01902 | more layers helps |
| side_bn512h8_L15-29_1step_2k | layers 15-29, 2k | 0.01406 | 0.01432 | longer helps |
| side_bn512h8_L0-29_1step_1k | all layers | 0.01455 | 0.01455 | all layers strong |
| side_bn512h8_L0-29_1step_2k | all layers, 2k | 0.01050 | 0.01065 | improves |
| side_bn512h8_L0-29_1step_4k | all layers, 4k | 0.00781 | 0.00810 | best one-step result |

- key evidence: `_cluster/action_overfit_ep0_v0_side_bn512h8_L0-29_1step_4k/summary.json`
  and exported videos under `videos_best/`.

Analysis:
- Side adapter all-layers is clearly the strongest of the tested one-step
  designs. This supports using side adapter for the 25-step fresh-noise run.

Next:
- Run side adapter at 25 diffusion steps with fresh training noise and random
  eval noise.

## 2026-06-08/09 - 25-Step Fresh-Noise Side Adapter Pilot

Goal:
- Test whether side adapter can learn one-sample overfit behavior under the
  more realistic 25-step fresh-noise setting.

Hypothesis:
- If the side adapter can condition on Wan hidden states/noisy latents/timestep,
  training with fresh initial noise should produce stable eval over random
  eval noises, unlike fixed-context/out-of-basin replay.

Change:
- Added `--noise_mode fresh` and random-noise export support.
- Trained side adapter all layers, hidden 512/head 8, 25 diffusion steps, fresh
  Gaussian initial noise.
- Generated random eval videos for seeds `1000..1003`.

Command / Job:
- command: `run_action_conditioned_overfit.sh` with
  `MODE=side_adapter`, `SIDE_ADAPTER_LAYERS=0-29`,
  `SIDE_ADAPTER_HIDDEN=512`, `SAMPLING_STEPS=25`, `NOISE_MODE=fresh`,
  pilot `TOTAL_STEPS=100`; export with `EVAL_NOISE_MODE=random`,
  `NUM_EVAL_NOISES=4`, `EVAL_SEED_START=1000`
- job_id: pilot job id not preserved in local summary; export artifacts fetched
- run_dir: remote
  `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_pilot`
- logs: fetched loss curve and train log in `_cluster/loss_curves/`
- artifacts:
  `_cluster/loss_curves/side_fresh25_pilot_*`,
  `_cluster/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_pilot/videos_best_random_eval_s1000_1003/`

Result:
- status: passed
- metrics/artifacts:
  - pilot best train loss `0.0985446`
  - fixed-reference final eval `0.1002754`
  - random eval latent MSEs:
    - seed1000 `0.0986487`
    - seed1001 `0.0986912`
    - seed1002 `0.0995091`
    - seed1003 `0.1000559`
  - null-only baselines much worse, e.g. seed1000 `2.539`, seed1001 `4.936`,
    seed1002 `2.493`
- key evidence:
  `_cluster/loss_curves/side_fresh25_pilot_summary.json`,
  `_cluster/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_pilot/videos_best_random_eval_s1000_1003/metrics.json`

Analysis:
- This is the first evidence that side adapter can do a meaningful 25-step
  fresh-noise one-sample run. The loss is still much higher than one-step
  fixed-noise overfit, but random eval is stable and close to train loss.

Next:
- Launch longer 10k-step fresh-noise overfit with same train/eval guidance
  scale.

## 2026-06-09 - 10k 25-Step Fresh-Noise Overfit Submission

Goal:
- Run a longer side adapter one-sample overfit to see whether the 25-step
  fresh-noise loss can continue down.

Hypothesis:
- The 100-step pilot is undertrained; 10k steps with warmup/cosine LR may
  substantially improve fresh-noise 25-step overfit.

Change:
- Submitted 10k side adapter job with train guidance scale matching eval
  guidance scale.

Command / Job:
- command:
  `sbatch --time=36:00:00 --job-name=act-side-fresh25-10k --export=ALL,MODE=side_adapter,RUN_TAG=side_bn512h8_L0-29_fresh_25step_10k_lr5e-5,TOTAL_STEPS=10000,SAMPLING_STEPS=25,SEED=0,NOISE_MODE=fresh,SIDE_ADAPTER_LAYERS=0-29,SIDE_ADAPTER_HIDDEN=512,SIDE_ADAPTER_HEADS=8,LR=5e-5,WARMUP_STEPS=500,LR_MIN_RATIO=0.1,LOG_INTERVAL=20,CKPT_INTERVAL=1000 run_action_conditioned_overfit.sh`
- job_id: `9478714`
- run_dir: remote
  `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5`
- logs:
  `slurm_outputs/action-overfit/out_act-side-fresh25-10k_9478714.log`,
  `slurm_outputs/action-overfit/err_act-side-fresh25-10k_9478714.log`
- artifacts: expected `train_log.csv`, checkpoints, random-eval videos after completion

Result:
- status: stalled
- metrics/artifacts:
  - job remained pending due `ReqNodeNotAvail, Reserved for maintenance`
  - generic `gpu` long-job probes estimated later starts than `ailab`
- key evidence: `scontrol show job 9478714` before maintenance showed
  partition `ailab`, QOS `gpu-medium`, start estimate around 2026-06-10
  evening.

Analysis:
- This is an external cluster availability blocker, not a code/training
  failure.

Next:
- When Della returns, monitor job `9478714`; if it completes, plot loss curve
  and export random-noise eval videos.

## 2026-06-09 - DROID High-Action Window Dataset Planning

Goal:
- Prepare a full DROID training cache without exceeding a 500 GiB cache budget,
  using 50-100 high-action windows per eligible episode when possible.

Hypothesis:
- Selecting high-action windows avoids wasting cache/training on stationary
  early frames and makes a feasible full-DROID side adapter training set.

Change:
- Added:
  - `scripts/make_droid_window_plan.py`
  - `scripts/precompute_features_droid_plan.py`
  - `run_make_droid_window_plan.sh`
  - `run_precompute_droid_window_plan_array.sh`
- Planned train/val windows, view `0`, frames `33`, candidate stride `4`, fp16
  cache estimate.

Command / Job:
- command: `run_make_droid_window_plan.sh`
- job_id: planner jobs completed before current worklog; exact IDs not
  preserved in local summary
- run_dir: remote `runs/droid_window_plans`
- logs: remote Slurm logs under `slurm_outputs/droid-window-cache/`
- artifacts:
  `runs/droid_window_plans/train_v0_33f_stride4_cap500g_fp16_top50.jsonl`,
  `runs/droid_window_plans/train_v0_33f_stride4_cap500g_fp16_top50.summary.json`,
  val equivalents

Result:
- status: passed
- metrics/artifacts:
  - old cache before planning: `289` train, `29` val samples
  - raw file-eligible train episodes: `94,321` per view
  - train plan selected `1,440,554` windows from `88,237` episodes
  - skipped too-short train episodes: `6,084`
  - train estimated cache: `321.99 GiB`
  - val plan selected `14,636` windows, estimated `3.27 GiB`
  - selected windows/episode quantiles: min `1`, p10 `3`, p50 `12`,
    p90 `40`, p99 `50`, max `50`
- key evidence: remote summaries recorded in monitoring notes; local scripts
  encode the plan logic.

Analysis:
- The plan satisfies the 500 GiB cap and prioritizes big actions. It is large
  enough for a real full-DROID run while still tractable on scratch.

Next:
- Materialize val and train caches in fp16.

## 2026-06-09 - DROID Cache Materialization

Goal:
- Precompute Wan VAE latents/actions for planned DROID windows.

Hypothesis:
- fp16 latent cache entries are accurate enough for training and reduce the
  footprint enough to fit the 500 GiB target.

Change:
- Ran smoke/pilot cache jobs, val full cache, and train chunked arrays.
- Patched precompute script to exit nonzero on skips/fails.

Command / Job:
- command: `run_precompute_droid_window_plan_array.sh`
- job_id:
  - smoke `9478715_0`: 2 windows
  - val pilot `9478801_0`: 100 windows
  - val full array `9478848`
  - train first array `9478947`
- run_dir: remote `data/droid_cache_windows_v0/{train,val}`
- logs: remote `slurm_outputs/droid-window-cache/`
- artifacts: fp16 `z_I0.pt`, `z_video.pt`, `actions.npy`, `meta.json`

Result:
- status: partially passed / stalled by maintenance
- metrics/artifacts:
  - smoke `9478715_0`: `done ok=2 skip=0 fail=0`
  - val pilot `9478801_0`: `done ok=100 skip=0 fail=0`
  - val full cache complete: `14,636` windows, about `3.3G`
  - verified fp16 cache entry:
    - `z_video` dtype `torch.float16`, shape `(48,9,12,20)`
    - `z_I0` shape `(48,1,12,20)`
    - actions shape `(32,7)`
  - train cache reached `140,679` windows before maintenance blocked progress.
- key evidence: monitoring notes and remote logs; local cache code in
  `scripts/precompute_features_droid_plan.py`

Analysis:
- Cache pipeline is valid. Della maintenance, not data/code failures, blocked
  completion.

Next:
- Resume chunk submitter when Della returns; verify no shard failed/timed out.

## 2026-06-09 - DROID Dataset Trainer Implementation

Goal:
- Train side adapter on the planned DROID cache with the same closed-loop
  rollout objective used in overfit diagnostics.

Hypothesis:
- Full-DROID training should use fresh Gaussian initial noise, 25 diffusion
  steps, guide scale `5.0`, and cached latents loaded lazily from manifest to
  avoid millions of eager filesystem stats.

Change:
- Added:
  - `scripts/train_action_conditioned_wan_droid.py`
  - `run_action_conditioned_droid.sh`
  - manifest-backed `LazyTripletLatentDataset`
  - train/val CSV logging
- Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to DROID and
  overfit wrappers.

Command / Job:
- command: syntax checks, remote compile checks, DROID H200 smoke.
- job_id: H200 smoke `9479332`
- run_dir: remote `runs/action_droid_smoke_partial_h200_1step`
- logs:
  `slurm_outputs/action-droid/out_act-droid-h200-smoke_9479332.log`,
  `slurm_outputs/action-droid/err_act-droid-h200-smoke_9479332.log`
- artifacts: remote smoke `config.json`, `train_log.csv`, `val_log.csv`,
  `summary.json`

Result:
- status: passed
- metrics/artifacts:
  - H200 smoke completed exit `0:0` in `00:01:36`
  - train samples `8`, val samples `2`, mode `side_adapter`
  - steps `25`, train steps `1`, noise mode `fresh`, guide scale `5.0`
  - train loss `2.03346`, val loss `2.90328`
  - trainable params `239.50M`
  - `MaxRSS` about `41,384,956K`
- key evidence: remote `summary.json` printed during monitoring.

Analysis:
- Trainer is wired correctly and can execute one full 25-step rollout +
  validation on H200. The high one-step smoke loss is expected before training.

Next:
- Let post-cache submitter launch a small DROID smoke after full cache
  completion, then full 10k training.

## 2026-06-09 - Gputest Trainer Smoke Failure

Goal:
- Validate the DROID trainer early on a short `gpu-test` allocation while the
  cache array was running.

Hypothesis:
- A 40 GB `gputest` A100 might be enough for a one-step trainer smoke.

Change:
- Submitted a one-step DROID trainer smoke directly to `gputest`.

Command / Job:
- command: direct `sbatch` wrapper with `--qos=gpu-test`, `TOTAL_STEPS=1`,
  `MAX_TRAIN_SAMPLES=8`, `MAX_VAL_SAMPLES=2`
- job_id: `9479259`
- run_dir: remote `runs/action_droid_smoke_partial_gputest_1step`
- logs:
  `slurm_outputs/action-droid/out_act-droid-wrap-smoke_9479259.log`,
  `slurm_outputs/action-droid/err_act-droid-wrap-smoke_9479259.log`
- artifacts: partial config/log CSVs

Result:
- status: failed
- metrics/artifacts:
  - reached Wan model execution
  - failed with CUDA OOM on 40 GB A100:
    `torch.OutOfMemoryError: CUDA out of memory`
- key evidence: `err_act-droid-wrap-smoke_9479259.log`

Analysis:
- This was a hardware-capacity mismatch, not a trainer code failure. H200 smoke
  passed immediately after.

Next:
- Use H200/ailab for trainer validation and training; reserve `gpu-test` for
  VAE cache shards that fit 40 GB.

## 2026-06-09 - Gputest Cache Backfill Strategy

Goal:
- Continue DROID cache materialization while `ailab` was blocked by maintenance.

Hypothesis:
- Smaller VAE precompute shards can fit within short `gpu-test` backfill
  windows on 40 GB A100s, unlike full trainer jobs.

Change:
- Added:
  - `run_precompute_droid_window_plan_array_anygpu.sh`
  - configurable `scripts/submit_droid_train_cache_chunks.sh`
- Tested 1024-way sharding and switched pending train-cache work from
  128-shard `ailab` array to 1024-shard `gpu-test` chunks.

Command / Job:
- command:
  - 1024 pilot direct `sbatch --qos=gpu-test --time=00:10:00`
  - remote submitter with `TOTAL_SHARDS=1024`, `START_SHARD=97`,
    `CHUNK_SIZE=24`, `CONCURRENCY=3`
- job_id:
  - 1024 pilot `9479442`
  - active chunk array `9479448`
- run_dir: remote `data/droid_cache_windows_v0/train`
- logs:
  `slurm_outputs/droid-window-cache/out_droid-cache-1024-pilot_9479442.log`,
  `slurm_outputs/droid-window-cache/out_droid-train-cache-s097-120_9479448_*.log`
- artifacts: additional cached train windows

Result:
- status: partially passed / stalled by maintenance
- metrics/artifacts:
  - 1024 pilot shard `96/1024`: `1407` rows, completed in `5:26`,
    `ok=1407 skip=0 fail=0`
  - first `gpu-test` chunk wave shards `97`, `98`, `99` completed cleanly,
    each `ok=1407 skip=0 fail=0`
  - train cache count reached `140,679`
  - remaining `9479448_[100-120]` pending with `ReqNodeNotAvail`
- key evidence: monitoring notes and remote submitter log.

Analysis:
- Small cache shards work on `gpu-test`, but backfill availability disappeared
  as maintenance started. The submitter bug around empty job ids was fixed
  after a too-large `CHUNK_SIZE=96` submission hit QOS submit limits.

Next:
- When Della returns, check `9479448` and submitter PID `2821907`; resume or
  restart chunking only after verifying no duplicate/incomplete shards.

## 2026-06-09 - Della Maintenance / SSH Blocker

Goal:
- Continue monitoring jobs and submitters.

Hypothesis:
- SSH failures were initially stale gateway ControlMaster / auth issues, then
  Della maintenance made the login host unreachable from the gateway.

Change:
- Diagnosed SSH via `tigressgateway` and Python socket tests from the gateway.
- Created global Della workflow skill for future cluster usage:
  `/Users/lzha/.codex/skills/della-gpu-cluster-workflow/SKILL.md`.

Command / Job:
- command:
  - `ssh tigressgateway 'hostname && date'`
  - `ssh della-gpu 'hostname && date'`
  - gateway socket test to `della-gpu.princeton.edu:22`
- job_id: n/a
- run_dir: n/a
- logs: terminal output
- artifacts: Della skill

Result:
- status: blocked
- metrics/artifacts:
  - `tigressgateway` reachable
  - from gateway, `della-gpu.princeton.edu` resolved to `128.112.173.250`
  - socket connection to Della returned `No route to host`
  - gateway has no `squeue` and no `/scratch/gpfs/...`, so jobs cannot be
    monitored from gateway alone.
- key evidence: terminal diagnostics.

Analysis:
- Current blocker is Della maintenance/network reachability, not local SSH
  auth and not code. Remote jobs and submitters launched before maintenance may
  remain queued/running according to Slurm once the cluster returns.

Next:
- After `ssh della-gpu 'hostname && date'` works, immediately check:
  - `squeue -u $(whoami)`
  - cache counts under `data/droid_cache_windows_v0/{train,val}`
  - submitter PIDs/logs
  - job `9478714`
  - array `9479448`

## 2026-06-09 13:10 PDT - Della SSH Restored, GPU Nodes Still Unavailable

Goal:
- Resume the overfit and DROID full-training loop after the user reported that
  `della-gpu` was back.

Hypothesis:
- SSH access may be restored before Slurm GPU partitions are usable; queued
  jobs should be inspected before launching duplicates.

Change:
- Re-read the Della and robotics cluster workflow skills.
- Checked SSH, queue state, node state, cache counts, submitter liveness, job
  accounting, and test-only Slurm allocation routes.
- Did not launch or resubmit jobs because new test-only GPU allocations failed.

Command / Job:
- command:
  - `ssh della-gpu 'hostname; date; squeue -u $USER ...'`
  - `ssh della-gpu 'sinfo -o ...'`
  - `ssh della-gpu 'sacct -j 9478714,9479448 ...'`
  - `ssh della-gpu 'pgrep -af "submit_droid_train_cache_chunks|wait_and_submit_action_droid_training|train_action_conditioned_wan" || true'`
  - `sbatch --test-only` probes for `ailab/ailab` and `gpu-test`
- job_id: existing `9478714`, existing array `9479448_[100-120]`
- run_dir: remote `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs: existing Slurm logs under `slurm_outputs/action-overfit` and
  `slurm_outputs/droid-window-cache`
- artifacts: no new artifacts

Result:
- status: blocked by cluster GPU node state
- metrics/artifacts:
  - SSH works: host `della-gpu.princeton.edu`, remote time
    `Tue Jun 9 16:03:14 EDT 2026`
  - `9478714` is pending on `ailab`:
    `(ReqNodeNotAvail, Reserved for maintenance)`
  - `9479448_[100-120]` is pending on `gputest`: `(ReqNodeNotAvail)`
  - remote submitters are not running anymore.
  - DROID cache counts are unchanged: train `140,679`, val `14,636`.
  - `sinfo` shows GPU partitions still down/draining:
    `ailab` has `3` drain and `15` down nodes; `gputest` has `1` drain and
    `111` down nodes.
  - `sbatch --test-only` for both `ailab` and `gpu-test` returned
    `allocation failure: Requested node configuration is not available`.
- key evidence: live Slurm queries and test-only allocation probes.

Analysis:
- Della login access has returned, but GPU scheduling has not. The correct
  action is to leave the already queued jobs in place and avoid duplicate
  submissions until at least one target GPU route accepts a test-only
  allocation.

Next:
- Recheck `sinfo`, `squeue`, and `sbatch --test-only` periodically.
- When `gpu-test` becomes usable, let or restart the bounded cache chunking
  from shard `100` onward after confirming no overlapping submitter is alive.
- When `ailab` becomes usable, monitor `9478714`; after completion, fetch loss
  curves and export/evaluate random-noise videos.
- Once the train cache reaches the planned `1,440,554` windows, allow the
  DROID training submitter to launch smoke/full H200 training and monitor loss.

## 2026-06-09 13:25 PDT - Guarded Submitter Restart Blocked By Gateway Auth

Goal:
- Restore unattended orchestration without submitting duplicate GPU jobs.

Hypothesis:
- The existing cache submitter can safely resume by waiting on current array
  `9479448` and then continuing from shard `121`; the DROID training submitter
  can safely wait on full cache completion.

Change:
- Inspected local and remote copies of:
  - `scripts/submit_droid_train_cache_chunks.sh`
  - `scripts/wait_and_submit_action_droid_training.sh`
- Verified `submit_droid_train_cache_chunks.sh` supports `FIRST_WAIT_JOB`, so
  a safe resume command would use `FIRST_WAIT_JOB=9479448 START_SHARD=121`.
- Verified `wait_and_submit_action_droid_training.sh` gates smoke/full DROID
  training on the expected cache counts.

Command / Job:
- attempted command:
  - launch cache submitter with
    `LOG_PATH=runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_121.log`,
    `FIRST_WAIT_JOB=9479448`, `START_SHARD=121`, `TOTAL_SHARDS=1024`,
    `SBATCH_QOS=gpu-test`
  - launch DROID training submitter with
    `LOG_PATH=runs/droid_window_plans/action_droid_training_submitter_resume.log`
- job_id: n/a
- run_dir: remote `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs: no new remote logs confirmed
- artifacts: no new artifacts

Result:
- status: blocked by gateway authentication
- metrics/artifacts:
  - launch attempt failed with `Connection closed by UNKNOWN port 65535`.
  - clean BatchMode probe to `tigressgateway` then failed with
    `Permission denied (publickey,keyboard-interactive)`.
  - clean BatchMode probe to `della-gpu` failed through the gateway with the
    same auth error.
  - no local SSH/proxy processes were left running after cleanup.

Analysis:
- SSH login was temporarily usable via an existing authenticated gateway
  session, but clearing the stale-looking multiplex socket removed that path.
  The next step requires the user to refresh SSH/Duo gateway auth locally before
  Codex can launch remote submitters again.

Next:
- User should re-authenticate with a direct local `ssh della-gpu` or
  `ssh tigressgateway` attempt.
- After SSH works again, run the guarded cache submitter resume:
  `FIRST_WAIT_JOB=9479448 START_SHARD=121 TOTAL_SHARDS=1024 ...`.
- Also restart the cache-gated DROID training submitter if no matching process
  is already alive.

## 2026-06-09 13:20 PDT - Local Version-Control Snapshot

Goal:
- Bring the local `Wan2.2` development tree under Git so future cluster syncs
  and launches can record exact implementation commits, following the updated
  `robotics-cluster-development-core` contract.

Hypothesis:
- A clean source/docs/tests commit plus conservative artifact ignores is enough
  to preserve all implementation edits so far without committing generated
  videos, logs, checkpoints, or cached DROID windows.

Change:
- Read the latest `robotics-cluster-development-core/SKILL.md`.
- Added ignore rules for local tool state, generated run artifacts, caches,
  logs, videos, checkpoints, and tensor dumps.
- Prepared this worklog entry for the local snapshot commit.

Version Control:
- branch: local Git repo did not exist before this checkpoint.
- base_commit: n/a
- implementation_commit: `e07e71f`
- changed_files:
  - `.gitignore`
  - `WORKLOG.md`
  - all source/docs/tests/scripts currently staged in the initial snapshot
- remote_commit/status: Della unavailable due maintenance / gateway access;
  no new remote sync or launch.

Command / Job:
- command: `git init && git add <explicit source/docs/tests/scripts files> && git commit`
- job_id: n/a
- run_dir: local `/Users/lzha/code/Wan2.2`
- logs: n/a
- artifacts: local Git history, commit `e07e71f`

Result:
- status: passed
- metrics/artifacts:
  - initial snapshot commit `e07e71f`
  - staged snapshot contained 162 files and 40,285 inserted lines
- key evidence:
  - `Wan2.2` had no `.git` directory before this checkpoint.
  - Shell syntax check passed for all staged `*.sh` files.
  - Python syntax compilation passed for `models`, `scripts`, `wan`, and
    top-level Python entrypoints.
  - Focused `git diff --check` passed for the authored/changed action-adapter,
    z-init, DROID planning, and worklog files.

Analysis:
- Because there was no existing local Git history, the first commit must be a
  clean implementation snapshot rather than a minimal patch against a previous
  baseline. Generated `_cluster` artifacts are documented in the worklog and
  kept out of Git.
- Full-repo `git diff --check` is noisy for the initial snapshot because the
  upstream Wan files already contain trailing whitespace; I did not rewrite
  unrelated upstream files for this checkpoint.

Next:
- Use commit `e07e71f` as the local implementation baseline for future Della
  syncs and job worklog entries.

## 2026-06-09 13:33 PDT - Fork-Based Git Deployment Initialization

Goal:
- Replace the old rsync-for-code workflow with the Git workflow required by
  the updated Della and generic cluster skills.

Hypothesis:
- The safe one-time fix is to preserve the rsync-derived snapshot, attach the
  repo to the real upstream `Wan-Video/Wan2.2` history, create a user fork for
  writable development, and make future Della deployments pull exact commits
  from that fork.

Change:
- Read the latest `della-gpu-cluster-workflow` and
  `robotics-cluster-development-core` skills.
- Fetched upstream `https://github.com/Wan-Video/Wan2.2.git` at `42bf4cf`.
- Created fork `https://github.com/lihzha/Wan2.2`.
- Preserved the standalone rsync-derived local history on branch
  `rsync-snapshot`.
- Reset local `main` to `upstream/main`, applied all local action-adapter,
  z-init, DROID, docs, tests, and worklog changes on top of upstream history,
  and avoided deleting upstream tracked sample videos that were absent from the
  rsync copy.
- Updated `scripts/della_loop.sh` and `docs/della_workflow.md` so tracked code
  uses `commit -> push -> Della git fetch/reset`, while rsync remains only for
  logs/results/artifacts.

Version Control:
- branch: `main`
- base_commit: `42bf4cf` (`upstream/main`)
- implementation_commit: `b99d040`
- push/pull: fork push succeeded; Della pull blocked because `ssh della-gpu`
  still fails with `Connection closed by UNKNOWN port 65535`
- changed_files:
  - action-adapter implementation, trainers, exporters, tests, and docs
  - z-init/inversion analysis scripts and wrappers
  - DROID cache planning/training scripts and wrappers
  - `scripts/della_loop.sh`
  - `docs/della_workflow.md`
  - `WORKLOG.md`
- remote_commit/status:
  - `origin/main` verified at `3c5c8f8ee93b5e88ee718d11e2b6141fbf387fb5`
    before this final worklog update
  - Della checkout not reachable

Command / Job:
- command:
  - `gh repo fork Wan-Video/Wan2.2 --clone=false --default-branch-only`
  - `git fetch upstream main --tags`
  - `git switch -C main upstream/main`
  - `git apply --index <local_changes_no_deletes.patch>`
  - `git commit -m "Add action adapter experiments and Git-based Della workflow"`
- job_id: n/a
- run_dir: local `/Users/lzha/code/Wan2.2`
- logs: n/a
- artifacts: fork `https://github.com/lihzha/Wan2.2`, local branch
  `rsync-snapshot`

Result:
- status: passed locally and pushed to fork; cluster conversion pending
- metrics/artifacts:
  - upstream-based commit `b99d040`
  - fork worklog/provenance commit `3c5c8f8`
  - no tracked deletions relative to upstream, so upstream sample videos are
    preserved
  - `scripts/della_loop.sh deploy-code` now pushes local `HEAD`, initializes or
    updates the Della checkout with Git, resets tracked files to the fetched
    commit, and verifies the remote HEAD matches
- key evidence:
  - shell syntax check passed for all `*.sh` files
  - Python compile check passed for `models`, `scripts`, `wan`, and top-level
    Python entrypoints
  - focused `git diff --check` passed for the authored workflow/action/z-init
    files

Analysis:
- The earlier `.git` absence was a direct consequence of the old rsync workflow
  excluding `.git/`. The updated skill correctly treats this as a provenance
  problem: the cluster should run a Git checkout, not an opaque rsync mirror.
- Rebuilding on top of upstream history is better than pushing the standalone
  `e07e71f`/`14f5458` root history because it keeps future upstream merges and
  fork comparisons meaningful.

Next:
- When Della is reachable, run `scripts/della_loop.sh deploy-code main` to
  initialize/update `/scratch/gpfs/AM43/lz3952/Wan2.2` as a Git checkout and
  verify the remote commit before launching more jobs.

## 2026-06-09 13:47 PDT - Della Git Checkout Conversion

Goal:
- Convert the existing Della scratch tree from an rsync-contaminated checkout
  into a clean Git deployment from the fork.

Hypothesis:
- Della is reachable again, and the scratch tree can be safely reset to the
  fork commit while preserving untracked runtime state such as checkpoints,
  caches, runs, and logs.

Change:
- Confirmed `ssh della-gpu` works.
- Inspected `/scratch/gpfs/AM43/lz3952/Wan2.2`; it was a Git checkout at
  upstream `42bf4cf` with many rsync-copied files untracked and two tracked
  Wan files modified.
- Updated `scripts/della_loop.sh deploy-code` to use forced checkout for
  tracked paths during first-time conversion.
- Added uppercase video and PDF artifact patterns to `.gitignore`.

Version Control:
- branch: `main`
- base_commit: `7a3ae33`
- implementation_commit: `90402c9`
- push/pull:
  - pushed `90402c9` to `origin/main`
  - deployed `90402c9` to Della with `scripts/della_loop.sh deploy-code main`
  - final worklog-only update will be pushed/deployed after this entry
- changed_files:
  - `.gitignore`
  - `scripts/della_loop.sh`
  - `WORKLOG.md`
- remote_commit/status:
  - Della reachable
  - remote checkout before conversion: `42bf4cf`
  - remote tracked dirty files before conversion:
    `wan/modules/vae2_2.py`, `wan/textimage2video.py`
  - remote checkout after conversion: `90402c9`
  - remote tracked status after conversion: clean

Command / Job:
- command: `scripts/della_loop.sh deploy-code main`
- job_id: n/a
- run_dir: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs: terminal output
- artifacts: remote Git checkout

Result:
- status: passed
- metrics/artifacts:
  - Della remote HEAD matched local/fork commit
    `90402c9c49d0ccde23d6d71acc61c3822cffa499`
  - remote tracked status was clean with `git status --short --untracked-files=no`
  - runtime paths still existed after conversion:
    `data`, `runs`, `slurm_outputs`, `Wan2.2-TI2V-5B`, `.venv`
  - current queue after Della came back still showed pending maintenance/node
    availability for jobs `9478714` and `9479448_[100-120]`
- key evidence:
  - `deploy-code` printed matching remote and expected heads
  - `scripts/della_loop.sh remote-git-status` returned `main` at `90402c9`

Analysis:
- The remote tree already had upstream Git history, but the old rsync flow had
  placed our development files into it as untracked files. A normal checkout
  could reject those paths, so the deploy helper must force tracked path
  replacement while leaving unrelated untracked runtime outputs alone.

Next:
- Commit and deploy this final worklog update so local, fork, and Della remain
  aligned before relaunching or monitoring jobs.

## 2026-06-09 20:42 PDT - Monitoring Loop Restart

Goal:
- Resume active monitoring after Della SSH came back and keep both the overfit
  and full DROID training pipeline moving.

Hypothesis:
- The previously pending Slurm jobs should resume after maintenance, and the
  DROID train cache submitter needs to be restarted from shard `121` because
  the pre-maintenance submitter stopped after array `9479448`.

Change:
- Verified Della checkout was clean at `f5083b2`.
- Confirmed overfit job `9478714` was running on `della-i23g1`.
- Confirmed cache array `9479448_100-120` completed cleanly.
- Restarted remote cache submitter from shard `121` with `gpu-test`, chunk size
  `24`, concurrency `3`, total shards `1024`.
- Restarted the cache-gated DROID training waiter.

Version Control:
- branch: `main`
- base_commit: `f5083b2`
- implementation_commit: no code change
- push/pull: local/fork/Della were aligned at `f5083b2` before monitoring;
  this worklog-only update is pushed to GitHub but not deployed to Della yet
  because SSH auth expired again during artifact fetch
- changed_files:
  - `WORKLOG.md`
- remote_commit/status:
  - Della checkout: `f5083b2`, tracked status clean at monitor start
  - SSH auth later failed with `Permission denied` on `tigressgateway`

Command / Job:
- command:
  - `nohup env ... START_SHARD=121 ... bash scripts/submit_droid_train_cache_chunks.sh`
  - `nohup env POLL_SECONDS=900 ... bash scripts/wait_and_submit_action_droid_training.sh`
  - bounded polling of `squeue`, `sacct`, cache counts, and overfit loss
- job_id:
  - overfit: `9478714`
  - resumed cache array: `9491200` for shards `121-144`
  - remote cache submitter PID: `1871034`
  - remote DROID training waiter PID: `1883834`
- run_dir: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs:
  - `slurm_outputs/action-overfit/out_act-side-fresh25-10k_9478714.log`
  - `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_121.log`
  - `runs/droid_window_plans/action_droid_training_submitter_resume.log`
- artifacts:
  - overfit run
    `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5`
  - DROID cache under `data/droid_cache_windows_v0`

Result:
- status: active, locally blocked from further polling by SSH auth
- metrics/artifacts:
  - overfit was healthy at step `2620/10000`, loss `0.054346`, recent mean
    about `0.073`; H200 GPU was active and checkpoints existed.
  - cache count increased from `170,221` to `178,752` during the monitor
    window; validation remained complete at `14,636`.
  - cache array `9491200` was progressing: shards `121-124` and `126`
    completed cleanly by the last poll, while `125`, `127`, and `128` were
    running.
  - DROID training waiter was running and correctly waiting for
    `1,440,554` train cache windows.
- key evidence:
  - `sacct` showed `9478714` running and cache tasks returning
    `COMPLETED|0:0`.
  - `train_log.csv` was advancing from step `2560` to `2620`.

Analysis:
- The overfit loss curve is stable enough to continue. A single spike around
  step `2240` is not enough to justify an LR change because subsequent losses
  returned to roughly `0.05`.
- The cache pipeline is now moving again. The old submit-limit error in
  `train_cache_chunk_submitter_gputest_1024.nohup.log` is historical, not from
  the resumed submitter.
- The current blocker is local SSH/Duo authentication, not the remote jobs.
  The remote submitters should continue without this local session.

Next:
- Refresh SSH auth to `tigressgateway`/`della-gpu`.
- After auth works, poll `9491200`, `9478714`, and the two submitter logs.
- Fetch and plot the current overfit `train_log.csv`; the fetch attempt failed
  only because SSH auth expired.
- Deploy this worklog-only commit to Della after auth is restored.

## 2026-06-09 22:12 PDT - Local Watch Restart Blocked By Della Auth

Goal:
- Continue the local watch loop for the active Della overfit job, DROID cache
  submitter, and post-cache DROID training waiter.

Hypothesis:
- Della SSH auth had recovered, so a bounded local watcher could resume polling
  queue state, loss curves, cache counts, and submitter logs.

Change:
- No code change. Cleaned up a stale inherited local SSH proxy/session before
  attempting a fresh direct poll.

Version Control:
- branch: `main`
- base_commit: `9ec0395d66877658c9dd4daf5c869d09c02d2db9`
- implementation_commit: pending worklog-only update
- push/pull: not pushed or deployed yet; Della auth blocked remote pull
- changed_files:
  - `WORKLOG.md`
- remote_commit/status:
  - last live-verified remote commit before this attempt:
    `9ec0395d66877658c9dd4daf5c869d09c02d2db9`, clean
  - current attempt could not run `git rev-parse` on Della because SSH to
    `della-gpu` failed with `Permission denied (keyboard-interactive)`

Command / Job:
- command:
  - local cleanup:
    `ps ... grep 'ssh .*della|ssh .*tigress'`, then `kill <stale ssh proxy>`
  - auth checks:
    `ssh -o BatchMode=yes tigressgateway 'hostname && date'`
    and `ssh -o BatchMode=yes della-gpu 'hostname && date'`
  - reachability check:
    `ssh tigressgateway 'python3 - ... connect(("della-gpu.princeton.edu", 22))'`
- job_id:
  - overfit: `9478714`
  - last observed cache array: `9493108`
  - remote cache submitter PID: `1871034`
  - remote DROID training waiter PID: `1883834`
- run_dir: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs:
  - `slurm_outputs/action-overfit/out_act-side-fresh25-10k_9478714.log`
  - `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_121.log`
  - `runs/droid_window_plans/action_droid_training_submitter_resume.log`
- artifacts:
  - local plots last refreshed at `_cluster/loss_curves/`

Result:
- status: blocked by Della login authentication
- metrics/artifacts:
  - gateway auth works: `tigressgateway.rc.princeton.edu` responded at
    `Wed Jun 10 01:11:27 AM EDT 2026`.
  - Della host is reachable from the gateway:
    `128.112.173.250`, `SSH-2.0-OpenSSH_8.7`.
  - Della login rejects noninteractive auth:
    `Permission denied (publickey,keyboard-interactive)`.
  - no local SSH monitor/proxy process remained after cleanup.
- key evidence:
  - direct `ssh -o BatchMode=yes della-gpu 'hostname && date'` failed with
    `Permission denied (keyboard-interactive)`.
  - nested gateway-to-Della probe failed with
    `Permission denied (publickey,keyboard-interactive)` after host-key setup.

Analysis:
- This is not a Della network or maintenance outage: the login host is
  reachable and returns an SSH banner. The blocker is local authentication to
  `della-gpu`, likely an expired or missing keyboard-interactive/Duo session.
- Last live state before this auth failure remained healthy: overfit job
  `9478714` was running around step `3460` with recent losses near `0.058`,
  cache count had reached about `265,724` train windows, and cache submitter
  array `9493108` was processing shards `169-192`.

Next:
- Refresh local authentication to `della-gpu`.
- Restart bounded polling of `9478714`, current cache array, cache count,
  submitter PIDs/logs, and error scans.
- Fetch updated `train_log.csv` and cache submitter log, regenerate loss/cache
  SVGs under `_cluster/loss_curves/`, then commit and deploy this worklog entry.

## 2026-06-09 23:47 PDT - Della Watch Resumed

Goal:
- Resume the local watch loop after Della SSH authentication was restored.

Hypothesis:
- The remote submitters continued correctly while local auth was unavailable,
  so the next local poll should show monotonic cache progress and a stable
  overfit loss curve.

Change:
- No code change. Deployed the previous worklog-only commit to Della and
  refreshed local ignored monitor artifacts.

Version Control:
- branch: `main`
- base_commit: `60f7a585d385bbd2ee0ec27c8ed50bf4072ebf62`
- implementation_commit: pending worklog-only update
- push/pull: `scripts/della_loop.sh deploy-code main` fast-forwarded Della to
  `60f7a585d385bbd2ee0ec27c8ed50bf4072ebf62`
- changed_files:
  - `WORKLOG.md`
- remote_commit/status:
  - remote commit: `60f7a585d385bbd2ee0ec27c8ed50bf4072ebf62`
  - remote tracked status clean; untracked model/scratch directories present:
    `6/`, `Wan2.2-TI2V-5B/`

Command / Job:
- command:
  - `scripts/della_loop.sh deploy-code main`
  - repeated bounded polls of `squeue`, cache directory counts, overfit
    `train_log.csv`, submitter logs, `sacct`, and error scans
  - `rsync` fetched only small CSV/log artifacts for local plots
- job_id:
  - overfit: `9478714`
  - cache array completed: `9495400` for shards `217-240`
  - current cache array: `9496241` for shards `241-264`
  - remote cache submitter PID: `1871034`
  - remote DROID training waiter PID: `1883834`
- run_dir: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs:
  - `slurm_outputs/action-overfit/out_act-side-fresh25-10k_9478714.log`
  - `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_121.log`
  - `runs/droid_window_plans/action_droid_training_submitter_resume.log`
- artifacts:
  - local loss plot:
    `_cluster/loss_curves/side_fresh25_10k_loss_current.svg`
  - local cache progress plot:
    `_cluster/loss_curves/droid_cache_progress_current.svg`
  - local summary:
    `_cluster/loss_curves/current_monitor_summary.json`

Result:
- status: active and healthy
- metrics/artifacts:
  - overfit `9478714` running on `della-i23g1`; latest observed step
    `4160/10000`, loss `0.0638015`, recent-10 mean `0.0626973`,
    recent-25 mean `0.0627811`.
  - DROID cache count advanced to `339,499/1,440,554` train windows by direct
    directory count; validation remained complete at `14,636`.
  - cache array `9495400` completed cleanly and the submitter launched
    `9496241` for shards `241-264`.
  - scratch filesystem had about `1.3T` available at the latest `df` poll.
- key evidence:
  - submitter log line:
    `[2026-06-10 02:45:13] job 9495400 completed cleanly`
  - no current traceback/OOM/failure lines appeared in the current error scan.

Analysis:
- No intervention is needed yet. The overfit loss is stable around `0.06`;
  the previous large gradient/loss spike appears transient and the optimizer
  recovered. The LR is already decaying, so lowering LR now would likely slow
  the ongoing test without clear benefit.
- Cache generation is continuing at roughly one 24-shard chunk every few
  minutes under the `gpu-test` array limit. The DROID training waiter remains
  correctly blocked on the full train-cache target.

Next:
- Continue polling overfit loss and cache arrays.
- When overfit `9478714` completes, run/fetch random-noise eval videos and
  record final video paths.
- When cache reaches the train target, verify that the waiter submits the DROID
  smoke/full training jobs and then monitor their loss/validation curves.

## 2026-06-10 00:03 PDT - Monitor Loop Reblocked By Della Auth

Goal:
- Continue the local monitor loop for the active overfit job, DROID cache
  submitter, and post-cache DROID training waiter.

Hypothesis:
- The SSH session used for the previous resumed watch might still be valid, so
  a new bounded poll should be able to refresh queue state and loss/cache
  curves.

Change:
- No code change.

Version Control:
- branch: `main`
- base_commit: `bc8ab2270986002ca7cd77519fb863d43d5c7071`
- implementation_commit: pending worklog-only update
- push/pull: local commit can be pushed to GitHub; Della pull is blocked until
  login auth is refreshed
- changed_files:
  - `WORKLOG.md`
- remote_commit/status:
  - last live-verified remote commit:
    `bc8ab2270986002ca7cd77519fb863d43d5c7071`
  - current remote status could not be refreshed because `ssh della-gpu`
    failed with `Permission denied (keyboard-interactive)`

Command / Job:
- command:
  - `ssh -o BatchMode=yes -o ConnectTimeout=20 della-gpu 'hostname && date ...'`
  - `ssh -o BatchMode=yes -o ConnectTimeout=15 tigressgateway 'hostname && date'`
  - gateway-side TCP probe to `della-gpu.princeton.edu:22`
- job_id:
  - overfit: `9478714`
  - last observed cache array: `9496241`
  - remote cache submitter PID: `1871034`
  - remote DROID training waiter PID: `1883834`
- run_dir: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs:
  - `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_121.log`
  - `runs/droid_window_plans/action_droid_training_submitter_resume.log`

Result:
- status: blocked by Della login authentication
- metrics/artifacts:
  - `tigressgateway` remains reachable and authenticated.
  - `della-gpu.princeton.edu` is reachable from the gateway and returns
    `SSH-2.0-OpenSSH_8.7`.
  - direct Della login fails with
    `Permission denied (keyboard-interactive)`.
  - no local SSH monitor/proxy processes remained after the failed attempt.
- key evidence:
  - gateway command returned
    `tigressgateway.rc.princeton.edu` at `Wed Jun 10 03:03:15 AM EDT 2026`.
  - Della TCP probe returned `tcp_connect=ok`.

Analysis:
- This is the same auth/session failure pattern as the previous blocker: Della
  is reachable, but noninteractive login is rejected. The remote jobs and
  remote submitters should continue independently, but local Codex cannot
  monitor or intervene until Della auth is refreshed.

Next:
- Refresh local authentication to `della-gpu`.
- Resume bounded polling of queue, overfit loss, cache counts, submitter logs,
  and error scans.
- Deploy this worklog entry to Della after SSH is restored.

## 2026-06-10 01:21 PDT - Monitor Loop Resumed Without BatchMode

Goal:
- Continue monitoring the active overfit job, DROID cache submitter, and
  post-cache training waiter after confirming that direct `ssh della-gpu`
  works from the user's terminal.

Hypothesis:
- The previous local monitor failures were caused by forcing
  `BatchMode=yes`, which disables the keyboard-interactive path used by the
  working Della SSH config.

Change:
- No code change. Changed monitor invocation practice: use normal
  `ssh della-gpu ...` rather than `ssh -o BatchMode=yes della-gpu ...`.

Version Control:
- branch: `main`
- base_commit: `a4cd8a99f45943ce50fe04b5e7ea90fec716b0fc`
- implementation_commit: pending worklog-only update
- push/pull: deployed `a4cd8a99f45943ce50fe04b5e7ea90fec716b0fc` to Della
  before monitoring
- changed_files:
  - `WORKLOG.md`
- remote_commit/status:
  - remote commit: `a4cd8a99f45943ce50fe04b5e7ea90fec716b0fc`
  - remote tracked status clean; untracked scratch/model directories remain
    `6/`, `Wan2.2-TI2V-5B/`

Command / Job:
- command:
  - `ssh -o ConnectTimeout=20 della-gpu 'hostname && date'`
  - `ssh -G della-gpu` to inspect auth/control settings
  - `scripts/della_loop.sh deploy-code main`
  - two bounded local monitor loops polling queue, cache counts, overfit
    `train_log.csv`, submitter/waiter tails, and error scans
  - `rsync` fetched only small CSV/log artifacts for local plots
- job_id:
  - overfit: `9478714`
  - completed cache array: `9497106` for shards `265-288`
  - current cache array: `9497638` for shards `289-312`
  - remote cache submitter PID: `1871034`
  - remote DROID training waiter PID: `1883834`
- run_dir: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs:
  - `slurm_outputs/action-overfit/out_act-side-fresh25-10k_9478714.log`
  - `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_121.log`
  - `runs/droid_window_plans/action_droid_training_submitter_resume.log`
- artifacts:
  - `_cluster/loss_curves/side_fresh25_10k_loss_current.svg`
  - `_cluster/loss_curves/droid_cache_progress_current.svg`
  - `_cluster/loss_curves/current_monitor_summary.json`

Result:
- status: active and healthy
- metrics/artifacts:
  - `ssh della-gpu` succeeded without `BatchMode`; `ssh -G` showed
    `kbdinteractiveauthentication yes`, `passwordauthentication yes`,
    `controlmaster auto`, and control path
    `/Users/lzha/.ssh/sockets/22-della-gpu.princeton.edu-lz3952`.
  - overfit `9478714` remains running on `della-i23g1`; latest local fetched
    log reached step `4840/10000`, loss `0.0643683`.
  - recent-25 loss mean is inflated by one prior spike at step `4620`, but
    recent-25 median is `0.0624064` and clean mean excluding loss `>0.2` is
    `0.0625621`.
  - cache advanced from about `393,204` to direct count `408,643` during the
    monitor window; waiter log reached `400,666/1,440,554` train windows and
    `14,636/14,636` val windows.
  - cache array `9497106` completed cleanly and the submitter launched
    `9497638` for shards `289-312`.
- key evidence:
  - submitter log line:
    `[2026-06-10 04:17:14] job 9497106 completed cleanly`.
  - no current traceback/OOM/failure lines appeared in the error scan.

Analysis:
- The Della auth issue was self-inflicted by `BatchMode=yes`; future monitor
  commands should use the normal SSH config so keyboard-interactive and
  ControlMaster can work.
- The overfit training still does not call for an LR/batch-size change. Spikes
  are occasional and recover immediately under gradient clipping. The robust
  loss statistics remain flat near `0.062`, so changing LR mid-run would reduce
  interpretability of this 10k-step overfit test.
- Cache generation is progressing monotonically under the `gpu-test` array
  limit. Disk free space was about `1.2T`, still enough for the planned
  remaining cache at the current scale.

Next:
- Continue bounded polling of `9478714`, `9497638`, cache counts, and error
  scans.
- If overfit completes, generate/fetch random-noise eval videos and record
  final paths.
- If cache reaches the train target, verify the waiter submits DROID smoke/full
  training and monitor those curves.

## 2026-06-10 01:30 PDT - Current-Cache DROID Training Submitted

Goal:
- Start DROID side-adapter training using the cache available now, without
  waiting for the full planned `1,440,554` train-window cache.

Hypothesis:
- A fixed snapshot manifest over currently materialized cache directories lets
  training start safely while the cache submitter continues in the background.

Change:
- No code change. Created a remote snapshot manifest from existing train cache
  directories older than two minutes to avoid directories being actively
  written by cache jobs.

Version Control:
- branch: `main`
- base_commit: `3ded7bd2add2bc1a29b9390b987d02549db43f54`
- implementation_commit: pending worklog-only update
- push/pull: code already deployed to Della at `3ded7bd2add2bc1a29b9390b987d02549db43f54`
- changed_files:
  - `WORKLOG.md`
- remote_commit/status:
  - remote commit at submission: `3ded7bd2add2bc1a29b9390b987d02549db43f54`

Command / Job:
- command:
  - create snapshot manifest from
    `data/droid_cache_windows_v0/train` directories with `mtime > 2min`
  - submit smoke:
    `sbatch --time=00:30:00 --job-name=act-droid-cur-smoke ... run_action_conditioned_droid.sh`
  - submit 10k full run:
    `sbatch --time=36:00:00 --dependency=afterok:9497851 --job-name=act-droid-cur-10k ... run_action_conditioned_droid.sh`
- job_id:
  - smoke: `9497851`
  - full 10k: `9497852`, dependency `afterok:9497851`
  - still-running overfit occupying `ailab` slot: `9478714`
- run_dir: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- logs:
  - `slurm_outputs/action-droid/out_act-droid-cur-smoke_9497851.log`
  - `slurm_outputs/action-droid/err_act-droid-cur-smoke_9497851.log`
  - `slurm_outputs/action-droid/out_act-droid-cur-10k_9497852.log`
  - `slurm_outputs/action-droid/err_act-droid-cur-10k_9497852.log`
- artifacts:
  - snapshot manifest:
    `runs/droid_window_plans/train_current_cache_414672_20260610_042850.jsonl`
  - snapshot summary:
    `runs/droid_window_plans/train_current_cache_414672_20260610_042850.summary.json`
  - smoke output:
    `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_smoke`
  - full output:
    `runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k`

Result:
- status: submitted, pending on scheduler
- metrics/artifacts:
  - snapshot manifest contains `414,835` train names.
  - total train cache count at snapshot was about `415,751`; validation cache
    was complete at `14,636`.
  - smoke job `9497851` is pending with reason `QOSMaxJobsPerUserLimit` because
    overfit job `9478714` is still running in `ailab`.
  - full job `9497852` is pending with reason `Dependency`, as intended.
- key evidence:
  - `squeue -j 9497851,9497852,9478714` showed:
    `9497851 PENDING (QOSMaxJobsPerUserLimit)`,
    `9497852 PENDING (Dependency)`,
    `9478714 RUNNING`.

Analysis:
- The initial checked snapshot implementation used Python `Path.exists()` over
  hundreds of thousands of directories and was too slow on GPFS, so it was
  killed and replaced by a `find`-based snapshot. This avoids a code change and
  is adequate because old directory mtimes exclude actively written cache
  entries.
- The training is started from Slurm's perspective but cannot begin execution
  until the current `ailab` QOS slot frees or the user chooses to cancel the
  overfit job. Keeping the dependency protects the full run from launching if
  the smoke fails.

Next:
- Monitor `9497851`; when it starts, inspect stdout/stderr and verify five
  smoke steps plus validation complete cleanly.
- If smoke completes, confirm `9497852` starts and monitor loss/validation.
- Decide whether to let overfit `9478714` finish or cancel it to prioritize the
  current-cache DROID training.

## Comparison Summary

### Noise / Z-Init Experiments

| Attempt | Key setting | Result | Evidence | Decision |
| --- | --- | --- | --- | --- |
| existing zinit probe | 115 existing artifacts | no clear global manifold; deterministic duplicates identical | different-sample cosine mean `0.010`, RMSE `1.165` | probe stochastic/multi-sample |
| basin replay | own vs fresh/cross z_init | wrong noise fails badly | own MSE `0.015-0.026`; fresh often `2-3+`; cross `~0.8-1.8` | initial basin matters |
| stochastic one-sample | ep0, seeds 0-7, eta 0/0.3/0.6 | pipeline works, sample too small | 24 endpoints | expand |
| dataset pilot | eps 0-3, eta 0.3 | same-sample closer than different-sample | same RMSE `0.713`, diff RMSE `1.301` | promising but small |
| 10k DDIM | eta 0, 10k samples | high-dimensional, no obvious low-rank structure | PC100 only `10.45%` explained | do not prioritize z_init prediction |

### Adapter Experiments

| Attempt | Key setting | Result | Evidence | Decision |
| --- | --- | --- | --- | --- |
| pre-context 1-step | fixed noise, 1k | works but weak | final eval `0.0815` | keep as ablation |
| side late/default | fixed noise | better than pre-context | final eval `0.0400-0.0475` | side stronger |
| side layers 15-29 | fixed noise | improves | final eval `0.0143` at 2k | more layers helps |
| side layers 0-29 | fixed noise | best one-step | final eval `0.00810` at 4k | use all layers |
| side 25-step fresh pilot | all layers, 100 steps | stable random eval | random eval `0.0986-0.1001` | launch 10k |
| side 25-step fresh 10k | all layers, 10k | pending maintenance | job `9478714` | monitor after maintenance |

### DROID Full Training Pipeline

| Attempt | Key setting | Result | Evidence | Decision |
| --- | --- | --- | --- | --- |
| high-action plan | view 0, 33 frames, stride 4, fp16 | feasible under cap | train `1,440,554`, val `14,636`, est `~325 GiB` | materialize cache |
| val cache | fp16 windows | passed | `14,636` complete, `3.3G` | train cache next |
| train cache ailab | 128 shards | partially passed then maintenance | `135,051+` windows, no skips/fails | reroute |
| train cache gputest | 1024 shards | viable but maintenance blocked | pilot `1407` in `5:26`; count `140,679` | resume later |
| trainer gputest | 40 GB A100 | OOM | job `9479259` | do not train on 40 GB |
| trainer H200 | 1-step smoke | passed | job `9479332`, train loss `2.033`, val `2.903` | full train after cache |

## Immediate Resume Checklist

See `HANDOFF.md` for the concise handoff. At minimum:

1. Refresh SSH/auth with normal `ssh della-gpu`; do not force `BatchMode=yes`.
2. Verify remote commit and queue:
   `cd /scratch/gpfs/AM43/lz3952/Wan2.2 && git rev-parse HEAD && squeue -u $(whoami)`.
3. Inspect jobs:
   - `9478714` overfit 10k
   - `9497851` current-cache DROID smoke
   - `9497852` dependent current-cache DROID 10k
   - current `droid-train-cache-*` array after `9497638`
4. Check cache counts under `data/droid_cache_windows_v0/{train,val}`.
5. Tail logs:
   - `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5/train_log.csv`
   - `slurm_outputs/action-droid/out_act-droid-cur-smoke_9497851.log`
   - `slurm_outputs/action-droid/err_act-droid-cur-smoke_9497851.log`
   - `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_121.log`
   - `runs/droid_window_plans/action_droid_training_submitter_resume.log`
6. If `9478714` completed, export/fetch random-noise eval videos and record
   final paths.
7. If `9497851` passed, monitor `9497852`. If `9497851` failed, debug before
   allowing the full run to proceed.

## 2026-06-11 03:15 PDT - H200 DROID batch-size probe and step9000 eval

Goal:
- Determine the largest real DROID action-conditioned rollout training batch
  that fits the current single-H200 allocation, and keep the active eval/cache
  loop documented.

Hypothesis:
- The current code path was artificially limited to `batch_size=1`; after
  vectorizing the closed-loop rollout over Wan's list-of-samples API, the H200
  may fit a larger batch, but the exact limit needs an optimizer-step probe.

Change:
- Added batched rollout support in `scripts/train_action_conditioned_wan.py`.
- Updated DROID noise creation in
  `scripts/train_action_conditioned_wan_droid.py` to return one `z_init` per
  sample.
- Added `scripts/profile_action_droid_batch_size.py` for one-step GPU memory
  profiling.
- Added `DROID_BATCH_SIZE` to
  `scripts/wait_and_submit_action_droid_training.sh` so the full-cache waiter
  can submit future smoke/full training with an explicit batch size.

Version Control:
- agent_id: main orchestrator
- worktree: `/home/lzha/code/Wan2.2`
- probe_worktree: `/home/lzha/code/Wan2.2-batchsize-probe`
- branch: `main`; probe branch `codex/batchsize-probe-20260611`
- base_commit: `2e67e07`
- probe_commits: `3154e5f`, `b12e415`
- implementation_commit: pending
- remote_probe_commit: `b12e415` at
  `/scratch/gpfs/AM43/lz3952/worktrees/Wan2.2/codex-batchsize-probe-20260611`
- changed_files:
  - `scripts/train_action_conditioned_wan.py`
  - `scripts/train_action_conditioned_wan_droid.py`
  - `scripts/profile_action_droid_batch_size.py`
  - `scripts/wait_and_submit_action_droid_training.sh`
  - `WORKLOG.md`

Command / Job:
- validation:
  - `python3 -m py_compile scripts/train_action_conditioned_wan.py scripts/train_action_conditioned_wan_droid.py scripts/profile_action_droid_batch_size.py`
  - `bash -n scripts/wait_and_submit_action_droid_training.sh`
- profile jobs:
  - `9540013` failed before memory measurement due missing autocast in the
    new batched unconditional path.
  - `9540057` reran high-to-low batch sizes on an H200.
- profile result JSONL:
  `_cluster/batch_size_probe_h200/profile_h200_current_shape_9540057_20260611_060506.jsonl`
- step9000 eval:
  - job: `9539937`
  - local videos:
    `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step9000_ep399_v0_s00004_s1000_1001/`
  - viz-open:
    `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step9000_ep399_v0_s00004_s1000_1001`

Result:
- status: passed
- batch-size profile:
  - `batch_size=8`: OOM, PyTorch allocated `137.85 GiB`.
  - `batch_size=7`: OOM, PyTorch allocated `136.97 GiB`.
  - `batch_size=6`: OOM, PyTorch allocated `137.14 GiB`.
  - `batch_size=5`: passed, max allocated `133,937.6 MiB`, max reserved
    `135,538.0 MiB`, free after step `6,851.1 MiB`.
- conclusion: exact max-fit batch size for the current H200/25-step side
  adapter setup is `5`; `4` is the conservative option with more headroom.
- step9000 eval:
  - validation loss improved to `0.19321248633787036`.
  - seed1000 latent MSE `0.21804796159267426` vs null `1.9798624515533447`.
  - seed1001 latent MSE `0.22290533781051636` vs null `4.661829471588135`.
  - videos validate at 320x192, 33 frames, 16 FPS.

Analysis:
- Batch 5 fits but is tight. Batch 6 fails at essentially full H200 memory, so
  using 5 is a max-fit choice and 4 is safer against allocator fragmentation.
- The step9000 videos still preserve scene/table/object layout better than
  null rollouts, but the gripper/robot region blurs into a cloudy smear by
  mid/late frames. This remains consistent with global latent loss being
  dominated by static background and the action adapter not yet learning sharp
  moving robot geometry.

Next:
- Commit/push the batch-size implementation and worklog.
- Update the canonical Della checkout to the committed code.
- Restart only the full-cache waiter with `DROID_BATCH_SIZE=5` before the
  cache reaches `1,440,554/1,440,554`; leave active training job `9497852` and
  cache submitter jobs untouched.
- Continue monitoring current DROID training through step10000 and fetch/inspect
  the final eval videos.

## 2026-06-11 03:25 PDT - Resume cache submitter after stale sacct rows

Goal:
- Keep the full DROID train cache moving after the previous chunk submitter
  exited on stale `sacct RUNNING` rows.

Hypothesis:
- Array `9539503` shards `982-984` actually completed cleanly; `sacct` was
  stale while `scontrol show job` and shard logs had the authoritative terminal
  state.

Change:
- Hardened `scripts/submit_droid_train_cache_chunks.sh` and
  `scripts/wait_and_submit_action_droid_training.sh` to accept
  `sacct RUNNING 0:0` rows only when `scontrol show job <row>` reports
  `JobState=COMPLETED` and `ExitCode=0:0`.
- Replaced the old full-cache waiter PID `1883834` with PID `298171` using
  `DROID_BATCH_SIZE=5` and
  `LOG_PATH=runs/droid_window_plans/action_droid_training_submitter_resume_bs5.log`.

Version Control:
- base_commit: `a58bab8`
- implementation_commit: pending
- changed_files:
  - `scripts/submit_droid_train_cache_chunks.sh`
  - `scripts/wait_and_submit_action_droid_training.sh`
  - `WORKLOG.md`

Command / Job:
- validation:
  - `bash -n scripts/submit_droid_train_cache_chunks.sh scripts/wait_and_submit_action_droid_training.sh`
- stale chunk checked:
  - `scontrol show job 9539503_982`
  - `scontrol show job 9539503_983`
  - `scontrol show job 9539503_984`
- replacement waiter:
  - `nohup env POLL_SECONDS=900 DROID_BATCH_SIZE=5 LOG_PATH=runs/droid_window_plans/action_droid_training_submitter_resume_bs5.log bash scripts/wait_and_submit_action_droid_training.sh ...`

Result:
- `9539503_982`, `_983`, and `_984` are `COMPLETED ExitCode=0:0` by
  `scontrol`; their logs end with `done ok=1407 skip=0 fail=0`.
- Replacement waiter log starts with
  `waiting for cache train=1440554 val=14636 batch_size=5`.
- Cache is not complete yet; the previous waiter count was
  `1,381,851/1,440,554` at 06:07 EDT.

Analysis:
- The cache submitter failed from scheduler-accounting staleness, not from a
  data/cache failure. Resuming at shard `985` avoids repeating completed shard
  work.

Next:
- Commit/push the stale-accounting patch, update Della, and relaunch the cache
  submitter from `START_SHARD=985` with the same gputest settings.

## 2026-06-11 04:40 PDT - Full DROID train cache complete

Goal:
- Finish the remaining DROID train-cache shards so the batch-size-5 full-cache
  training waiter can launch the smoke/full training sequence.

Hypothesis:
- Resuming from `START_SHARD=985` after the stale `sacct` fix should complete
  shards `985-1023` without recomputing already-finished shards.

Change:
- No code change in this attempt. Continued monitoring the patched submitter
  from commit `b5b18e9`.

Version Control:
- implementation_commit: `b5b18e9`
- remote_commit/status: `/scratch/gpfs/AM43/lz3952/Wan2.2` at `b5b18e9`
- changed_files: `WORKLOG.md`

Command / Job:
- cache submitter PID: `322448`
- cache submitter log:
  `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024_resume_from_985_robust_20260611_062500.log`
- final cache array: `9540854`
- final shard range: `1009-1023`
- batch-5 waiter PID: `318322`
- batch-5 waiter log:
  `runs/droid_window_plans/action_droid_training_submitter_resume_bs5.log`

Result:
- status: passed
- verified cache counts: train `1,440,554/1,440,554`, val
  `14,636/14,636`.
- `sacct -j 9540854` reports all shard tasks `COMPLETED` with `ExitCode=0:0`.
- shard logs for `1009-1023` end with `done ok=... skip=0 fail=0`.
- submitter logged `job 9540854 completed cleanly` and
  `all train-cache chunks submitted and completed`.

Analysis:
- The cache pipeline is complete; the earlier blockage was scheduler accounting
  staleness, not data failure. The remaining gate is the batch-5 waiter polling
  the completed counts and submitting the full-cache smoke job.

Next:
- Monitor PID `318322` until it submits the `DROID_BATCH_SIZE=5` full-cache
  smoke job, then inspect that job's launch config and training logs before
  allowing the full 10k batch-5 run to proceed.
- Continue monitoring active current-cache job `9497852` through final
  validation/eval at step `10000`.

## 2026-06-11 04:55 PDT - Guard full-cache waiter job-id parsing

Goal:
- Prevent the full-cache waiter from launching the full DROID run before the
  smoke job actually completes.

Hypothesis:
- `submit_sbatch` was logging the human-readable sbatch output to stdout inside
  command substitution, so `smoke_job` contained both `Submitted batch job ...`
  and the numeric job id. That made the wait/sacct checks ineffective.

Change:
- Updated `scripts/wait_and_submit_action_droid_training.sh` to call
  `sbatch --parsable`, validate numeric Slurm job ids, retry briefly for the
  primary `sacct` row after a job leaves `squeue`, and fail if no primary
  accounting row exists.
- Canceled accidentally-launched full batch-5 job `9541650` before any training
  step logged.

Version Control:
- base_commit: `b5b18e9`
- implementation_commit: pending
- changed_files:
  - `scripts/wait_and_submit_action_droid_training.sh`
  - `WORKLOG.md`

Command / Job:
- validation: `bash -n scripts/wait_and_submit_action_droid_training.sh`
- smoke job: `9541649`
- canceled premature full job: `9541650`

Result:
- `9541650` was canceled after launch logging only; `sacct` reports
  `CANCELLED by 363214`.
- `9541649` is running with `batch_size=5`, `total_steps=5`,
  `max_train_samples=128`, and output
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_smoke_bs5`.

Analysis:
- The batch-5 code path is still being tested by the smoke job. The incorrect
  waiter success was an orchestration bug, not a training result.

Next:
- Commit/push the waiter guard, update the Della checkout, inspect smoke logs
  to completion, and submit the full batch-5 run only after the smoke job has
  real `COMPLETED 0:0` accounting plus train/val logs.

## 2026-06-11 05:00 PDT - Batch-5 smoke passed and full run submitted

Goal:
- Validate the full-cache batch-size-5 training path and launch the real 10k
  full-cache run only after a clean smoke.

Hypothesis:
- The profiler's max-fit batch size `5` should also run in the real DROID
  full-cache trainer when using the fixed batched rollout/autocast code.

Change:
- Deployed commit `8c6cc5d` to the canonical Della checkout.
- Manually submitted the full batch-5 training job after smoke completion,
  because the prior waiter process had already exited after the job-id parsing
  bug.

Version Control:
- implementation_commit: `8c6cc5d`
- push/pull: pushed to `origin/main` and fast-forwarded
  `/scratch/gpfs/AM43/lz3952/Wan2.2` to `8c6cc5d`.
- changed_files: `WORKLOG.md`

Command / Job:
- smoke job: `9541649`
- premature full job canceled: `9541650`
- full batch-5 job: `9541718`
- smoke run dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_smoke_bs5`
- full run dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5`

Result:
- smoke status: passed
- `sacct -j 9541649` reports `COMPLETED 0:0`.
- smoke launch confirmed `batch_size=5`, `total_steps=5`,
  `max_train_samples=128`, `max_val_samples=4`.
- smoke train losses: `2.9840`, `3.5863`, `4.6761`, `4.7428`, `4.2348`.
- smoke validation: `val_loss=4.170713365077972`, `n_val=4`.
- full batch-5 job `9541718` is submitted and initially pending.

Analysis:
- The smoke proves the real batched DROID trainer can execute batch size `5`
  through training and validation. The high smoke losses are expected from a
  five-step fresh full-cache smoke and are not a convergence signal.

Next:
- Monitor `9541718` until it starts, confirm launch config and early train
  rows, then continue through validation/eval checkpoints.
- Continue monitoring current-cache job `9497852` to step `10000` and inspect
  the final periodic eval videos.

## 2026-06-11 05:35 PDT - Current-cache DROID final eval and batch-5 walltime risk

Goal:
- Finish and inspect the step-10000 eval for current-cache DROID training, and
  keep the full-cache batch-size-5 run under active monitoring.

Hypothesis:
- The current-cache run should show at least stable final validation/eval
  behavior versus step 9000, while the full-cache batch-5 run should continue
  past startup without OOM.

Change:
- No source change. Manually submitted the missing step-10000 eval after the
  periodic watcher was no longer alive.

Version Control:
- implementation_commit: `1809c29`
- changed_files:
  - `WORKLOG.md`
  - `HANDOFF.md`

Command / Job:
- completed current-cache train job: `9497852`
- manual step-10000 eval job: `9542335`
- active full-cache batch-5 train job: `9541718`
- eval command:
  `sbatch --parsable --job-name=act-droid-eval-s10000 ... scripts/export_action_conditioned_wan_video.py --ckpt_path runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/ckpt_latest.pt --triplets_root data/droid_cache_windows_v0/val --overfit_one ep399_v0_s00004 --output_dir runs/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step10000_ep399_v0_s00004_s1000_1001 --eval_noise_mode random --eval_seed_start 1000 --num_eval_noises 2 --include_null`
- local eval artifacts:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step10000_ep399_v0_s00004_s1000_1001/`
- viz-open:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_currentcache_414835_20260610_042850_10k/videos_latest_step10000_ep399_v0_s00004_s1000_1001`

Result:
- current-cache train job `9497852`: `COMPLETED 0:0`.
- final validation improved slightly to `0.19281196547672153` at step 10000
  from `0.19321248633787036` at step 9000.
- step-10000 eval job `9542335`: `COMPLETED 0:0`.
- eval videos validated at 320x192, 33 frames, 16 FPS.
- step-10000 eval metrics:
  - seed1000 latent MSE `0.22305864095687866` vs null
    `1.9798624515533447`.
  - seed1001 latent MSE `0.2230650782585144` vs null
    `4.661829471588135`.
- visual inspection: samples preserve the table/object layout and are far
  better than null, but the robot/gripper region still becomes a gray cloudy
  smear by mid/end frames. Step 10000 does not visually fix the step-9000 blur.
- full-cache batch-5 job `9541718` is running on `della-i21g3`; early train
  rows through step `120` are logged with no OOM or traceback.

Analysis:
- Current-cache training made a small validation gain from 9000 to 10000, but
  held-out random-noise eval did not improve and qualitatively still has the
  same moving-robot blur. This supports the earlier diagnosis that global
  latent loss is learning static scene structure much better than sharp moving
  robot geometry.
- Batch size `5` is operationally viable on H200, but at the observed
  full-cache speed of about `13.6s/step`, a 10k run likely needs roughly
  38 hours. The submitted job has a 36-hour time limit, and `scontrol update`
  to 48 hours was denied.

Next:
- Await user confirmation before adding DROID trainer resume support
  (`RESUME_CKPT`, load model/step, optional optimizer restore, and save
  optimizer for future checkpoints).
- Continue monitoring `9541718`; first major checkpoint/validation is expected
  at step `1000`.

## 2026-06-11 08:55 PDT - Full-cache batch-5 first validation

Goal:
- Verify that the full-cache batch-size-5 run survives through its first
  checkpoint and validation.

Hypothesis:
- If the batch-size patch is sound beyond smoke, the first 1000 steps should
  complete without OOM/NaN and write both latest and best-val checkpoints.

Change:
- No source change. Continued read-only monitoring of job `9541718`.

Version Control:
- implementation_commit: `887bfd1`
- changed_files:
  - `WORKLOG.md`
  - `HANDOFF.md`

Command / Job:
- active job: `9541718` `act-droid-win-10k`
- run dir:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5`
- logs:
  `slurm_outputs/action-droid/out_act-droid-win-10k_9541718.log`,
  `slurm_outputs/action-droid/err_act-droid-win-10k_9541718.log`

Result:
- status: passed first validation; job still running
- step `1000` train loss: `0.3422377109527588`
- step `1000` val loss: `0.2770900116302073` over `32` val samples
- job continued to at least step `1020`.
- `ckpt_latest.pt` and `ckpt_best_val.pt` were written at step 1000, both
  about `914M`.
- stderr/error scan has no traceback, OOM, or NaN hits.

Analysis:
- Batch size `5` is viable beyond smoke and through checkpoint/validation.
  Early full-cache validation is higher than the old current-cache run's late
  validation, as expected for a fresh run at only 1000 steps on the much larger
  cache.
- The walltime concern remains: 36 hours is likely short for 10k steps at the
  observed `~13.6s/step`, and Slurm denied extending the running job.

Next:
- Continue monitoring `9541718`; next validation/checkpoint is step `2000`.
- Await user confirmation before implementing resume support.

## 2026-06-11 10:05 PDT - Full-cache batch-5 step-1000 eval

Goal:
- Run and inspect the first periodic eval for the full-cache batch-size-5
  training run.

Hypothesis:
- At step 1000, the full-cache run should be better than null but likely worse
  than the completed current-cache 10k run because it is still early in
  training.

Change:
- No source change. Copied `ckpt_latest.pt` to `ckpt_step1000.pt` before eval
  so the eval job could not accidentally load a later checkpoint.

Version Control:
- implementation_commit: `b67341c`
- changed_files:
  - `WORKLOG.md`
  - `HANDOFF.md`

Command / Job:
- eval job: `9551286`
- checkpoint:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/ckpt_step1000.pt`
- remote videos:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step1000_ep399_v0_s00004_s1000_1001/`
- local videos:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step1000_ep399_v0_s00004_s1000_1001/`
- viz-open:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step1000_ep399_v0_s00004_s1000_1001`

Result:
- status: passed
- `sacct -j 9551286` reports `COMPLETED 0:0`.
- videos validate at 320x192, 33 frames, 16 FPS.
- metrics:
  - seed1000 latent MSE `0.29633891582489014` vs null
    `1.9798624515533447`.
  - seed1001 latent MSE `0.2788466513156891` vs null
    `4.661829471588135`.
- contact sheet:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step1000_ep399_v0_s00004_s1000_1001/droid_full_bs5_step1000_eval_contact_sheet.jpg`

Analysis:
- The eval is clearly better than null, but visually worse than the completed
  current-cache 10k eval. The table layout is recognizable, but mid/end frames
  show stronger ghosting and object/robot smearing. This is acceptable for an
  early step-1000 checkpoint on the full 1.44M-window cache.

Next:
- Continue monitoring `9541718` and launch another eval at step `2000`.

## 2026-06-11 14:35 PDT - Full-cache batch-5 step-2000 eval

Goal:
- Validate the second full-cache batch-size-5 checkpoint and compare it with
  step 1000.

Hypothesis:
- If training is progressing, step-2000 validation and fixed eval MSE should
  improve versus step 1000, though qualitative robot smear may remain.

Change:
- No source change. Copied `ckpt_latest.pt` to `ckpt_step2000.pt` before eval.

Version Control:
- implementation_commit: `841aa65`
- changed_files:
  - `WORKLOG.md`
  - `HANDOFF.md`

Command / Job:
- eval job: `9557082`
- checkpoint:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/ckpt_step2000.pt`
- remote videos:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step2000_ep399_v0_s00004_s1000_1001/`
- local videos:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step2000_ep399_v0_s00004_s1000_1001/`
- viz-open:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step2000_ep399_v0_s00004_s1000_1001`

Result:
- training validation improved from `0.2770900116302073` at step 1000 to
  `0.24543552426621318` at step 2000.
- eval job `9557082` completed with `ExitCode=0:0`.
- videos validate at 320x192, 33 frames, 16 FPS.
- metrics:
  - seed1000 latent MSE `0.27191445231437683` vs null
    `1.9798624515533447`.
  - seed1001 latent MSE `0.2750750482082367` vs null
    `4.661829471588135`.
- contact sheet:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step2000_ep399_v0_s00004_s1000_1001/droid_full_bs5_step2000_eval_contact_sheet.jpg`

Analysis:
- Step 2000 is a real numeric improvement over step 1000, especially seed1000
  (`0.2963 -> 0.2719`). Qualitatively, the same robot/gripper haze remains in
  mid/end frames; the table/object layout is recognizable and null remains much
  worse.

Next:
- Continue monitoring `9541718`; next validation/eval target is step `3000`.

## 2026-06-11 15:15 PDT - DROID DDP trainer implementation

Goal:
- Add multi-GPU training support for the DROID action-conditioned Wan trainer
  and prepare an 8-GPU Della launch path without disturbing the active
  single-GPU batch-5 run.

Hypothesis:
- The current trainer can use DDP with minimal risk because the Wan backbone is
  frozen and each rank can run the same local closed-loop rollout while DDP
  synchronizes only the trainable adapter gradients.

Change:
- Added torchrun/NCCL initialization from `WORLD_SIZE`, `RANK`, and
  `LOCAL_RANK`.
- Set the local CUDA device before constructing Wan, and added a rank-aware
  Wan pipeline builder so ranks do not all load onto GPU 0.
- Added `DistributedSampler` for training, DDP wrapping for
  `ActionConditionedWanModel`, rank-0-only config/log/validation/checkpoint
  writes, and synchronization barriers around validation/checkpoint events.
- Added `run_action_conditioned_droid_dist.sh`, a Della launcher for
  one-node multi-GPU DROID training using `torchrun`.

Version Control:
- agent_id: orchestrator plus worker `019eb8ba-cc6f-7fc0-96df-29ceb166dfd2`
- branch: `codex/droid-ddp-8gpu`
- base_commit: `f37022874c588817d4ed77d463e3d27745053df4`
- implementation_commit: `dd6f1c829a968f00c47947d835a9e6ee1f36d127`
- changed_files:
  - `scripts/train_action_conditioned_wan_droid.py`
  - `run_action_conditioned_droid_dist.sh`
  - `WORKLOG.md`
  - `HANDOFF.md`

Command / Job:
- local static checks:
  `git diff --check -- scripts/train_action_conditioned_wan_droid.py run_action_conditioned_droid_dist.sh`
  `bash -n run_action_conditioned_droid_dist.sh`
  `bash -n run_action_conditioned_droid.sh`
  `/usr/bin/python3 -m py_compile scripts/train_action_conditioned_wan_droid.py`
- Slurm route check:
  `sbatch --test-only --account=am43 --partition=ailab --qos=ailab --nodes=1 --gres=gpu:8 --cpus-per-task=64 --mem=720G --time=00:20:00 --job-name=dist-smoke-test --wrap=hostname`

Result:
- status: implementation/static-validation passed; cluster smoke pending.
- Static checks passed locally.
- Della accepted the 8-GPU allocation shape, with an estimated start around
  `2026-06-12T23:38:58` at the time of the test-only probe.
- Active single-GPU run `9541718` was left untouched and continued running.

Analysis:
- The patch preserves single-process behavior when `WORLD_SIZE=1`.
- The main remaining risk is runtime DDP behavior with the large frozen Wan
  module wrapped inside DDP; a short multi-GPU smoke is required before the
  full 8-GPU training run.

Next:
- Commit and push the DDP branch, deploy the exact commit to an isolated Della
  worktree, run a short distributed smoke, inspect logs/checkpoints, then submit
  the full 8-GPU training run if the smoke is clean.

## 2026-06-11 15:27 PDT - DDP smoke and 8-GPU training queue

Goal:
- Queue a safe distributed-training launch chain that validates DDP before any
  full 8-GPU training starts.

Hypothesis:
- A one-step, one-diffusion-step smoke is enough to catch rank/device/DDP
  wiring problems before committing 8 H200s to the 25-step full run.

Change:
- No source change after commit `f284b18340e1d111bcb30b31fd07a4ed8da0ecfc`.
- Created isolated Della worktree:
  `/scratch/gpfs/AM43/lz3952/worktrees/Wan2.2/codex-droid-ddp-8gpu`
  at commit `f284b18340e1d111bcb30b31fd07a4ed8da0ecfc`.
- Linked untracked runtime assets from the canonical checkout:
  `.venv`, `Wan2.2-TI2V-5B`, and `data`.

Version Control:
- branch: `codex/droid-ddp-8gpu`
- launch_commit: `f284b18340e1d111bcb30b31fd07a4ed8da0ecfc`
- remote_commit/status: detached at `f284b18340e1d111bcb30b31fd07a4ed8da0ecfc`;
  untracked symlinks for runtime assets only.

Command / Job:
- remote validation:
  - `bash -n run_action_conditioned_droid_dist.sh`
  - `.venv/bin/python -m py_compile scripts/train_action_conditioned_wan_droid.py`
  - manual invocation of all `tests/test_action_conditioned_wan.py::test_*`
    functions because `pytest` is not installed in the Della venv.
- queued Slurm chain:
  - 2-GPU smoke job `9564218`, output
    `runs/action_droid_dist_ddp2_smoke_f284b18`
  - 8-GPU smoke job `9564219`, dependency `afterok:9564218`, output
    `runs/action_droid_dist_ddp8_smoke_f284b18`
  - 8-GPU full training job `9564220`, dependency `afterok:9564219`, output
    `runs/action_droid_dist_side_bn512h8_L0-29_fresh_25step_fullcache_10k_lr5e-5_bs5x8_ddp_f284b18`

Result:
- status: queued
- Remote validation passed. Manual focused tests passed `11/11`.
- Slurm accepted all jobs. Initial queue state:
  - `9564218` pending, reason `(None)`
  - `9564219` pending, reason `(Dependency)`
  - `9564220` pending, reason `(Dependency)`
- Route probes estimated short smoke starts around `2026-06-13` and long
  8-GPU full-duration starts around `2026-06-18`, but live queue should be
  rechecked rather than trusted.

Analysis:
- Submitting the dependent full run now preserves queue position without
  allowing it to start before DDP smoke validation.
- The 8-GPU run uses local batch `5`, global batch `40`, LR `5e-5`, 10k
  optimizer steps, 25 diffusion steps, and full train/val manifests.

Next:
- Monitor jobs `9564218`, `9564219`, and `9564220`.
- If smoke fails, inspect logs, patch on a new commit/worktree, and cancel or
  replace dependent jobs.
- Continue monitoring active single-GPU run `9541718`; launch/fetch step-3000
  eval when its checkpoint appears.

## 2026-06-11 15:58 PDT - DDP 2-GPU smoke passed, barrier warning fix

Goal:
- Inspect the first distributed smoke and remove any warning that could become
  a larger-world-size hang risk before launching the 8-GPU smoke.

Hypothesis:
- The DDP trainer works functionally, but the NCCL barrier should specify the
  rank-local CUDA device to avoid ambiguous device mapping warnings.

Change:
- Updated the distributed barrier helper to pass
  `device_ids=[torch.cuda.current_device()]` when CUDA is available.
- Canceled pending old jobs `9564219` and `9564220` so replacement 8-GPU jobs
  can run from a new immutable commit.

Version Control:
- branch: `codex/droid-ddp-8gpu`
- base_commit: `062743f`
- implementation_commit: `f978882964601df9be761a12a5baf6f0db24bc1f`
- changed_files:
  - `scripts/train_action_conditioned_wan_droid.py`
  - `WORKLOG.md`

Command / Job:
- completed smoke job: `9564218`
- canceled old dependent jobs: `9564219`, `9564220`
- local checks:
  `git diff --check -- scripts/train_action_conditioned_wan_droid.py`
  `/usr/bin/python3 -m py_compile scripts/train_action_conditioned_wan_droid.py`

Result:
- status: 2-GPU smoke passed; barrier patch static-check passed.
- `sacct -j 9564218` reports `COMPLETED 0:0`, elapsed `00:01:55`.
- Smoke output:
  - distributed `True`, world size `2`
  - local batch `1`, global batch `2`
  - one training step loss `0.306015`
  - validation loss `1.511151`
  - checkpoint and summary completed.
- stderr only showed deprecation warnings, one NCCL barrier device warning, and
  a non-fatal DDP grad-stride performance warning.

Analysis:
- Functional DDP wiring is validated for two ranks.
- The barrier warning is easy to remove and should be fixed before the 8-rank
  smoke. The grad-stride warning is performance-only and does not require
  blocking the 8-GPU smoke.

Next:
- Commit/push the barrier fix, deploy a new isolated Della worktree commit, and
  submit replacement 8-GPU smoke plus dependent 8-GPU full run.

## 2026-06-11 16:03 PDT - Replacement 8-GPU DDP queue

Goal:
- Relaunch the 8-GPU smoke/full chain from the barrier-fix commit.

Hypothesis:
- The barrier-fix commit should preserve the successful 2-GPU behavior while
  eliminating the NCCL ambiguous-device warning before the 8-rank smoke.

Change:
- Created new isolated Della worktree:
  `/scratch/gpfs/AM43/lz3952/worktrees/Wan2.2/codex-droid-ddp-8gpu-barrierfix`
  at commit `7cb94a9671926d12f20b253d64ad37152795f577`.
- Linked untracked runtime assets from the canonical checkout.

Version Control:
- branch: `codex/droid-ddp-8gpu`
- launch_commit: `7cb94a9671926d12f20b253d64ad37152795f577`
- remote_commit/status: detached at `7cb94a9671926d12f20b253d64ad37152795f577`;
  untracked symlinks for `.venv`, `Wan2.2-TI2V-5B`, and `data`.

Command / Job:
- remote checks:
  - `bash -n run_action_conditioned_droid_dist.sh`
  - `.venv/bin/python -m py_compile scripts/train_action_conditioned_wan_droid.py`
- replacement queue:
  - 8-GPU smoke job `9565756`, output
    `runs/action_droid_dist_ddp8_smoke_7cb94a9`
  - 8-GPU full job `9565757`, dependency `afterok:9565756`, output
    `runs/action_droid_dist_side_bn512h8_L0-29_fresh_25step_fullcache_10k_lr5e-5_bs5x8_ddp_7cb94a9`

Result:
- status: queued
- `squeue` shows `9565756` pending with reason `(None)` and `9565757` pending
  with reason `(Dependency)`.
- `squeue --start` did not yet provide a concrete start time for `9565756`.

Analysis:
- The full 8-GPU run remains guarded by the smoke dependency.
- The active single-GPU run `9541718` continues independently from the
  canonical Della checkout.

Next:
- Monitor `9565756`. If it passes, allow `9565757` to proceed and inspect
  launch logs early. If it fails, cancel `9565757`, patch, and resubmit.
- Continue monitoring `9541718`; trigger step-3000 eval when checkpoint is
  written.

## 2026-06-11 16:36 PDT - Full-cache batch-5 step-3000 eval

Goal:
- Validate the third full-cache batch-size-5 checkpoint and continue the
  periodic eval cadence.

Hypothesis:
- If the larger full-cache run continues learning, training validation and
  fixed held-out random-noise eval MSE should improve versus step `2000`.

Change:
- No source change. Copied `ckpt_latest.pt` to `ckpt_step3000.pt` before eval.

Version Control:
- branch: `codex/droid-ddp-8gpu`
- implementation_commit: `n/a` for eval; training source still canonical
  single-GPU run code from `f370228`.
- changed_files:
  - `WORKLOG.md`
  - `HANDOFF.md`

Command / Job:
- eval job: `9566483`
- checkpoint:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/ckpt_step3000.pt`
- remote videos:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step3000_ep399_v0_s00004_s1000_1001/`
- local videos:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step3000_ep399_v0_s00004_s1000_1001/`
- viz-open URL:
  `http://localhost:8765/view?path=Wan2.2/_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step3000_ep399_v0_s00004_s1000_1001`

Result:
- status: passed
- `sacct -j 9566483` reports `COMPLETED 0:0`, elapsed `00:02:23`.
- Training validation improved:
  - step 1000: `0.2770900116302073`
  - step 2000: `0.24543552426621318`
  - step 3000: `0.23995679058134556`
- Eval videos validate at 320x192, 33 frames, 16 FPS.
- step-3000 eval metrics:
  - seed1000 latent MSE `0.2574395537376404` vs null
    `1.9798624515533447`.
  - seed1001 latent MSE `0.26347997784614563` vs null
    `4.661829471588135`.
- contact sheet:
  `_cluster/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step3000_ep399_v0_s00004_s1000_1001/droid_full_bs5_step3000_eval_contact_sheet.jpg`

Analysis:
- Numeric eval improved versus step 2000 on both seeds
  (`0.2719 -> 0.2574`, `0.2751 -> 0.2635`).
- Qualitatively, the output remains clearly better than null and preserves the
  scene layout, but the robot/gripper region still turns into a gray/hazy smear
  after motion starts. Larger-batch full-cache training is helping the metric
  but has not solved the motion-region blur.
- `viz_open` is not installed on the local PATH; the local viewer URL above is
  reachable and was used for visualization.

Next:
- Continue monitoring `9541718`; next eval target is step `4000`.
- Continue monitoring DDP 8-GPU smoke `9565756` and dependent full run
  `9565757`.

## 2026-06-11 17:25 PDT - 8-GPU DDP smoke passed

Goal:
- Validate the barrier-fix DDP trainer on all 8 GPUs before the full 8-GPU
  training job starts.

Hypothesis:
- The barrier-fix commit should remove the NCCL ambiguous-device barrier
  warning seen in the 2-GPU smoke while preserving functional DDP training.

Change:
- No source change. Fetched 8-GPU smoke logs and small run metadata locally.

Version Control:
- launch_commit: `7cb94a9671926d12f20b253d64ad37152795f577`
- local record branch: `codex/droid-ddp-8gpu`
- changed_files:
  - `WORKLOG.md`
  - `HANDOFF.md`

Command / Job:
- 8-GPU smoke job: `9565756`
- run dir:
  `/scratch/gpfs/AM43/lz3952/worktrees/Wan2.2/codex-droid-ddp-8gpu-barrierfix/runs/action_droid_dist_ddp8_smoke_7cb94a9`
- local fetched artifacts:
  - `_cluster/action_droid_dist_ddp8_smoke_7cb94a9/`
  - `_cluster/slurm_outputs/action-droid-dist/out_act-ddp8-smoke_9565756.log`
  - `_cluster/slurm_outputs/action-droid-dist/err_act-ddp8-smoke_9565756.log`
- dependent full 8-GPU job: `9565757`

Result:
- status: passed
- Smoke output reports `distributed=True`, `world_size=8`, local batch `1`,
  global batch `8`.
- Completed one optimizer step:
  loss `0.267435759305954`, grad norm `92.42627716064453`, lr `1e-7`.
- Rank-0 validation/checkpointing completed:
  val loss `1.5293481349945068`.
- `ckpt_latest.pt`, `ckpt_best_val.pt`, `config.json`, `summary.json`,
  `train_log.csv`, and `val_log.csv` were written.
- `9565757` dependency was released and is now pending with reason `(None)`;
  estimated start was `2026-06-12T06:31:24` at the latest check.
- `sacct` still showed a stale parent `PENDING` row for `9565756`, but the
  logs, output files, and released dependency all indicate successful
  completion.

Analysis:
- The NCCL barrier warning is gone after passing rank-local barrier
  `device_ids`.
- The remaining DDP grad-stride warning is performance-only and did not prevent
  a clean 8-rank optimizer step; it can be optimized later if step throughput
  is poor.
- Full 8-GPU training is now safely dependency-released.

Next:
- Monitor full 8-GPU job `9565757`; inspect launch logs immediately when it
  starts.
- Continue monitoring single-GPU job `9541718`; next eval target is step `4000`.

## 2026-06-11 20:26 PDT - Full-cache batch-5 step-4000 eval queued

Goal:
- Validate the fourth full-cache batch-size-5 checkpoint and keep the periodic
  eval cadence active while the full 8-GPU DDP run waits in queue.

Hypothesis:
- If the larger full-cache run continues learning, step `4000` validation and
  held-out fixed random-noise eval should improve versus step `3000`.

Change:
- No source change. Copied `ckpt_latest.pt` to `ckpt_step4000.pt` after the
  step-4000 checkpoint timestamp advanced.

Version Control:
- branch: `codex/droid-ddp-8gpu`
- implementation_commit: `n/a` for eval; single-GPU training source remains
  the canonical Della checkout used by job `9541718`.
- changed_files:
  - `WORKLOG.md`
  - `HANDOFF.md`

Command / Job:
- active training job: `9541718`
- eval job: `9573559`
- checkpoint:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/ckpt_step4000.pt`
- remote videos:
  `runs/action_droid_side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5_bs5/videos_step4000_ep399_v0_s00004_s1000_1001/`
- 8-GPU full DDP job: `9565757`

Result:
- status: eval queued
- Training validation improved:
  - step 1000: `0.2770900116302073`
  - step 2000: `0.24543552426621318`
  - step 3000: `0.23995679058134556`
  - step 4000: `0.21933504613116384`
- `ckpt_step4000.pt` timestamp:
  `2026-06-11 23:25:52 -0400`, size `958230026` bytes.
- Eval job `9573559` is pending on priority with estimated start
  `2026-06-12T00:01:31` Della time.
- Current DROID training error scan is clean for OOM, traceback, NaN,
  RuntimeError, and CUDA errors.
- Full 8-GPU job `9565757` remains pending on priority; latest estimate
  `2026-06-12T09:58:21` Della time with a 2-day time limit.

Analysis:
- The metric trend is healthy through step `4000`; the validation drop from
  `0.2399568` to `0.2193350` is the strongest interval improvement in this
  batch-5 run so far.
- Step `3980` showed a large grad-norm spike (`857.10`), but step `4000`
  returned to `4.03`, no error signatures appeared, and validation improved,
  so this currently looks like an outlier batch rather than instability.

Next:
- Monitor eval job `9573559`; fetch videos locally, validate video metadata,
  inspect a contact sheet, and record metrics/artifacts when it completes.
- Keep monitoring single-GPU training job `9541718`; next eval target is step
  `5000`.
- Keep monitoring 8-GPU DDP job `9565757`; inspect launch logs immediately
  when it starts.
