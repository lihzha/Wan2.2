# Wan2.2 Action Conditioning Worklog

This worklog follows `robotics-cluster-development-core`. It was created
retrospectively on 2026-06-09 from local code, fetched cluster artifacts under
`_cluster/`, and the active Della monitoring history. Della was down for
maintenance during creation, so final queue state after the last observed
snapshot is not live-verified here.

## Current State As Of 2026-06-09

Goal:
- Learn an action/I0-conditioned adapter for frozen Wan2.2 that can control
  generated videos without fine-tuning the backbone, with training/eval
  guidance scale aligned and with fresh initial diffusion noise.

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
- Side adapter can overfit one sample at 1 diffusion step very strongly. A
  25-step fresh-noise pilot also gives stable random-noise eval near the pilot
  train loss, but needs a longer 10k-step run for a real overfit test.
- Full DROID training is blocked on cache completion and Della maintenance, not
  on trainer correctness: a one-step H200 DROID trainer smoke completed.

Latest known remote state before Della maintenance blocked SSH:
- Overfit 10k job `9478714`, run
  `runs/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_10k_lr5e-5`,
  was pending on `ailab` due maintenance / unavailable nodes.
- DROID val window cache complete: `14,636` windows.
- DROID train window cache reached `140,679` windows.
- DROID train cache remote submitter PID was `2821907`.
- Active/pending 1024-shard GPU-test cache array was `9479448_[100-120]`.
- Post-cache training submitter PID was previously `2661285`, waiting for
  train `1,440,554` and val `14,636` cache counts, then set to submit DROID
  smoke and full 10k training.

Key artifact paths:
- Local fetched artifacts: `_cluster/`
- Random-eval videos for 25-step fresh-noise pilot:
  `_cluster/action_overfit_ep0_v0_side_bn512h8_L0-29_fresh_25step_pilot/videos_best_random_eval_s1000_1003/`
- Loss curve for 25-step pilot:
  `_cluster/loss_curves/side_fresh25_pilot_loss.svg.png`
- DROID planning scripts and trainers are in `scripts/` and Slurm wrappers in
  the repo root.

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

When Della returns:

1. Check SSH and maintenance state:
   `ssh della-gpu 'hostname && date && squeue -u $(whoami) -o "%.18i %.9P %.32j %.8T %.20M %.60R"'`
2. Check cache counts:
   `find data/droid_cache_windows_v0/train -mindepth 1 -maxdepth 1 -type d | wc -l`
   and same for val.
3. Check remote submitters:
   `pgrep -af submit_droid_train_cache_chunks`,
   `pgrep -af wait_and_submit_action_droid_training`.
4. Tail:
   `runs/droid_window_plans/train_cache_chunk_submitter_gputest_1024.log`,
   `runs/droid_window_plans/action_droid_training_submitter.log`.
5. Inspect `9479448` and `9478714` with `squeue`/`scontrol`/`sacct`.
6. If cache completes, verify DROID smoke/full training submission and plot
   `train_log.csv`/`val_log.csv`.
7. If 10k overfit completes, fetch logs, plot loss, export random eval videos,
   and record remote/local video paths.
