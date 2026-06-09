# Della Development Loop

Default paths:

- Local working copy: `~/code/Wan2.2`
- Della working copy: `/scratch/gpfs/AM43/lz3952/Wan2.2`
- SSH alias: `della-gpu`

Tracked code, scripts, configs, tests, and docs must move through Git:
commit locally, push to the fork, then pull/update the checkout on Della.
Use `rsync` only for large untracked runtime state such as datasets,
checkpoints, generated videos, logs, and scratch caches.

One-time local setup:

```bash
cd ~/code
git clone git@github.com:lihzha/Wan2.2.git Wan2.2
cd Wan2.2
git remote add upstream https://github.com/Wan-Video/Wan2.2.git
```

Normal edit/deploy/run loop:

```bash
cd ~/code/Wan2.2
git status --short
git add <explicit-files>
git commit -m "<message>"
scripts/della_loop.sh deploy-code main
scripts/della_loop.sh submit run_zinit_probe_droid.sh
scripts/della_loop.sh status
scripts/della_loop.sh fetch-logs slurm_outputs/zinit-probe
scripts/della_loop.sh fetch-results runs/zinit_probe_droid _cluster/zinit_probe_droid
```

`deploy-code` pushes local `HEAD` to the fork and makes Della fetch/reset its
tracked files to that exact commit. If `/scratch/gpfs/AM43/lz3952/Wan2.2`
does not yet have `.git`, the helper initializes Git in place, keeps untracked
runtime directories such as `data/`, `runs/`, `slurm_outputs/`, and
`Wan2.2-TI2V-5B/`, and overwrites tracked source files from the fork.

Before each launch, record both local and remote commits in `WORKLOG.md`.
The `submit` helper refuses to submit if the remote checkout has dirty tracked
changes.
