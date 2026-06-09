#!/usr/bin/env bash
set -euo pipefail

# Local helper for the edit -> commit -> push -> cluster pull -> sbatch -> fetch loop.
# Defaults match this project; override with env vars if needed.

REMOTE=${REMOTE:-della-gpu}
REMOTE_DIR=${REMOTE_DIR:-/scratch/gpfs/AM43/lz3952/Wan2.2}
LOCAL_DIR=${LOCAL_DIR:-$HOME/code/Wan2.2}
FETCH_DIR=${FETCH_DIR:-$LOCAL_DIR/_cluster}
GIT_PULL_URL=${GIT_PULL_URL:-}

current_branch() {
  git -C "$LOCAL_DIR" branch --show-current
}

public_pull_url() {
  if [ -n "$GIT_PULL_URL" ]; then
    printf "%s\n" "$GIT_PULL_URL"
    return
  fi

  local origin_url
  origin_url=$(git -C "$LOCAL_DIR" remote get-url origin)
  case "$origin_url" in
    git@github.com:*.git)
      printf "https://github.com/%s\n" "${origin_url#git@github.com:}"
      ;;
    git@github.com:*)
      printf "https://github.com/%s.git\n" "${origin_url#git@github.com:}"
      ;;
    *)
      printf "%s\n" "$origin_url"
      ;;
  esac
}

require_clean_tracked_tree() {
  if ! git -C "$LOCAL_DIR" diff --quiet || ! git -C "$LOCAL_DIR" diff --cached --quiet; then
    echo "tracked local changes are not committed; commit before deploying" >&2
    git -C "$LOCAL_DIR" status --short
    exit 2
  fi
}

quote_remote_cmd() {
  local quoted=()
  for arg in "$@"; do
    quoted+=("$(printf "%q" "$arg")")
  done
  printf "%s " "${quoted[@]}"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/della_loop.sh pull-code [branch]
  scripts/della_loop.sh push-code [branch]
  scripts/della_loop.sh deploy-code [branch]
  scripts/della_loop.sh remote-git-status
  scripts/della_loop.sh submit <sbatch args...>
  scripts/della_loop.sh status
  scripts/della_loop.sh fetch-logs [remote_subdir]
  scripts/della_loop.sh fetch-results <remote_subdir> [local_subdir]

Examples:
  scripts/della_loop.sh deploy-code main
  scripts/della_loop.sh submit run_zinit_probe_droid.sh
  scripts/della_loop.sh fetch-logs slurm_outputs/zinit-probe
  scripts/della_loop.sh fetch-results runs/zinit_probe_droid _cluster/zinit_probe_droid

Tracked code/config/script changes must go through Git. Use rsync only for
logs, datasets, checkpoints, generated videos, and other untracked artifacts.
EOF
}

cmd=${1:-}
if [ -z "$cmd" ]; then
  usage
  exit 2
fi
shift || true

case "$cmd" in
  pull-code)
    branch=${1:-$(current_branch)}
    git -C "$LOCAL_DIR" fetch origin "$branch"
    git -C "$LOCAL_DIR" pull --ff-only origin "$branch"
    ;;

  push-code)
    branch=${1:-$(current_branch)}
    require_clean_tracked_tree
    git -C "$LOCAL_DIR" push origin "HEAD:${branch}"
    ;;

  deploy-code)
    branch=${1:-$(current_branch)}
    require_clean_tracked_tree
    pull_url=$(public_pull_url)
    local_head=$(git -C "$LOCAL_DIR" rev-parse HEAD)
    git -C "$LOCAL_DIR" push origin "HEAD:${branch}"
    ssh "$REMOTE" \
      "REMOTE_DIR=$(printf "%q" "$REMOTE_DIR") GIT_PULL_URL=$(printf "%q" "$pull_url") BRANCH=$(printf "%q" "$branch") EXPECTED_HEAD=$(printf "%q" "$local_head") bash -s" <<'EOF'
set -euo pipefail
mkdir -p "$REMOTE_DIR"
cd "$REMOTE_DIR"
if [ ! -d .git ]; then
  git init
fi
git remote remove origin >/dev/null 2>&1 || true
git remote add origin "$GIT_PULL_URL"
git fetch origin "$BRANCH"
git checkout -f -B "$BRANCH" FETCH_HEAD
git reset --hard FETCH_HEAD
if git lfs version >/dev/null 2>&1; then
  git lfs pull
fi
remote_head=$(git rev-parse HEAD)
echo "[remote-git] branch=$BRANCH head=$remote_head expected=$EXPECTED_HEAD"
if [ "$remote_head" != "$EXPECTED_HEAD" ]; then
  echo "[remote-git] head mismatch" >&2
  exit 1
fi
git status --short --untracked-files=no
EOF
    ;;

  remote-git-status)
    ssh "$REMOTE" "cd $(printf "%q" "$REMOTE_DIR") && git rev-parse --is-inside-work-tree && git branch --show-current && git rev-parse HEAD && git status --short --untracked-files=no"
    ;;

  submit)
    if [ "$#" -eq 0 ]; then
      echo "submit requires sbatch arguments" >&2
      exit 2
    fi
    remote_args=$(quote_remote_cmd "$@")
    ssh "$REMOTE" "cd $(printf "%q" "$REMOTE_DIR") && \
      git rev-parse HEAD && \
      git diff --quiet && git diff --cached --quiet && \
      sbatch ${remote_args}"
    ;;

  status)
    ssh "$REMOTE" "squeue -u \$(whoami)"
    ;;

  fetch-logs)
    remote_subdir=${1:-slurm_outputs}
    mkdir -p "$FETCH_DIR"
    rsync -av "$REMOTE:$REMOTE_DIR/$remote_subdir/" "$FETCH_DIR/$remote_subdir/"
    ;;

  fetch-results)
    if [ "$#" -lt 1 ]; then
      echo "fetch-results requires a remote subdir, e.g. runs/zinit_probe_droid" >&2
      exit 2
    fi
    remote_subdir=$1
    local_subdir=${2:-$FETCH_DIR/$remote_subdir}
    mkdir -p "$local_subdir"
    rsync -av --prune-empty-dirs \
      --include='*/' \
      --include='*.json' \
      --include='*.jsonl' \
      --include='*.csv' \
      --include='*.txt' \
      --include='*.log' \
      --include='*.out' \
      --include='*.err' \
      --exclude='*' \
      "$REMOTE:$REMOTE_DIR/$remote_subdir/" "$local_subdir/"
    ;;

  *)
    usage
    exit 2
    ;;
esac
