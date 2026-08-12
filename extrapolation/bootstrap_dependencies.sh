#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
agents_dir="$script_dir/agents"
ayla_proxy="${GIT_HTTP_PROXY:-fwdproxy:8080}"

clone_pinned() {
  local repository="$1"
  local destination="$2"
  local revision="$3"

  if [[ ! -d "$destination/.git" ]]; then
    git -c http.proxy="$ayla_proxy" clone --filter=blob:none "$repository" "$destination"
  fi
  git -C "$destination" -c http.proxy="$ayla_proxy" fetch origin "$revision"
  git -C "$destination" checkout --detach "$revision"
}

clone_pinned \
  https://github.com/triton-lang/triton.git \
  "$agents_dir/triton-upstream" \
  faa8c288ddca78215fa130d6850a987711e06f1c

clone_pinned \
  https://github.com/NVIDIA/cutlass.git \
  "$agents_dir/cutlass-upstream" \
  dcf215af68a2d08d305076c152a06f201728cd53
