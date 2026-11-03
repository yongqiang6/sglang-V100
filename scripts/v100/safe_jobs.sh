#!/usr/bin/env bash
# Usage: safe_jobs.sh [GiB per compiler process, default 6]
set -euo pipefail

job_gib="${1:-6}"

if [[ -n "${MAX_JOBS:-}" ]]; then
  printf '%s\n' "$MAX_JOBS"
  exit 0
fi

cpu_jobs="$(nproc)"
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"

# Respect a cgroup memory limit when Docker has one. memory.max contains "max"
# for an unlimited container.
if [[ -r /sys/fs/cgroup/memory.max ]]; then
  cgroup_max="$(</sys/fs/cgroup/memory.max)"
  cgroup_used="$(</sys/fs/cgroup/memory.current)"
  if [[ "$cgroup_max" =~ ^[0-9]+$ ]] && (( cgroup_max > cgroup_used )); then
    cgroup_available_kib=$(( (cgroup_max - cgroup_used) / 1024 ))
    (( cgroup_available_kib < available_kib )) && \
      available_kib="$cgroup_available_kib"
  fi
fi

# Keep 4 GiB for the OS/linker. The default 6 GiB per compiler process is
# measured: cicc peaks at ~5.5 GiB on csrc/gemm/fp8_gemm_kernel.cu (sm_70).
memory_jobs=$(( (available_kib - 4 * 1024 * 1024) / (job_gib * 1024 * 1024) ))
(( memory_jobs < 1 )) && memory_jobs=1
(( memory_jobs < cpu_jobs )) && cpu_jobs="$memory_jobs"
printf '%s\n' "$cpu_jobs"
