#!/usr/bin/env bash
# Run all vec-mx-* benchmarks in parallel on VCS (GENV256D128ShuttleConfig).
# Logs go to vec-mx-logs/<benchmark>.log. Uses a job pool to limit parallelism.
#
# Usage:
#   ./run-vec-mx-parallel.sh [ -j N ]
#   -j N   max parallel jobs (default: 8)
#
# Run from chipyard root or from generators/saturn/benchmarks/.
# Prereq: simv already built: cd sims/vcs && make CONFIG=GENV256D128ShuttleConfig

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$SCRIPT_DIR"
# Chipyard root: two levels up from generators/saturn/benchmarks
CHIPYARD_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VCS_DIR="$CHIPYARD_ROOT/sims/vcs"
CONFIG="${CONFIG:-GENV256D128ShuttleConfig}"
LOG_DIR="$BENCHMARKS_DIR/vec-mx-logs"
mkdir -p "$LOG_DIR"

# Default parallelism
NJOBS=8
while getopts "j:" opt; do
  case $opt in
    j) NJOBS="$OPTARG" ;;
    *) echo "Usage: $0 [-j N]"; exit 1 ;;
  esac
done

# All vec-mx-* benchmarks we can run (must have .riscv)
BMARKS=(
  vec-mx-unary
  vec-mx-binary
  vec-mx-matmul
  vec-mx-fmacc
  vec-mx-axpy
  vec-mx-dotprod
  vec-mx-scale
  vec-mx-bias-add
  vec-mx-reduction-sum
  vec-mx-matmul-64
  vec-mx-matmul-128
  vec-mx-binary-large
  vec-mx-unary-large
  vec-mx-max-reduction
  vec-mx-l2norm
  vec-mx-gemv
  vec-mx-masked-binary
  vec-mx-linear
  vec-mx-rmsnorm
  vec-mx-relu
  vec-mx-conv1d
)

BINARIES=()
for b in "${BMARKS[@]}"; do
  if [ -f "$BENCHMARKS_DIR/$b.riscv" ]; then
    BINARIES+=( "$BENCHMARKS_DIR/$b.riscv" )
  else
    echo "Skip (no binary): $b"
  fi
done

if [ ${#BINARIES[@]} -eq 0 ]; then
  echo "No vec-mx-*.riscv found in $BENCHMARKS_DIR. Run 'make riscv' there first."
  exit 1
fi

echo "=== Running ${#BINARIES[@]} vec-mx benchmarks in parallel (-j $NJOBS) ==="
echo "CONFIG=$CONFIG  VCS_DIR=$VCS_DIR  LOG_DIR=$LOG_DIR"
echo ""

# Job pool: run up to NJOBS at a time
run_one() {
  local bin="$1"
  local name="$(basename "$bin" .riscv)"
  local log="$LOG_DIR/${name}.log"
  local ret=0
  (cd "$VCS_DIR" && make CONFIG="$CONFIG" BINARY="$bin" run-binary-fast > "$log" 2>&1) || ret=$?
  if [ $ret -eq 0 ]; then
    echo "PASS $name"
  else
    echo "FAIL $name (see $log)"
  fi
  return $ret
}
export -f run_one
export VCS_DIR CONFIG LOG_DIR

# Job pool: start up to NJOBS at a time, wait for one to finish before starting next
running=0
for bin in "${BINARIES[@]}"; do
  while [ "$running" -ge "$NJOBS" ]; do
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  done
  run_one "$bin" &
  running=$((running + 1))
done
wait

# Summary (use success markers; avoid matching "pipefail" in the logged command)
echo ""
echo "=== Logs in $LOG_DIR ==="
failed=0
for bin in "${BINARIES[@]}"; do
  name="$(basename "$bin" .riscv)"
  if [ ! -f "$LOG_DIR/${name}.log" ]; then
    echo "MISS $name"
    failed=$((failed + 1))
  elif grep -qE "All tests passed|\\\$finish" "$LOG_DIR/${name}.log" 2>/dev/null; then
    echo "PASS $name"
  elif grep -qiE "FAIL|failure|error:|timeout|abort" "$LOG_DIR/${name}.log" 2>/dev/null; then
    echo "FAIL $name"
    failed=$((failed + 1))
  else
    echo "PASS $name"
  fi
done
[ "$failed" -gt 0 ] && exit 1
exit 0
