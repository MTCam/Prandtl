#!/usr/bin/env bash
set -euo pipefail

# --- Harness state (do NOT set -e inside per-case runs) ---
declare -a SUCCEEDED=()
declare -a FAILED=()

# Zero-pad cycle index to 6 digits: 0 -> 000000, 100 -> 000100
fmt_cycle() {
  printf "%06d" "${1:-0}"
}

# Verify expected ParaView outputs exist for N steps under out/ParaView
# Requires ParaView.pvd and both Cycle000000 and Cycle00NNNN
check_outputs() {
  local outdir="$1" ; local nsteps="$2"
  local pv="${outdir}/ParaView/ParaView.pvd"
  local c0="${outdir}/ParaView/Cycle$(fmt_cycle 0)"
  local cN="${outdir}/ParaView/Cycle$(fmt_cycle "${nsteps}")"
  [[ -f "${pv}" && -d "${c0}" && -d "${cN}" ]]
}


# Default knobs
NSTEPS=100
TOP=$(pwd)
BUILDDIR="${TOP}/build"
EXE="${BUILDDIR}/Prandtl"
RUNDIR="${TOP}/RunTests"
LISTFILE=""
ONECFG=""

usage() {
  cat <<EOF
Usage: $0 [-n STEPS] [-b BUILDDIR] [-e EXECUTABLE] [-o RUNDIR] (-c CONFIG.json | -l LIST.txt)

  -n STEPS      Number of steps to run (default: ${NSTEPS})
  -b BUILDDIR   Build directory (default: ${BUILDDIR})
  -e EXECUTABLE Path to Prandtl executable (default: ${EXE})
  -o RUNDIR     Directory to run in (default: ${RUNDIR})
  -c CONFIG     Single example config.json to run
  -l LIST       List file with one config.json path per line (comments (#) allowed)

Examples:
  $0 -c TestCases/NavierStokes/2D/LidDrivenCavity/config.json
  $0 -l examples.txt
EOF
}

# ---- Parse args
while getopts ":n:b:e:o:c:l:h" opt; do
  case $opt in
    n) NSTEPS="${OPTARG}";;
    b) BUILDDIR="${OPTARG}"; EXE="${BUILDDIR}/Prandtl";;
    e) EXE="${OPTARG}";;
    o) RUNDIR="${OPTARG}";;
    c) ONECFG="${OPTARG}";;
    l) LISTFILE="${OPTARG}";;
    h) usage; exit 0;;
    \?) echo "Unknown option -$OPTARG" >&2; usage; exit 2;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2;;
  esac
done

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq not found; please install jq." >&2
  exit 2
fi
if [[ ! -x "${EXE}" ]]; then
  echo "ERROR: Prandtl executable not found at ${EXE}" >&2
  exit 2
fi

# ---- Resolve which configs to run
declare -a CFGS
if [[ -n "${ONECFG}" && -n "${LISTFILE}" ]]; then
  echo "ERROR: choose either -c or -l, not both." >&2; exit 2
elif [[ -n "${ONECFG}" ]]; then
  CFGS+=("${ONECFG}")
elif [[ -n "${LISTFILE}" ]]; then
  while IFS= read -r line; do
    # skip empties and comments
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    CFGS+=("${line}")
  done < "${LISTFILE}"
else
  echo "ERROR: must provide -c CONFIG.json or -l LIST.txt" >&2; exit 2
fi

# ---- Ensure run sandbox
mkdir -p "${RUNDIR}"
# Copy the executable into the run dir (your preferred workflow)
cp -f "${EXE}" "${RUNDIR}/Prandtl"

# ---- Function to run one example
run_one() {
  local cfg_rel="$1"
  local cfg_abs
  cfg_abs="$(cd "$(dirname "${cfg_rel}")" && pwd)/$(basename "${cfg_rel}")"

  if [[ ! -f "${cfg_abs}" ]]; then
    echo "ERROR: config not found: ${cfg_rel}" >&2
    return 1
  fi

  echo "==> Running example: ${cfg_rel}"
  echo "    Working dir: ${RUNDIR}"

  # Prepare per-example working area
  local exname
  exname="$(basename "$(dirname "${cfg_abs}")")"   # e.g., LidDrivenCavity
  local work="${RUNDIR}/${exname}"
  rm -rf "${work}"
  mkdir -p "${work}"
  local outdir="${work}/out"
  mkdir -p "${outdir}"

  # Create a patched config inside work/
  local patched="${work}/config.patched.json"
  local nsteps="${NSTEPS}"
  if [[ "${nsteps}" == "0" ]]; then
      nsteps=100
  fi
  jq --argjson N "${nsteps}" --arg out "${outdir}" '
    def isnum: type=="number";
    . as $root
    | ($root.runTime // {}) as $rt
    | .runTime = (
        $rt
        | .visualize = true
        | .paraview  = true
        | .visit     = false
        | .nancheck  = true
        # favor 10-step cadence when divisible to ensure Cycle000000 & Cycle00NNNN appear
        | .vis_steps = ( if ($N % 10 == 0) then 10
                         else (((($N/2)|floor) | if .==0 then 1 else . end))
                         end )
        | .variable_dt = false
        | ( if (.dt? | isnum) then .
            elif (.final_time? | isnum) then (.dt = (.final_time / $N))
            else (.dt = 0.0000001)
            end )
        | .final_time = (.dt * $N)
        | .initial_save_dt = (.dt * .vis_steps)
        | .output_file_path = $out
        | .checkpoint_load = false
      )
  ' "${cfg_abs}" > "${patched}"

  # Run from the per-example dir; keep your “two levels down” invariant
  # Run example (isolate failures; do NOT exit on first error)
  set +e
  ( cd "${work}" && mpiexec -n 2 ../Prandtl -c "${patched}" )
  local run_rc=$?
  set -e

  # Basic regression: require ParaView.pvd + Cycle000000 + Cycle00NNNN
  if [[ ${run_rc} -eq 0 ]] && check_outputs "${outdir}" "${NSTEPS}"; then
    echo "✓ Example OK: ${exname} (outputs in ${outdir})"
    SUCCEEDED+=("${cfg_rel}")
    return 0
  else
    echo "✗ Example FAILED: ${exname}"
    [[ ${run_rc} -ne 0 ]] && echo "  - runtime exit code: ${run_rc}"
    if [[ ! -f "${outdir}/ParaView/ParaView.pvd" ]]; then
      echo "  - missing: ${outdir}/ParaView/ParaView.pvd"
    fi
    if [[ ! -d "${outdir}/ParaView/Cycle$(fmt_cycle 0)" ]]; then
      echo "  - missing: ${outdir}/ParaView/Cycle$(fmt_cycle 0)"
    fi
    if [[ ! -d "${outdir}/ParaView/Cycle$(fmt_cycle "${NSTEPS}")" ]]; then
      echo "  - missing: ${outdir}/ParaView/Cycle$(fmt_cycle "${NSTEPS}")"
    fi
    FAILED+=("${cfg_rel}")
    return 1
  fi

  echo "✓ Example OK: ${exname} (outputs in ${outdir})"
}

# ---- Iterate
rc=0
for cfg in "${CFGS[@]}"; do
  # if ! run_one "${cfg}"; then
  #  rc=1
  # fi
  run_one "${cfg}" || rc=1
done

# ---- Summary
echo
echo "===== Example Summary ====="
echo "Total: ${#CFGS[@]} | Succeeded: ${#SUCCEEDED[@]} | Failed: ${#FAILED[@]}"
if (( ${#SUCCEEDED[@]} > 0 )); then
  printf '  ✓ %s\n' "${SUCCEEDED[@]}"
fi
if (( ${#FAILED[@]} > 0 )); then
  printf '  ✗ %s\n' "${FAILED[@]}"
fi

exit ${rc}
