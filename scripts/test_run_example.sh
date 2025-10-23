#!/usr/bin/env bash
set -euo pipefail

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

  jq --argjson N "${NSTEPS}" --arg out "${outdir}" '
    def isnum: type=="number";
    . as $root
    | ($root.runTime // {}) as $rt
    | .runTime = (
        $rt
        | .visualize = true
        | .paraview  = true
        | .visit     = false
        | .nancheck  = true
        | .vis_steps = (((($N/2)|floor) | if .==0 then 1 else . end))
        | .variable_dt = false
        | ( if (.dt? | isnum) then .
            elif (.final_time? | isnum) then (.dt = (.final_time / $N))
            else (.dt = 0.001)
          end )
        | .final_time = (.dt * $N)
        | .output_file_path = $out
      )
  ' "${cfg_abs}" > "${patched}"

  # Run from the per-example dir; keep your “two levels down” invariant
  ( cd "${work}" && ../Prandtl -c "${patched}" )

  # Basic regression hook: ensure at least one file in out/Paraview
  if ! find "${outdir}/Paraview" -type f -maxdepth 1 -print -quit | grep -q .; then
    echo "ERROR: No output files produced in ${outdir}" >&2
    return 2
  fi

  echo "✓ Example OK: ${exname} (outputs in ${outdir})"
}

# ---- Iterate
rc=0
for cfg in "${CFGS[@]}"; do
  if ! run_one "${cfg}"; then
    rc=1
  fi
done

exit ${rc}
