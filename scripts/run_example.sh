#!/usr/bin/env bash
set -euo pipefail

# --- Harness state (do NOT set -e inside per-case runs) ---
declare -a SUCCEEDED=()
declare -a FAILED=()

fmt_cycle() {
  printf "%06d" "${1:-0}"
}

check_outputs() {
  local outdir="$1"
  local nsteps="$2"
  local pv="${outdir}/ParaView/ParaView.pvd"
  local c0="${outdir}/ParaView/Cycle$(fmt_cycle 0)"
  local cN="${outdir}/ParaView/Cycle$(fmt_cycle "${nsteps}")"
  [[ -f "${pv}" && -d "${c0}" && -d "${cN}" ]]
}

# Default knobs
TOP=$(pwd)
BUILDDIR="${TOP}/build"
EXE="${BUILDDIR}/Prandtl"
RUNDIR="${TOP}/RunTests"

LISTFILE=""
ONECFG=""

NSTEPS=100
DT=0.0001
CFL=""
NSTEPS_OVERRIDE=0
DT_OVERRIDE=0
CFL_OVERRIDE=0

NMPIRANKS=2
DEVICE="cpu"
NHOSTS="1"
PLATFORM="auto"

HOST_SHORT="$(hostname -s)"

usage() {
  cat <<EOF2
Usage: $0 [-n STEPS] [-t DT] [-d CFL] [-b BUILDDIR] [-e EXECUTABLE] [-o RUNDIR] \
          [-p NP] [-H NHOSTS] [-r DEVICE] [-P PLATFORM] (-c CONFIG.json | -l LIST.txt)

  -n STEPS      Number of steps to run (default: ${NSTEPS})
  -t DT         Fixed timestep to use
  -d CFL        Fixed CFL to use
  -b BUILDDIR   Build directory (default: ${BUILDDIR})
  -e EXECUTABLE Path to Prandtl executable (default: ${EXE})
  -o RUNDIR     Directory to run in (default: ${RUNDIR})
  -p NP         Number of MPI ranks to run (default: ${NMPIRANKS})
  -H NHOSTS     Number of compute hosts/nodes to use (default: ${NHOSTS})
  -r DEVICE     Compute device to run on (e.g. cpu or hip, default: ${DEVICE})
  -P PLATFORM   Launcher platform: auto | local | tuolumne (default: ${PLATFORM})
  -c CONFIG     Single example config.json to run
  -l LIST       List file with one config.json path per line (comments (#) allowed)
  -h            Show this help message

Notes:
  * Use either -t or -d, not both.
  * PLATFORM=auto chooses from hostname.

Examples:
  $0 -c TestCases/NavierStokes/2D/LidDrivenCavity/config.json
  $0 -l examples.txt -p 4 -r hip -H 2 -P tuolumne
  $0 -c config.json -n 200 -t 1.0e-5
EOF2
}

# ---- Parse args
while getopts ":n:t:d:b:e:o:p:r:c:l:H:P:h" opt; do
  case $opt in
    n) NSTEPS="${OPTARG}"; NSTEPS_OVERRIDE=1 ;;
    t) DT="${OPTARG}"; DT_OVERRIDE=1 ;;
    d) CFL="${OPTARG}"; CFL_OVERRIDE=1 ;;
    b) BUILDDIR="${OPTARG}"; EXE="${BUILDDIR}/Prandtl" ;;
    e) EXE="${OPTARG}" ;;
    o) RUNDIR="${OPTARG}" ;;
    p) NMPIRANKS="${OPTARG}" ;;
    r) DEVICE="${OPTARG}" ;;
    c) ONECFG="${OPTARG}" ;;
    l) LISTFILE="${OPTARG}" ;;
    H) NHOSTS="${OPTARG}" ;;
    P) PLATFORM="${OPTARG}" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done

if [[ "${DT_OVERRIDE}" -eq 1 && "${CFL_OVERRIDE}" -eq 1 ]]; then
  echo "ERROR: choose either fixed timestep (-t) or fixed CFL (-d), not both." >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq not found; please install jq." >&2
  exit 2
fi
if [[ ! -x "${EXE}" ]]; then
  echo "ERROR: Prandtl executable not found at ${EXE}" >&2
  exit 2
fi

# ---- Resolve which configs to run
declare -a CFGS=()
if [[ -n "${ONECFG}" && -n "${LISTFILE}" ]]; then
  echo "ERROR: choose either -c or -l, not both." >&2
  exit 2
elif [[ -n "${ONECFG}" ]]; then
  CFGS+=("${ONECFG}")
elif [[ -n "${LISTFILE}" ]]; then
  while IFS= read -r line; do
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    CFGS+=("${line}")
  done < "${LISTFILE}"
else
  echo "ERROR: must provide -c CONFIG.json or -l LIST.txt" >&2
  exit 2
fi

# ---- Ensure run sandbox
mkdir -p "${RUNDIR}"
cp -f "${EXE}" "${RUNDIR}/Prandtl"

resolve_platform() {
  case "${PLATFORM}" in
    auto)
      case "${HOST_SHORT}" in
        tuo*) echo "tuolumne" ;;
        *)    echo "local" ;;
      esac
      ;;
    local|tuolumne)
      echo "${PLATFORM}"
      ;;
    *)
      echo "ERROR: unsupported platform '${PLATFORM}'. Use auto, local, or tuolumne." >&2
      return 1
      ;;
  esac
}

build_launcher() {
  local resolved_platform="$1"
  case "${resolved_platform}" in
    tuolumne)
      printf 'flux run --exclusive -N "%s" -n "%s"' "${NHOSTS}" "${NMPIRANKS}"
      ;;
    local)
      printf 'mpiexec -n "%s"' "${NMPIRANKS}"
      ;;
    *)
      echo "ERROR: no launcher available for platform '${resolved_platform}'" >&2
      return 1
      ;;
  esac
}

patch_config() {
  local cfg_abs="$1"
  local patched="$2"
  local nsteps="$3"

  if [[ "${DT_OVERRIDE}" -eq 1 ]]; then
    jq --argjson N "${nsteps}" \
       --argjson DT "${DT}" '
      . as $root
      | ($root.runTime // {}) as $rt
      | .runTime = (
          $rt
          | .visualize = true
          | .paraview  = true
          | .visit     = false
          | .nancheck  = true
          | .output_file_path = "./out"
          | .checkpoint_load = false
          | .variable_dt = false
          | .dt = $DT
          | .final_time = ($N * $DT)
        )
    ' "${cfg_abs}" > "${patched}"
  else
    jq --argjson N "${nsteps}" '
      . as $root
      | ($root.runTime // {}) as $rt
      | .runTime = (
          $rt
          | .visualize = true
          | .paraview  = true
          | .visit     = false
          | .nancheck  = true
          | .output_file_path = "./out"
          | .checkpoint_load = false
        )
    ' "${cfg_abs}" > "${patched}"
  fi
}


run_one() {
  local cfg_rel="$1"
  local cfg_abs
  cfg_abs="$(cd "$(dirname "${cfg_rel}")" && pwd)/$(basename "${cfg_rel}")"

  if [[ ! -f "${cfg_abs}" ]]; then
    echo "ERROR: config not found: ${cfg_rel}" >&2
    return 1
  fi

  local resolved_platform
  resolved_platform="$(resolve_platform)" || return 1

  local mpi_launcher
  mpi_launcher="$(build_launcher "${resolved_platform}")" || return 1

  echo "==> Running example: ${cfg_rel}"
  echo "    Working dir: ${RUNDIR}"
  echo "    Platform: ${resolved_platform}"
  echo "    MPI ranks: ${NMPIRANKS}"
  echo "    Hosts: ${NHOSTS}"
  echo "    Device: ${DEVICE}"

  local exname
  exname="$(basename "$(dirname "${cfg_abs}")")"
  local work="${RUNDIR}/${exname}"
  rm -rf "${work}"
  mkdir -p "${work}"
  local outdir="${work}/out"
  mkdir -p "${outdir}"

  local patched="${work}/config.patched.json"
  local nsteps="${NSTEPS}"
  if [[ "${nsteps}" == "0" ]]; then
    nsteps=100
  fi

  # patch_config "${cfg_abs}" "${patched}" "${outdir}" "${nsteps}"
  patch_config "${cfg_abs}" "${patched}" "${nsteps}"

  set +e
  ( cd "${work}" && eval ${mpi_launcher} ../Prandtl -d "${DEVICE}" -c "${patched}" )
  local run_rc=$?
  set -e

  if [[ ${run_rc} -eq 0 ]] && check_outputs "${outdir}" "${nsteps}"; then
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
    if [[ ! -d "${outdir}/ParaView/Cycle$(fmt_cycle "${nsteps}")" ]]; then
      echo "  - missing: ${outdir}/ParaView/Cycle$(fmt_cycle "${nsteps}")"
    fi
    FAILED+=("${cfg_rel}")
    return 1
  fi
}

rc=0
for cfg in "${CFGS[@]}"; do
  run_one "${cfg}" || rc=1
done

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
