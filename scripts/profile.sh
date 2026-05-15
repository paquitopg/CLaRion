#!/usr/bin/env bash
# CLaRiON profiling helper.
#
# Picks the right profiler for your platform and avoids the common macOS
# traps (double `uv run`, SIP, hardened-runtime). Tries py-spy first, falls
# back to austin, then cProfile.
#
# Usage:
#     scripts/profile.sh                              # bench_pipeline, py-spy native flamegraph
#     scripts/profile.sh bench_encoder                # different target
#     scripts/profile.sh bench_matmul --include-large
#     PROFILER=austin scripts/profile.sh              # force austin
#     PROFILER=cprofile scripts/profile.sh            # force cProfile
#
# Output:
#     reports/profile_<TARGET>_<PROFILER>.<ext>
#         .svg for py-spy / austin (open in a browser)
#         .prof for cProfile (view with `uv run snakeviz <file>`)

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- defaults ---- #
TARGET="${1:-bench_pipeline}"; shift || true
EXTRA_ARGS=("$@")
PROFILER="${PROFILER:-auto}"
RATE="${RATE:-250}"
DURATION="${DURATION:-30}"

VENV_PY="${VENV_PY:-.venv/bin/python}"
REPORTS_DIR="reports"
mkdir -p "${REPORTS_DIR}"

# Locate the actual Python binary so the profiler attaches to it directly,
# not to a `uv run` launcher wrapping it. This is the #1 macOS gotcha.
if [[ ! -x "${VENV_PY}" ]]; then
    echo "Cannot find venv Python at ${VENV_PY}." >&2
    echo "Run 'uv sync' or set VENV_PY to your interpreter path." >&2
    exit 1
fi

# Verify the Cython extensions are built — profiling without them just shows
# numpy at the top of the stack and the whole exercise is wasted.
if ! "${VENV_PY}" -c "from src.parallel import cython_encoder, cython_index" 2>/dev/null; then
    echo "Cython extensions not loaded. Run:" >&2
    echo "    uv run python setup.py build_ext --inplace" >&2
    exit 1
fi

TARGET_CMD=("${VENV_PY}" -m "src.benchmarks.${TARGET}" "${EXTRA_ARGS[@]}")

case "${PROFILER}" in
    auto)
        if command -v py-spy >/dev/null 2>&1 || "${VENV_PY}" -c "import py_spy" 2>/dev/null; then
            PROFILER=pyspy
        elif command -v austin >/dev/null 2>&1; then
            PROFILER=austin
        else
            PROFILER=cprofile
        fi
        ;;
esac

OUT_BASE="${REPORTS_DIR}/profile_${TARGET}_${PROFILER}"

case "${PROFILER}" in
    pyspy)
        OUT="${OUT_BASE}.svg"
        echo "Profiling with py-spy → ${OUT}"
        # --native captures C/Cython stack frames too.
        # On macOS, the venv python may be hardened-runtime; if py-spy errors
        # with "Failed to find python version from target process", try PID
        # attach (see README's profiling section) or fall back to austin.
        sudo py-spy record --native --rate "${RATE}" -o "${OUT}" -- \
             "${TARGET_CMD[@]}"
        echo "Open ${OUT} in a browser."
        ;;

    austin)
        OUT="${OUT_BASE}.svg"
        AUSTIN_LOG="${OUT_BASE}.austin.log"
        echo "Profiling with austin → ${OUT}"
        austin -i "${RATE}" -o "${AUSTIN_LOG}" -- "${TARGET_CMD[@]}"
        if command -v flamegraph.pl >/dev/null 2>&1 && command -v austin2flame >/dev/null 2>&1; then
            austin2flame "${AUSTIN_LOG}" | flamegraph.pl > "${OUT}"
            echo "Open ${OUT} in a browser."
        else
            echo "Install flamegraph.pl and austin-tui to render: brew install flamegraph"
            echo "Raw log: ${AUSTIN_LOG}"
        fi
        ;;

    cprofile)
        OUT="${OUT_BASE}.prof"
        echo "Profiling with cProfile → ${OUT}"
        echo "(cProfile only captures Python-level frames, not Cython internals.)"
        "${VENV_PY}" -m cProfile -o "${OUT}" -m "src.benchmarks.${TARGET}" "${EXTRA_ARGS[@]}"
        echo "View with: uv run snakeviz ${OUT}"
        ;;

    *)
        echo "Unknown PROFILER='${PROFILER}'. Choose: pyspy | austin | cprofile | auto." >&2
        exit 1
        ;;
esac
