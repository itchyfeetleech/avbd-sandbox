#!/usr/bin/env bash
#
# Build a headless, double-precision copy of the authors' reference AVBD solver
# so the JavaScript port can be diffed against it numerically.
#
# The reference sources are fetched into .cache/ (gitignored) rather than
# copied into this test tree. The repository's JavaScript ports retain the
# upstream MIT notice in THIRD_PARTY_NOTICES.md. Only two mechanical
# transformations are applied to the fetched test copy, both explained in
# reference_main.cpp:
#   - strip the OpenGL / Windows includes from solver.h
#   - widen `float` to `double` so precision matches JavaScript numbers
# No solver logic is modified.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="$HERE/.cache"
SRC="$CACHE/avbd-demo3d"
BUILD="$CACHE/build"
UPSTREAM="https://github.com/savant117/avbd-demo3d.git"

mkdir -p "$CACHE"

# --- Fetch the reference implementation -------------------------------------
if [ ! -d "$SRC/source" ]; then
  # Allow an already-cloned copy to be supplied, for offline use.
  if [ -n "${AVBD_REF3D_DIR:-}" ] && [ -d "$AVBD_REF3D_DIR/source" ]; then
    echo "Using reference sources from AVBD_REF3D_DIR=$AVBD_REF3D_DIR"
    mkdir -p "$SRC"
    cp -r "$AVBD_REF3D_DIR/source" "$SRC/"
    cp -f "$AVBD_REF3D_DIR/LICENSE" "$SRC/" 2>/dev/null || true
  else
    echo "Cloning reference implementation from $UPSTREAM ..."
    git clone -q --depth 1 "$UPSTREAM" "$SRC"
  fi
fi

# --- Copy solver sources (no rendering, no scenes) --------------------------
rm -rf "$BUILD"
mkdir -p "$BUILD"

for f in solver.h maths.h solver.cpp rigid.cpp force.cpp joint.cpp spring.cpp manifold.cpp collide.cpp; do
  cp "$SRC/source/$f" "$BUILD/$f"
done

# --- Apply the two mechanical transformations -------------------------------
python3 - "$BUILD" <<'PY'
import re, sys, pathlib

build = pathlib.Path(sys.argv[1])

# 1. Strip the platform / OpenGL preamble from solver.h. Everything between
#    "#pragma once" and the first project include is display plumbing.
header = build / "solver.h"
text = header.read_text()
start = text.index("#pragma once") + len("#pragma once")
end = text.index('#include "maths.h"')
header.write_text(text[:start] + "\n\n#include <cmath>\n\n" + text[end:])

# 2. Widen float -> double. Word boundaries leave float2/float3/float3x3 (which
#    are type NAMES, whose members are rewritten) untouched.
subs = [
    (re.compile(r'\bfloat\b'), 'double'),
    (re.compile(r'\bfabsf\b'), 'fabs'),
    (re.compile(r'\bsqrtf\b'), 'sqrt'),
    (re.compile(r'\bsinf\b'), 'sin'),
    (re.compile(r'\bcosf\b'), 'cos'),
    (re.compile(r'\btanf\b'), 'tan'),
    (re.compile(r'\bacosf\b'), 'acos'),
    (re.compile(r'\bFLT_MAX\b'), 'DBL_MAX'),
    (re.compile(r'\bFLT_EPSILON\b'), 'DBL_EPSILON'),
    # Drop the 'f' suffix on literals: 0.95f is a float constant and would be
    # rounded to float precision before widening, defeating the whole point.
    (re.compile(r'(?<![\w.])(\d+\.?\d*(?:[eE][-+]?\d+)?)[fF](?![\w.])'), r'\1'),
]

for path in build.glob("*"):
    if path.suffix not in (".h", ".cpp"):
        continue
    src = path.read_text()
    for pattern, repl in subs:
        src = pattern.sub(repl, src)
    path.write_text(src)

print("Transformed reference sources -> double precision, headless")
PY

# --- Compile ----------------------------------------------------------------
cp "$HERE/reference_main.cpp" "$BUILD/"

g++ -O2 -std=c++17 -I"$BUILD" \
  "$BUILD"/solver.cpp "$BUILD"/rigid.cpp "$BUILD"/force.cpp "$BUILD"/joint.cpp \
  "$BUILD"/spring.cpp "$BUILD"/manifold.cpp "$BUILD"/collide.cpp \
  "$BUILD"/reference_main.cpp \
  -o "$CACHE/avbd_reference"

echo "Built $CACHE/avbd_reference"
