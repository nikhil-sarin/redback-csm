#!/usr/bin/env bash
# Compile the Fortran CSM extension using f2py.
# The resulting csm.cpython-*.so is placed inside redback_csm/ so that
# `import csm` works from within the package.
#
# Usage:
#   bash setup_fortran.sh
#
# After running this, install the package in editable mode:
#   pip install -e .

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORTRAN_DIR="$SCRIPT_DIR/fortran"
OUT_DIR="$SCRIPT_DIR/redback_csm"

echo "Compiling Fortran CSM extension ..."
cd "$OUT_DIR"
python -m numpy.f2py -c \
    "$FORTRAN_DIR/csm.pyf" \
    "$FORTRAN_DIR/constants.f90" \
    "$FORTRAN_DIR/get_vals.f90" \
    "$FORTRAN_DIR/integration.f90" \
    "$FORTRAN_DIR/csm.f90" \
    -m csm

echo "Done. Extension compiled to $OUT_DIR/csm*.so"
