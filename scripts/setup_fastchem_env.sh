#!/bin/bash
# Setup FastChem environment variables
# All thermodynamic data files are stored within the project under data/fastchem_data/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export FASTCHEM_LOGK="$PROJECT_ROOT/data/fastchem_data/Kitzmann2023/logK.dat"
export FASTCHEM_COND="$PROJECT_ROOT/data/fastchem_data/Kitzmann2023/logK_condensates.dat"
export FASTCHEM_ELEM="$PROJECT_ROOT/data/fastchem_data/lodders_2003_extended.dat"

# Verify files exist
for f in "$FASTCHEM_LOGK" "$FASTCHEM_COND" "$FASTCHEM_ELEM"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing FastChem data file: $f"
        return 1 2>/dev/null || exit 1
    fi
done

echo "FastChem environment variables set:"
echo "  FASTCHEM_LOGK=$FASTCHEM_LOGK"
echo "  FASTCHEM_COND=$FASTCHEM_COND"
echo "  FASTCHEM_ELEM=$FASTCHEM_ELEM"
