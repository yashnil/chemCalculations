#!/bin/bash
# Setup FastChem environment variables with found paths

export FASTCHEM_LOGK="/Users/yashnilmohanty/Desktop/GGchem-master/data/Kitzmann2023/logK.dat"
export FASTCHEM_COND="/Users/yashnilmohanty/Desktop/GGchem-master/data/Kitzmann2023/logK_condensates.dat"
export FASTCHEM_ELEM="/Users/yashnilmohanty/Downloads/FastChem-master/input/element_abundances/lodders_2003_extended.dat"

echo "FastChem environment variables set:"
echo "  FASTCHEM_LOGK=$FASTCHEM_LOGK"
echo "  FASTCHEM_COND=$FASTCHEM_COND"
echo "  FASTCHEM_ELEM=$FASTCHEM_ELEM"
echo ""
echo "To use in current shell, run:"
echo "  source scripts/setup_fastchem_env.sh"
echo ""
echo "Then run: bash scripts/complete_pipeline.sh"
