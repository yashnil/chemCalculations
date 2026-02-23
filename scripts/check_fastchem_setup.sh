#!/bin/bash
# Check FastChem setup and provide guidance

echo "=================================================================================="
echo "FASTCHEM SETUP CHECK"
echo "=================================================================================="
echo ""

# Check environment variables
echo "Environment Variables:"
if [ -n "$FASTCHEM_LOGK" ]; then
    if [ -f "$FASTCHEM_LOGK" ]; then
        echo "  ✓ FASTCHEM_LOGK: $FASTCHEM_LOGK (file exists)"
    else
        echo "  ✗ FASTCHEM_LOGK: $FASTCHEM_LOGK (file NOT found)"
    fi
else
    echo "  ✗ FASTCHEM_LOGK: NOT SET"
fi

if [ -n "$FASTCHEM_COND" ]; then
    if [ -f "$FASTCHEM_COND" ]; then
        echo "  ✓ FASTCHEM_COND: $FASTCHEM_COND (file exists)"
    else
        echo "  ✗ FASTCHEM_COND: $FASTCHEM_COND (file NOT found)"
    fi
else
    echo "  ✗ FASTCHEM_COND: NOT SET"
fi

if [ -n "$FASTCHEM_ELEM" ]; then
    if [ -f "$FASTCHEM_ELEM" ]; then
        echo "  ✓ FASTCHEM_ELEM: $FASTCHEM_ELEM (file exists)"
    else
        echo "  ✗ FASTCHEM_ELEM: $FASTCHEM_ELEM (file NOT found)"
    fi
else
    echo "  ⚠️  FASTCHEM_ELEM: NOT SET (will try to infer from logK path)"
fi

echo ""
echo "Common FastChem Installation Locations:"
echo "  - /usr/local/share/FastChem/tables/logK.dat"
echo "  - ~/FastChem/tables/logK.dat"
echo "  - /opt/FastChem/tables/logK.dat"
echo ""

# Try to find FastChem files
echo "Searching for FastChem files..."
FOUND_LOGK=$(find ~ /usr/local /opt 2>/dev/null -name "logK.dat" -type f 2>/dev/null | head -1)
FOUND_COND=$(find ~ /usr/local /opt 2>/dev/null -name "logK_condensates.dat" -type f 2>/dev/null | head -1)
FOUND_ELEM=$(find ~ /usr/local /opt 2>/dev/null -path "*/element_abundances/*.dat" -type f 2>/dev/null | head -1)

if [ -n "$FOUND_LOGK" ]; then
    echo "  Found logK.dat: $FOUND_LOGK"
fi
if [ -n "$FOUND_COND" ]; then
    echo "  Found logK_condensates.dat: $FOUND_COND"
fi
if [ -n "$FOUND_ELEM" ]; then
    echo "  Found element abundances: $FOUND_ELEM"
fi

echo ""
echo "To set up FastChem, run:"
if [ -n "$FOUND_LOGK" ] && [ -n "$FOUND_COND" ]; then
    echo "  export FASTCHEM_LOGK=\"$FOUND_LOGK\""
    echo "  export FASTCHEM_COND=\"$FOUND_COND\""
    if [ -n "$FOUND_ELEM" ]; then
        echo "  export FASTCHEM_ELEM=\"$FOUND_ELEM\""
    fi
else
    echo "  export FASTCHEM_LOGK=\"/path/to/FastChem/tables/logK.dat\""
    echo "  export FASTCHEM_COND=\"/path/to/FastChem/tables/logK_condensates.dat\""
    echo "  export FASTCHEM_ELEM=\"/path/to/FastChem/element_abundances/asplund_2009.dat\""
fi

echo ""
echo "Then run: bash scripts/complete_pipeline.sh"
