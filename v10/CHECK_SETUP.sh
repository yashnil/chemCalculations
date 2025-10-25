#!/bin/bash
# Quick setup verification for v10

echo "=========================================="
echo "v10 Setup Verification"
echo "=========================================="
echo ""

# Check Python
echo "1. Checking Python..."
python --version || echo "❌ Python not found"
echo ""

# Check dependencies
echo "2. Checking dependencies..."
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>/dev/null || echo "❌ PyTorch not found"
python -c "import pandas; print('✅ Pandas:', pandas.__version__)" 2>/dev/null || echo "❌ Pandas not found"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)" 2>/dev/null || echo "❌ NumPy not found"
python -c "import sklearn; print('✅ Scikit-learn:', sklearn.__version__)" 2>/dev/null || echo "❌ Scikit-learn not found"
echo ""

# Check data file
echo "3. Checking data file..."
DATA_PATH="/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
if [ -f "$DATA_PATH" ]; then
    SIZE=$(ls -lh "$DATA_PATH" | awk '{print $5}')
    echo "✅ Data file found: $SIZE"
else
    echo "❌ Data file not found: $DATA_PATH"
    echo "   Update CSV_PATH in run_mlp.py"
fi
echo ""

# Check v10 files
echo "4. Checking v10 files..."
for file in run_mlp.py plot.py investigate.py README.md DOCUMENTATION.md START_HERE.md SUMMARY.md; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file missing"
    fi
done
echo ""

# Summary
echo "=========================================="
echo "Setup Check Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. If all checks passed: python run_mlp.py"
echo "2. If dependencies missing: pip install torch pandas numpy scikit-learn"
echo "3. If data file missing: Update CSV_PATH in run_mlp.py"
echo ""
echo "Documentation:"
echo "- Quick start: START_HERE.md"
echo "- Full docs: DOCUMENTATION.md"
echo "- Comparison: COMPARISON_WITH_V8_V9.md"
echo ""

