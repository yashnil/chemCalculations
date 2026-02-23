#!/bin/bash
# Run log_r2 computation outside sandbox

cd /Users/yashnilmohanty/Desktop/chemCalculations
python scripts/compute_log_r2_standalone.py

# Then update comparison metrics and plots
python scripts/update_plots_for_optimal_retrained.py

echo ""
echo "✅ Log R² computed and plots updated!"
