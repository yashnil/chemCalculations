#!/bin/bash
# Run log_r2 computation - fixes Python path issues

cd /Users/yashnilmohanty/Desktop/chemCalculations

# Add src/ to Python path so autoencoder_model can be imported
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

echo "=================================================================================="
echo "COMPUTING LOG R² FOR ALL OPTIMAL_RETRAINED MODELS"
echo "=================================================================================="
echo ""

# Run the computation
python scripts/compute_log_r2_standalone.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================================================="
    echo "UPDATING COMPARISON METRICS AND PLOTS"
    echo "=================================================================================="
    
    # Then update comparison metrics and plots
    python scripts/update_plots_for_optimal_retrained.py
    
    echo ""
    echo "=================================================================================="
    echo "✅ COMPLETE!"
    echo "=================================================================================="
    echo ""
    echo "All models now have log_r2 values in:"
    echo "  - plots/comparison_metrics.csv"
    echo "  - Individual summary.json files"
    echo ""
    echo "All plots have been regenerated with complete metrics."
else
    echo ""
    echo "⚠️  Script failed. Check error messages above."
    exit 1
fi
