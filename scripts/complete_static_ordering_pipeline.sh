#!/bin/bash
# Complete pipeline for static ordering comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==================================================================="
echo "COMPLETING STATIC ORDERING PIPELINE"
echo "==================================================================="
echo ""

# Check if models are still training
check_training() {
    local run_dir=$1
    local loss_file="$run_dir/loss_history.csv"
    
    if [ ! -f "$loss_file" ]; then
        echo "  ⏳ Not started"
        return 1
    fi
    
    local epochs=$(tail -n +2 "$loss_file" | wc -l | tr -d ' ')
    if [ "$epochs" -ge 200 ]; then
        echo "  ✅ Complete ($epochs epochs)"
        return 0
    else
        echo "  ⏳ In progress ($epochs/200 epochs)"
        return 1
    fi
}

echo "Checking training status..."
echo "  x160_static_24: $(check_training results/runs/runs_autoencoder_x160_static_24 && echo 'Ready' || echo 'Training...')"
echo "  x160_static_32: $(check_training results/runs/runs_autoencoder_x160_static_32 && echo 'Ready' || echo 'Training...')"
echo "  x160_static_36: $(check_training results/runs/runs_autoencoder_x160_static_36 && echo 'Ready' || echo 'Training...')"
echo ""

# Function to run diagnostics and update metrics
process_model() {
    local model_name=$1
    local run_dir="results/runs/runs_autoencoder_${model_name}"
    local csv_path="data/datasets/all_gas_fastchem_x160.csv"
    
    echo "Processing $model_name..."
    
    # Check if training is complete
    if [ ! -f "$run_dir/loss_history.csv" ]; then
        echo "  ⚠️  Training not started, skipping"
        return
    fi
    
    local epochs=$(tail -n +2 "$run_dir/loss_history.csv" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$epochs" -lt 200 ]; then
        echo "  ⚠️  Training not complete ($epochs/200 epochs), skipping"
        return
    fi
    
    # Run diagnostics if not already done
    if [ ! -f "$run_dir/diagnostics/global_metrics.txt" ]; then
        echo "  Running diagnostics..."
        CSV_PATH="$csv_path" BEST_MODULE="$run_dir/best_model.py" \
            python src/diagnostics.py --output-dir "$run_dir/diagnostics" 2>&1 | grep -E "(INFO|ERROR)" | tail -5
    else
        echo "  ✅ Diagnostics already complete"
    fi
    
    # Update comparison metrics
    echo "  Updating comparison_metrics.csv..."
    python scripts/update_comparison_metrics.py 2>&1 | grep -E "(Added|Skipped|Error)" | tail -3
}

# Process all models
for model in "x160_static_24" "x160_static_32" "x160_static_36"; do
    process_model "$model"
    echo ""
done

echo "==================================================================="
echo "GENERATING COMPARISON"
echo "==================================================================="
echo ""

python << 'PYTHON_SCRIPT'
import pandas as pd

df = pd.read_csv("plots/comparison_metrics.csv")

# Filter static ordering models
static_models = df[df["dataset"].str.contains("static", na=False)].copy()
baseline = df[df["dataset"] == "x160_logratio"]

if len(static_models) > 0:
    print("Static Ordering Results:")
    print("="*80)
    print()
    for _, row in static_models.iterrows():
        n_species = row["dataset"].split("_")[-1]
        print(f"{row['dataset']:25s} ({n_species} species):")
        print(f"  Test Loss: {row['test_loss']:.6f}")
        print(f"  Log MAE:   {row['log_mae']:.6f}")
        print(f"  Log R²:    {row['log_r2']:.6f}")
        print()
    
    if len(baseline) > 0:
        b = baseline.iloc[0]
        print("Baseline (x160_logratio, dynamic top-20):")
        print(f"  Test Loss: {b['test_loss']:.6f}")
        print(f"  Log MAE:   {b['log_mae']:.6f}")
        print(f"  Log R²:    {b['log_r2']:.6f}")
        print()
        
        # Find best static model
        best_static = static_models.loc[static_models['test_loss'].idxmin()]
        print("Best Static Model:", best_static['dataset'])
        print()
        
        print("Comparison:")
        loss_diff = ((b['test_loss'] - best_static['test_loss']) / b['test_loss']) * 100
        mae_diff = ((b['log_mae'] - best_static['log_mae']) / b['log_mae']) * 100
        print(f"  Test Loss: {loss_diff:+.2f}%")
        print(f"  Log MAE:   {mae_diff:+.2f}%")
else:
    print("No static ordering models found in comparison_metrics.csv")
PYTHON_SCRIPT

echo ""
echo "✅ Pipeline complete!"
echo ""
echo "Next steps:"
echo "  1. Review plots/comparison_metrics.csv"
echo "  2. Check diagnostic plots in results/runs/runs_autoencoder_*/diagnostics/"
echo ""

