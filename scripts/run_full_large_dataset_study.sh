#!/bin/bash
# run_full_large_dataset_study.sh
# ================================
# Complete pipeline to run large dataset study overnight:
# 1. Run FastChem for x240, x480, x640 (sequential)
# 2. Merge FastChem outputs
# 3. Train models
# 4. Run diagnostics
# 5. Update comparison metrics
#
# Usage:
#   ./run_full_large_dataset_study.sh
#
# Prerequisites:
#   - FASTCHEM_LOGK and FASTCHEM_COND environment variables must be set
#   - FastChem Python bindings installed (pyfastchem)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="$BASE_DIR/large_dataset_study.log"
echo "=== Large Dataset Study Log ===" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if [ -z "$FASTCHEM_LOGK" ] || [ -z "$FASTCHEM_COND" ]; then
        log_error "FASTCHEM_LOGK or FASTCHEM_COND not set!"
        echo "Please set:"
        echo "  export FASTCHEM_LOGK=/path/to/logK.dat"
        echo "  export FASTCHEM_COND=/path/to/logK_condensates.dat"
        exit 1
    fi
    
    if [ ! -f "$FASTCHEM_LOGK" ]; then
        log_error "FASTCHEM_LOGK file not found: $FASTCHEM_LOGK"
        exit 1
    fi
    
    if [ ! -f "$FASTCHEM_COND" ]; then
        log_error "FASTCHEM_COND file not found: $FASTCHEM_COND"
        exit 1
    fi
    
    python3 -c "import pyfastchem" 2>/dev/null || {
        log_error "pyfastchem not installed!"
        echo "Please install FastChem Python bindings"
        exit 1
    }
    
    log_success "Prerequisites check passed"
}

# Run FastChem for a single dataset
run_fastchem_for_dataset() {
    local tag=$1
    local jobs_root="$BASE_DIR/fastchem_jobs_${tag}"
    local output_csv="$BASE_DIR/data/datasets/all_gas_fastchem_${tag}.csv"
    
    log "=========================================="
    log "Processing dataset: $tag"
    log "=========================================="
    
    # Check if dataset already exists
    if [ -f "$output_csv" ]; then
        log_warning "Dataset $tag already exists: $output_csv"
        log "Skipping FastChem generation for $tag"
        return 0
    fi
    
    # Check if job shards exist
    if [ ! -d "$jobs_root" ]; then
        log_error "Job shards not found: $jobs_root"
        log "Run: python scripts/generate_large_datasets.py first"
        return 1
    fi
    
    local num_shards=$(ls -d "$jobs_root"/job_* 2>/dev/null | wc -l | tr -d ' ')
    log "Found $num_shards shards for $tag"
    
    # Run FastChem
    log "Running FastChem for $tag (this may take 2-4 hours)..."
    local fastchem_log="$BASE_DIR/fastchem_${tag}.log"
    
    if python3 scripts/data_generation/run_fastchem_all.py \
        --jobs-root "$jobs_root" \
        --logk "$FASTCHEM_LOGK" \
        --logk-cond "$FASTCHEM_COND" \
        --chunksize 128 >> "$fastchem_log" 2>&1; then
        log_success "FastChem completed for $tag"
    else
        log_error "FastChem failed for $tag (check $fastchem_log)"
        return 1
    fi
    
    # Merge results
    log "Merging FastChem outputs for $tag..."
    local merge_log="$BASE_DIR/merge_${tag}.log"
    
    if python3 scripts/data_generation/merge_fastchem_outputs.py \
        --jobs-root "$jobs_root" \
        --reference-csv "$BASE_DIR/data/datasets/all_gas_fastchem_x160.csv" \
        --output-csv "$output_csv" >> "$merge_log" 2>&1; then
        log_success "Merge completed for $tag"
        
        # Check output file
        if [ -f "$output_csv" ]; then
            local num_rows=$(wc -l < "$output_csv" | tr -d ' ')
            log "Merged dataset has $num_rows rows"
        fi
    else
        log_error "Merge failed for $tag (check $merge_log)"
        return 1
    fi
    
    log_success "Dataset $tag complete!"
}

# Train models
train_models() {
    log "=========================================="
    log "Training models"
    log "=========================================="
    
    python3 scripts/train_large_datasets.py 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Model training completed"
    else
        log_error "Model training failed"
        return 1
    fi
}

# Main execution
main() {
    echo ""
    echo "================================================================================"
    echo "LARGE DATASET STUDY - FULL PIPELINE"
    echo "================================================================================"
    echo ""
    echo "This script will:"
    echo "  1. Run FastChem for x240, x480, x640 (sequential)"
    echo "  2. Merge FastChem outputs"
    echo "  3. Train models"
    echo "  4. Run diagnostics"
    echo "  5. Update comparison metrics"
    echo ""
    echo "Estimated time: 6-12 hours (depending on hardware)"
    echo "Log file: $LOG_FILE"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Process each dataset
    for tag in x240 x480 x640; do
        run_fastchem_for_dataset "$tag"
    done
    
    # Train models
    train_models
    
    # Final summary
    echo ""
    echo "================================================================================"
    echo "STUDY COMPLETE!"
    echo "================================================================================"
    echo ""
    log_success "All datasets processed and models trained"
    echo ""
    echo "Results:"
    echo "  - Comparison metrics: plots/comparison_metrics.csv"
    echo "  - Comparison plot: plots/large_dataset_comparison.png"
    echo "  - Individual diagnostics: runs_autoencoder_*_static_32/diagnostics/"
    echo ""
    echo "Full log: $LOG_FILE"
    echo "Completed: $(date)" >> "$LOG_FILE"
}

# Run main
main "$@"
