#!/bin/bash
# cleanup_fastchem_jobs.sh
# Clean up FastChem job directories for datasets that have already been merged into CSV files

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

JOBS_DIR="results/fastchem_jobs"
DATASETS_DIR="data/datasets"

echo "=================================================================================="
echo "CLEANING UP FASTCHEM JOB DIRECTORIES"
echo "=================================================================================="
echo ""
echo "This script will remove FastChem job directories for datasets that have already"
echo "been merged into CSV files. This will free up significant disk space."
echo ""
echo "⚠️  SAFETY CHECKS:"
echo "  - Only deletes job directories if corresponding CSV exists AND is valid"
echo "  - CSV must have > 10 lines and > 1KB size"
echo "  - Shows exactly what will be deleted before asking for confirmation"
echo "  - You can review the list and abort if anything looks wrong"
echo ""

# Find all CSV files
echo "Checking which datasets have been merged..."
MERGED_DATASETS=()
for csv_file in "$DATASETS_DIR"/all_gas_fastchem_x*.csv; do
    if [ -f "$csv_file" ]; then
        basename=$(basename "$csv_file" .csv)
        tag=${basename#all_gas_fastchem_}
        MERGED_DATASETS+=("$tag")
    fi
done

echo "Found merged datasets: ${MERGED_DATASETS[*]}"
echo ""

# Find all job directories
echo "Checking FastChem job directories..."
JOBS_TO_DELETE=()
JOBS_TO_KEEP=()

for job_dir in "$JOBS_DIR"/fastchem_jobs_x*; do
    if [ -d "$job_dir" ]; then
        basename=$(basename "$job_dir")
        tag=${basename#fastchem_jobs_}
        
        # Check if corresponding CSV exists and is valid
        csv_file="$DATASETS_DIR/all_gas_fastchem_${tag}.csv"
        if [ -f "$csv_file" ]; then
            # Verify CSV is not empty and has reasonable size
            csv_size=$(wc -l < "$csv_file" 2>/dev/null || echo "0")
            csv_file_size=$(stat -f%z "$csv_file" 2>/dev/null || stat -c%s "$csv_file" 2>/dev/null || echo "0")
            
            # CSV should have at least header + some data rows, and be > 1KB
            if [ "$csv_size" -gt 10 ] && [ "$csv_file_size" -gt 1024 ]; then
                JOBS_TO_DELETE+=("$job_dir")
                echo "  ✓ $tag: CSV exists and is valid ($csv_size lines, $(numfmt --to=iec-i --suffix=B $csv_file_size 2>/dev/null || echo "${csv_file_size} bytes")), can delete job directory"
            else
                JOBS_TO_KEEP+=("$job_dir")
                echo "  ⚠️  $tag: CSV exists but appears invalid (size: $csv_size lines), keeping job directory"
            fi
        else
            JOBS_TO_KEEP+=("$job_dir")
            echo "  ⚠️  $tag: No CSV found, keeping job directory"
        fi
    fi
done

echo ""
echo "=================================================================================="
echo "SUMMARY"
echo "=================================================================================="
echo ""
echo "Job directories to DELETE (${#JOBS_TO_DELETE[@]}):"
for job_dir in "${JOBS_TO_DELETE[@]}"; do
    size=$(du -sh "$job_dir" 2>/dev/null | cut -f1)
    basename=$(basename "$job_dir")
    tag=${basename#fastchem_jobs_}
    csv_file="$DATASETS_DIR/all_gas_fastchem_${tag}.csv"
    csv_lines=$(wc -l < "$csv_file" 2>/dev/null || echo "0")
    echo "  - $job_dir ($size) → CSV has $csv_lines lines"
done

echo ""
echo "Job directories to KEEP (${#JOBS_TO_KEEP[@]}):"
for job_dir in "${JOBS_TO_KEEP[@]}"; do
    size=$(du -sh "$job_dir" 2>/dev/null | cut -f1)
    echo "  - $job_dir ($size)"
done

# Calculate total space to be freed
TOTAL_SIZE=0
for job_dir in "${JOBS_TO_DELETE[@]}"; do
    size_bytes=$(du -sk "$job_dir" 2>/dev/null | cut -f1)
    TOTAL_SIZE=$((TOTAL_SIZE + size_bytes))
done

TOTAL_SIZE_GB=$((TOTAL_SIZE / 1024 / 1024))
echo ""
echo "Total space to be freed: ~${TOTAL_SIZE_GB}GB"
echo ""

# Ask for confirmation
read -p "Delete these job directories? (y/n): " response
if [ "$response" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# Delete job directories
echo ""
echo "Deleting job directories..."
for job_dir in "${JOBS_TO_DELETE[@]}"; do
    echo "  Deleting: $job_dir"
    rm -rf "$job_dir"
done

echo ""
echo "=================================================================================="
echo "✅ CLEANUP COMPLETE!"
echo "=================================================================================="
echo ""
echo "Freed approximately ${TOTAL_SIZE_GB}GB of disk space."
echo ""
echo "You can now:"
echo "  1. Merge x2400 dataset: python scripts/data_generation/merge_fastchem_outputs.py \\"
echo "       --jobs-root results/fastchem_jobs/fastchem_jobs_x2400 \\"
echo "       --reference-csv data/datasets/all_gas_fastchem_x160.csv \\"
echo "       --output-csv data/datasets/all_gas_fastchem_x2400.csv"
echo "  2. Generate and train x2720 and x3040 models"
