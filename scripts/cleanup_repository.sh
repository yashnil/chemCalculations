#!/bin/bash
# cleanup_repository.sh
# Clean up unnecessary files and organize repository

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "CLEANING UP REPOSITORY"
echo "=================================================================================="
echo ""

# Step 1: Remove Python cache files
echo "Step 1: Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.py~" -delete 2>/dev/null || true
echo "  ✓ Removed __pycache__ directories and .pyc files"

# Step 2: Remove .DS_Store files (macOS)
echo ""
echo "Step 2: Removing .DS_Store files..."
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
echo "  ✓ Removed .DS_Store files"

# Step 3: Remove old log files in root
echo ""
echo "Step 3: Cleaning up log files..."
LOG_FILES=(
    "fastchem_x320.log"
    "fastchem_x800.log"
    "merge_x320.log"
    "merge_x800.log"
    "training_log.txt"
    "training_x160K.log"
    "training_x320K.log"
    "training_x480K.log"
    "training_x640K.log"
    "training_x800K.log"
)

for log_file in "${LOG_FILES[@]}"; do
    if [ -f "$log_file" ]; then
        rm "$log_file"
        echo "  ✓ Removed $log_file"
    fi
done

# Step 4: Remove temporary/outdated scripts
echo ""
echo "Step 4: Checking for outdated scripts..."
# Keep all scripts for now, but note which ones might be outdated

# Step 5: Organize documentation
echo ""
echo "Step 5: Organizing documentation..."
# Ensure all docs are in docs/ directory

# Step 6: Summary
echo ""
echo "=================================================================================="
echo "CLEANUP COMPLETE"
echo "=================================================================================="
echo ""
echo "Removed:"
echo "  - Python cache files (__pycache__, *.pyc)"
echo "  - macOS .DS_Store files"
echo "  - Old log files"
echo ""
echo "Repository is now cleaner and more organized."
