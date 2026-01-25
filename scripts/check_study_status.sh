#!/bin/bash
# Quick status check for the large dataset study

echo "=================================================================================="
echo "LARGE DATASET STUDY STATUS CHECK"
echo "=================================================================================="
echo ""

# Check if main process is running
MAIN_PID=$(ps aux | grep "run_full_large_dataset_study.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$MAIN_PID" ]; then
    echo "✅ Main study process running (PID: $MAIN_PID)"
    echo "   Runtime: $(ps -p $MAIN_PID -o etime= 2>/dev/null || echo 'unknown')"
else
    echo "❌ Main study process not running"
fi

echo ""

# Check FastChem processes
FASTCHEM_PIDS=$(ps aux | grep "run_fastchem_all.py" | grep -v grep | awk '{print $2}')
if [ -n "$FASTCHEM_PIDS" ]; then
    echo "✅ FastChem processes running:"
    for pid in $FASTCHEM_PIDS; do
        echo "   PID $pid: $(ps -p $pid -o etime= 2>/dev/null || echo 'unknown')"
    done
else
    echo "⏸️  No FastChem processes running (may be between datasets)"
fi

echo ""

# Check log file
if [ -f "large_dataset_study.log" ]; then
    echo "📄 Latest log entries:"
    tail -10 large_dataset_study.log | sed 's/^/   /'
else
    echo "⚠️  Log file not found"
fi

echo ""
echo "=================================================================================="
echo "To monitor in real-time:"
echo "  tail -f large_dataset_study.log"
echo "=================================================================================="
