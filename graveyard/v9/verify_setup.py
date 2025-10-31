#!/usr/bin/env python3
"""
Quick verification script for v9 setup
Run this before executing the full pipeline to catch any issues early.
"""

import sys
import os

def check_imports():
    """Verify all required packages are installed."""
    print("=" * 60)
    print("CHECKING IMPORTS...")
    print("=" * 60)
    
    required = {
        "numpy": "np",
        "pandas": "pd",
        "tensorflow": "tf",
        "sklearn": "sklearn",
        "joblib": "joblib",
        "optuna": "optuna",
        "matplotlib": "plt"
    }
    
    missing = []
    for pkg, alias in required.items():
        try:
            if pkg == "sklearn":
                import sklearn
                print(f"✅ {pkg:12s} version {sklearn.__version__}")
            elif pkg == "tensorflow":
                import tensorflow as tf
                print(f"✅ {pkg:12s} version {tf.__version__}")
            elif pkg == "matplotlib":
                import matplotlib.pyplot as plt
                import matplotlib
                print(f"✅ {pkg:12s} version {matplotlib.__version__}")
            else:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")
                print(f"✅ {pkg:12s} version {ver}")
        except ImportError:
            print(f"❌ {pkg:12s} NOT FOUND")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        return False
    return True

def check_data_path():
    """Check if data file exists."""
    print("\n" + "=" * 60)
    print("CHECKING DATA PATH...")
    print("=" * 60)
    
    csv_path = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
    
    if os.path.exists(csv_path):
        size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"✅ Data file found: {csv_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Data file NOT FOUND: {csv_path}")
        print("   Update CSV_PATH in utils.py if data is elsewhere")
        return False

def check_v9_scripts():
    """Verify all v9 scripts are present."""
    print("\n" + "=" * 60)
    print("CHECKING V9 SCRIPTS...")
    print("=" * 60)
    
    required_scripts = [
        "utils.py",
        "baseline_checks.py",
        "train_baseline.py",
        "tune.py",
        "finalize.py",
        "losses.py",
        "model_heads.py",
        "diagnostics.py"
    ]
    
    all_present = True
    for script in required_scripts:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} NOT FOUND")
            all_present = False
    
    return all_present

def test_utils_import():
    """Test importing and using utils.py."""
    print("\n" + "=" * 60)
    print("TESTING UTILS.PY...")
    print("=" * 60)
    
    try:
        from utils import _preprocess_dataframe, ARTE_DIR, CSV_PATH
        print(f"✅ utils.py imports successfully")
        print(f"   Artefacts directory: {ARTE_DIR}")
        print(f"   CSV path: {CSV_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error importing utils.py:")
        print(f"   {e}")
        return False

def test_model_heads():
    """Test model head creation."""
    print("\n" + "=" * 60)
    print("TESTING MODEL HEADS...")
    print("=" * 60)
    
    try:
        from model_heads import softplus_head
        import tensorflow as tf
        from tensorflow import keras
        
        # Try creating a simple model
        test_model = keras.Sequential([
            keras.layers.Input((6,)),
            keras.layers.Dense(32, activation="gelu"),
            softplus_head(116)
        ])
        
        print(f"✅ Model head works")
        print(f"   Test model created with 6 inputs")
        print(f"   Output shape: {test_model.output_shape}")
        
        # Test with dummy data
        import numpy as np
        dummy_input = np.random.randn(5, 6).astype("float32")
        dummy_output = test_model.predict(dummy_input, verbose=0)
        
        # Check normalization
        row_sums = dummy_output.sum(axis=1)
        if np.allclose(row_sums, 1.0, atol=1e-4):
            print(f"✅ Output normalization correct (Σ ≈ 1.0)")
        else:
            print(f"⚠️  Output sums: {row_sums} (should be 1.0)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model heads:")
        print(f"   {e}")
        return False

def test_losses():
    """Test loss functions."""
    print("\n" + "=" * 60)
    print("TESTING LOSS FUNCTIONS...")
    print("=" * 60)
    
    try:
        from losses import composite_loss, _mae_log
        import tensorflow as tf
        import numpy as np
        
        # Create dummy data
        y_true = np.random.rand(10, 116).astype("float32")
        y_true = y_true / y_true.sum(axis=1, keepdims=True)
        y_pred = np.random.rand(10, 116).astype("float32")
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        
        # Test loss
        loss_fn = composite_loss(lam=0.1)
        loss_val = loss_fn(tf.constant(y_true), tf.constant(y_pred))
        
        print(f"✅ Loss functions work")
        print(f"   Test loss value: {float(loss_val.numpy().mean()):.4e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing losses:")
        print(f"   {e}")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("V9 SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    checks = [
        ("Package imports", check_imports),
        ("Data file", check_data_path),
        ("V9 scripts", check_v9_scripts),
        ("Utils module", test_utils_import),
        ("Model heads", test_model_heads),
        ("Loss functions", test_losses),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}:")
            print(f"   {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8s} {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Ready to run the pipeline.")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. python baseline_checks.py")
        print("  2. python train_baseline.py")
        print("  3. python tune.py")
        print("  4. python finalize.py")
        print("  5. python diagnostics.py")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED. Fix issues before proceeding.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

