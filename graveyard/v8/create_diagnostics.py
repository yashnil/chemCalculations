#!/usr/bin/env python3
"""
Generate diagnostic plots for the trained model
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import keras
from utils import load_XY
from config import DEFAULT_CONFIG
from improved_model_heads import softmax_T_head

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_model_and_data():
    """Load trained model and test data"""
    model_path = Path("artefacts/improved_model.keras")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Define custom objects for model loading
    custom_objects = {
        'temp_scale': lambda x: x / 0.5,  # Reconstruct the temp_scale function
    }
    
    model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, species_cols = load_XY()
    
    return model, X_test, Y_test, species_cols

def create_parity_plot(y_true, y_pred, title="Parity Plot", save_path=None):
    """Create log-log parity plot"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove zeros for log plot
    mask = (y_true_flat > 0) & (y_pred_flat > 0)
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]
    
    # Create hexbin plot
    hexbin = ax.hexbin(y_true_flat, y_pred_flat, 
                       xscale='log', yscale='log',
                       gridsize=50, cmap='viridis', 
                       mincnt=1, linewidths=0.2)
    
    # Add 1:1 line
    lims = [
        max(y_true_flat.min(), y_pred_flat.min()),
        min(y_true_flat.max(), y_pred_flat.max())
    ]
    ax.plot(lims, lims, 'r--', alpha=0.8, lw=2, label='1:1 line')
    
    # Labels and formatting
    ax.set_xlabel('True Abundance', fontsize=12)
    ax.set_ylabel('Predicted Abundance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cb = plt.colorbar(hexbin, ax=ax)
    cb.set_label('Count', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def create_top_species_plots(y_true, y_pred, species_cols, n_species=10, save_dir=None):
    """Create individual parity plots for top N species"""
    # Calculate mean abundances
    mean_abundances = y_true.mean(axis=0)
    top_indices = np.argsort(mean_abundances)[::-1][:n_species]
    
    # Create subplot grid
    n_cols = 3
    n_rows = (n_species + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_species > 1 else [axes]
    
    for idx, species_idx in enumerate(top_indices):
        ax = axes[idx]
        species_name = species_cols[species_idx]
        
        y_true_sp = y_true[:, species_idx]
        y_pred_sp = y_pred[:, species_idx]
        
        # Remove zeros
        mask = (y_true_sp > 0) & (y_pred_sp > 0)
        y_true_sp = y_true_sp[mask]
        y_pred_sp = y_pred_sp[mask]
        
        if len(y_true_sp) > 0:
            # Scatter plot
            ax.hexbin(y_true_sp, y_pred_sp, 
                     xscale='log', yscale='log',
                     gridsize=30, cmap='viridis', 
                     mincnt=1, linewidths=0.2)
            
            # 1:1 line
            lims = [
                max(y_true_sp.min(), y_pred_sp.min()),
                min(y_true_sp.max(), y_pred_sp.max())
            ]
            ax.plot(lims, lims, 'r--', alpha=0.8, lw=1.5)
            
            ax.set_xlabel('True', fontsize=9)
            ax.set_ylabel('Predicted', fontsize=9)
            ax.set_title(f'{species_name}\n(rank {idx+1})', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_species, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_dir:
        save_path = Path(save_dir) / "top_species_parity.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def create_error_distribution(y_true, y_pred, save_path=None):
    """Create error distribution plots"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Log-space error
    y_true_log = np.log10(y_true + 1e-12)
    y_pred_log = np.log10(y_pred + 1e-12)
    errors = (y_pred_log - y_true_log).flatten()
    
    # Histogram
    axes[0].hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    axes[0].set_xlabel('Log10 Error (Predicted - True)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def create_learning_curve_plot(save_path=None):
    """Create learning curve plot from saved data"""
    lc_path = Path("artefacts/learning_curve.csv")
    if not lc_path.exists():
        print(f"Learning curve data not found at {lc_path}")
        return None
    
    df = pd.read_csv(lc_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE vs dataset size
    axes[0].plot(df['dataset_size'], df['mae'], 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[0].set_xlabel('Dataset Size', fontsize=11)
    axes[0].set_ylabel('Test MAE', fontsize=11)
    axes[0].set_title('MAE vs Dataset Size', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    # R² vs dataset size
    axes[1].plot(df['dataset_size'], df['r2'], 'o-', linewidth=2, markersize=8, color='darkgreen')
    axes[1].set_xlabel('Dataset Size', fontsize=11)
    axes[1].set_ylabel('Test R²', fontsize=11)
    axes[1].set_title('R² vs Dataset Size', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def create_residual_plot(y_true, y_pred, save_path=None):
    """Create residual plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_true_log = np.log10(y_true.flatten() + 1e-12)
    y_pred_log = np.log10(y_pred.flatten() + 1e-12)
    residuals = y_pred_log - y_true_log
    
    hexbin = ax.hexbin(y_true_log, residuals, 
                       gridsize=50, cmap='viridis', 
                       mincnt=1, linewidths=0.2)
    
    ax.axhline(0, color='red', linestyle='--', lw=2, alpha=0.8, label='Zero error')
    ax.set_xlabel('Log10(True Abundance)', fontsize=11)
    ax.set_ylabel('Log10(Predicted - True)', fontsize=11)
    ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cb = plt.colorbar(hexbin, ax=ax)
    cb.set_label('Count', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def main():
    """Generate all diagnostic plots"""
    print("="*60)
    print("Generating Diagnostic Plots")
    print("="*60)
    
    # Create output directory
    output_dir = Path("artefacts/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    print("\nLoading model and data...")
    model, X_test, Y_test, species_cols = load_model_and_data()
    
    # Make predictions
    print("Making predictions...")
    Y_pred = model.predict(X_test, verbose=0)
    
    # Ensure sum to 1
    Y_pred = Y_pred / (Y_pred.sum(axis=1, keepdims=True) + 1e-12)
    Y_test = Y_test / (Y_test.sum(axis=1, keepdims=True) + 1e-12)
    
    print(f"\nTest set shape: {Y_test.shape}")
    print(f"Predictions shape: {Y_pred.shape}")
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    print("\n1. Overall Parity Plot")
    create_parity_plot(Y_test, Y_pred, 
                      title="Overall Parity Plot (Log-Log Scale)",
                      save_path=output_dir / "overall_parity.png")
    
    print("\n2. Top 10 Species Parity Plots")
    create_top_species_plots(Y_test, Y_pred, species_cols,
                            n_species=10, save_dir=output_dir)
    
    print("\n3. Error Distribution")
    create_error_distribution(Y_test, Y_pred,
                             save_path=output_dir / "error_distribution.png")
    
    print("\n4. Residual Plot")
    create_residual_plot(Y_test, Y_pred,
                        save_path=output_dir / "residual_plot.png")
    
    print("\n5. Learning Curve")
    create_learning_curve_plot(save_path=output_dir / "learning_curve.png")
    
    # Calculate summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    from sklearn.metrics import mean_absolute_error, r2_score
    
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test.flatten(), Y_pred.flatten())
    
    y_true_log = np.log10(Y_test + 1e-12)
    y_pred_log = np.log10(Y_pred + 1e-12)
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    
    print(f"\nLinear space:")
    print(f"  MAE: {mae:.6e}")
    print(f"  R²: {r2:.6f}")
    print(f"\nLog space:")
    print(f"  MAE (log10): {mae_log:.6f}")
    
    print("\n" + "="*60)
    print(f"All plots saved to: {output_dir}")
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    main()

