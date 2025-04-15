#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

##############################################################################
# 1) LOAD & PREPROCESS
##############################################################################

def load_and_preprocess(csv_path):
    """
    Loads the CSV file from `csv_path`, drops NaNs, identifies species columns,
    and log10-transforms their abundances.
    Returns the processed DataFrame, plus the list of species columns.
    """
    # A) Load
    df = pd.read_csv(csv_path)
    print("Initial columns:", df.columns.tolist())

    # B) Drop rows with NaNs
    df.dropna(inplace=True)

    # C) Identify species columns vs. input columns
    excluded_cols = {
        'temperature','pressure',
        'comp_H','comp_O','comp_C','comp_N','comp_S',
        'comp_index'
    }
    # We'll store all columns not in excluded_cols as species
    species_cols = [c for c in df.columns if c not in excluded_cols]

    # D) Log-transform species abundance columns
    for sp in species_cols:
        # Clip to avoid log10(0)
        df[sp] = np.log10(df[sp].clip(lower=1e-99))

    print(f"Data loaded from {csv_path}. Shape after dropna(): {df.shape}")
    print(f"Species columns identified: {len(species_cols)} -> {species_cols[:5]} ...")

    return df, species_cols

##############################################################################
# 2) SPLIT INTO FEATURES, TARGETS, & TRAIN/TEST
##############################################################################

def prepare_data_for_model(df, species_cols):
    """
    Given the preprocessed DataFrame and list of species columns,
    splits into X(features) and y(targets).
    Then does train_test_split.
    Returns X_train, X_test, y_train, y_test.
    """
    # A) Define the input (X)
    X = df[['temperature','pressure','comp_H','comp_O','comp_C','comp_N','comp_S']]

    # B) The species columns as outputs (y)
    y = df[species_cols]

    # C) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,     # 20% test data
        shuffle=True,
        random_state=42    # reproducible
    )

    print("Train set shape:", X_train.shape, y_train.shape)
    print("Test set shape :", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

##############################################################################
# 3) TRAIN A NEURAL NETWORK
##############################################################################

def train_neural_net(X_train, y_train):
    """
    Trains an MLPRegressor on the given train set.
    Returns the trained model.
    """
    model = MLPRegressor(
        hidden_layer_sizes=(64,64),
        activation='relu',
        random_state=42,
        max_iter=300
    )
    print("\nTraining MLPRegressor...")
    model.fit(X_train, y_train)
    print("...Training complete!")
    return model

##############################################################################
# 4) EVALUATE THE MODEL
##############################################################################

def evaluate_model(model, X_test, y_test, species_cols):
    """
    Generates predictions, computes MSE & R² for each species,
    and prints overall metrics. Also returns y_pred for further analysis.
    """
    # A) Prediction
    y_pred = model.predict(X_test)

    # B) MSE & R²
    mse_vals = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2_vals = []
    for i, sp in enumerate(species_cols):
        r2_i = r2_score(y_test.iloc[:, i], y_pred[:, i])
        r2_vals.append(r2_i)
        #print(f"Species {sp}: MSE={mse_vals[i]:.4f}, R2={r2_i:.3f}")

    # Overall average
    mse_mean = np.mean(mse_vals)
    r2_mean = np.mean(r2_vals)

    print("\n===== MODEL EVALUATION =====")
    print(f"Overall MSE: {mse_mean:.4f}, Overall R2: {r2_mean:.3f}")

    return y_pred, mse_vals, r2_vals

##############################################################################
# 5) PLOTS: e.g. PREDICTED vs. TRUE for selected species
##############################################################################

def plot_pred_vs_true(y_test, y_pred, species_cols, n_species_to_plot=5):
    """
    Creates scatter plots (predicted vs. true) for a few species.
    Plots log10(abundance) predictions, color-coded or separate subplots.
    Saves each plot as a separate PNG.
    """
    # We'll just do the first n_species_to_plot species for illustration
    for i, sp in enumerate(species_cols[:n_species_to_plot]):
        plt.figure()
        # True vs. predicted
        true_vals = y_test.iloc[:, i].values
        pred_vals = y_pred[:, i]

        plt.scatter(true_vals, pred_vals, alpha=0.4, edgecolor='k')
        # Plot 1:1 reference line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel("True log10(Abundance)")
        plt.ylabel("Predicted log10(Abundance)")
        plt.title(f"Predicted vs. True for {sp}")
        plt.savefig(f"/Users/yashnilmohanty/Desktop/FastChem-Materials/newGraphs/pred_vs_true_{sp}.png", dpi=150)
        plt.close()
        print(f"Saved plot: pred_vs_true_{sp}.png")

##############################################################################
# MAIN
##############################################################################

def main():
    # 1) Load & preprocess
    df, species_cols = load_and_preprocess("/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv")

    # 2) Split data
    X_train, X_test, y_train, y_test = prepare_data_for_model(df, species_cols)

    # 3) Train MLP
    model = train_neural_net(X_train, y_train)

    # 4) Evaluate
    y_pred, mse_vals, r2_vals = evaluate_model(model, X_test, y_test, species_cols)

    # 5) Plot
    plot_pred_vs_true(y_test, y_pred, species_cols, n_species_to_plot=5)

if __name__ == "__main__":
    main()
