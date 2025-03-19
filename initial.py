#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyfastchem  # The pyFastChem interface

##############################################################################
# 1) Generate random compositions (H, O, C, N, S)
##############################################################################

def generate_random_compositions(n_compositions=10, seed=42):
    """
    Returns a list of random compositions for H, O, C, N, S.
    Each composition is a dict like {'H': <float>, 'O': <float>, ...}.
    """
    rng = np.random.default_rng(seed)
    compositions = []
    for _ in range(n_compositions):
        vals = rng.random(5)    # random positive numbers
        vals /= vals.sum()      # normalize so sum=1
        comp = {
            'H': vals[0],
            'O': vals[1],
            'C': vals[2],
            'N': vals[3],
            'S': vals[4],
        }
        compositions.append(comp)
    return compositions

##############################################################################
# 2) Sample Temperature & Pressure
##############################################################################

def sample_temperature_pressure(
    n_temp=20, n_press=20,
    T_min=100.0, T_max=3000.0,
    logP_min=-10.0, logP_max=5.0,
    seed=123
):
    """
    Randomly sample n_temp points in [T_min, T_max].
    Randomly sample n_press points in [10^(logP_min), 10^(logP_max)].
    Return T_flat, P_flat (each length n_temp * n_press).
    """
    rng = np.random.default_rng(seed)

    T_vals = rng.uniform(T_min, T_max, n_temp)
    logP_vals = rng.uniform(logP_min, logP_max, n_press)
    P_vals = 10.0**(logP_vals)

    # Create a grid (Cartesian product)
    T_grid, P_grid = np.meshgrid(T_vals, P_vals, indexing='xy')
    return T_grid.ravel(), P_grid.ravel()

##############################################################################
# 2.5) Write a custom abundance file for each composition
##############################################################################

def write_custom_abundance_file(comp, filename="custom_abund.dat"):
    """
    Writes a minimal abundance file for FastChem with e-, H, O, C, N, S.
    FastChem uses the 'X' scale: H=12 => X_elem = 12 + log10(elem_ratio/H_ratio).
    """
    h_ratio = comp['H']
    if h_ratio <= 0:
        raise ValueError("H ratio must be > 0")

    def ratio_to_logX(elem_ratio):
        if elem_ratio <= 0:
            return -999.0  # effectively zero abundance
        return 12.0 + np.log10(elem_ratio / h_ratio)

    # Build lines
    e_line = "e-  0.0"  # dummy
    h_line = f"H   {ratio_to_logX(comp['H']):.4f}"
    o_line = f"O   {ratio_to_logX(comp['O']):.4f}"
    c_line = f"C   {ratio_to_logX(comp['C']):.4f}"
    n_line = f"N   {ratio_to_logX(comp['N']):.4f}"
    s_line = f"S   {ratio_to_logX(comp['S']):.4f}"

    with open(filename, 'w') as f:
        f.write("# Custom abundance file for pyFastChem\n")
        f.write(e_line + "\n")
        f.write(h_line + "\n")
        f.write(o_line + "\n")
        f.write(c_line + "\n")
        f.write(n_line + "\n")
        f.write(s_line + "\n")

##############################################################################
# 3) Run pyFastChem for one composition
##############################################################################

def run_pyfastchem_for_composition(
    comp,
    T_array,
    P_array,
    out_dir="results",
    condensates=True
):
    """
    1) Writes an abundance file for `comp`.
    2) Constructs a pyFastChem object using that file + logK data.
    3) Calls calcDensities for the entire T,P array (400 points).
    4) Returns two DataFrames: (df_gas, df_cond)
       - df_gas has columns for all *gas-phase* species, plus T,P
       - df_cond has columns for *condensed* species, plus T,P
    """
    os.makedirs(out_dir, exist_ok=True)
    abundance_file = os.path.join(out_dir, "abund.dat")
    write_custom_abundance_file(comp, abundance_file)

    # Adjust these paths to your actual file locations
    logK_path = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK.dat"
    cond_path = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK_condensates.dat"

    # Convert to absolute paths (optional, but safer)
    abund_path = os.path.abspath(abundance_file)
    logK_path  = os.path.abspath(logK_path)
    cond_path  = os.path.abspath(cond_path)

    # Create FastChem object (with or without condensates)
    # Lower verbose level (e.g., 0) can reduce terminal spam from FastChem
    if condensates:
        fastchem_obj = pyfastchem.FastChem(
            abund_path,
            logK_path,
            cond_path,
            1  # <-- set to 0 for minimal prints
        )
    else:
        fastchem_obj = pyfastchem.FastChem(
            abund_path,
            logK_path,
            'none',
            1
        )

    # Build input
    input_data = pyfastchem.FastChemInput()
    input_data.temperature = T_array
    input_data.pressure    = P_array

    if condensates:
        input_data.equilibrium_condensation = True

    # Output
    output_data = pyfastchem.FastChemOutput()

    # Call the solver
    start_t = time.time()
    fastchem_flag = fastchem_obj.calcDensities(input_data, output_data)
    end_t = time.time()

    # Print how long this took
    n_points = len(T_array)
    print(f"    -> calcDensities() finished in {end_t - start_t:.2f} s for {n_points} T,P points.")

    # Summarize how many points converged vs. failed
    flag_arr = np.array(output_data.fastchem_flag)
    n_fail = np.count_nonzero(flag_arr != 0)
    n_ok   = n_points - n_fail
    print(f"    -> {n_ok}/{n_points} points converged; {n_fail} failed with non-zero flag.")

    if fastchem_flag != 0:
        msg = pyfastchem.FASTCHEM_MSG[fastchem_flag]
        print(f"    [Warning] fastchem_flag={fastchem_flag} => {msg} for composition {comp}")

    # ============ Gas-Phase Data ============

    # Convert gas-phase number densities => DataFrame
    nd_gas = np.array(output_data.number_densities)  # shape=(n_points, nGasSpecies)

    n_gas = fastchem_obj.getGasSpeciesNumber()
    print(n_gas)
    # Build a list of gas species names
    gas_species_names = []
    for i in range(n_gas):
        sp_name = fastchem_obj.getGasSpeciesSymbol(i)  # e.g. "H", "H2", "CH4", ...
        gas_species_names.append(sp_name)

    df_gas = pd.DataFrame(nd_gas, columns=gas_species_names)
    df_gas['temperature'] = T_array
    df_gas['pressure']    = P_array

    # ============ Condensed-Phase Data ============

    nd_cond = np.array(output_data.number_densities_cond)  # shape=(n_points, nCondSpecies)

    n_cond = fastchem_obj.getCondSpeciesNumber()
    cond_species_names = []
    for i in range(n_cond):
        sp_name = fastchem_obj.getCondSpeciesName(i)
        cond_species_names.append(sp_name)

    df_cond = pd.DataFrame(nd_cond, columns=cond_species_names)
    df_cond['temperature'] = T_array
    df_cond['pressure']    = P_array

    # Add composition info to each DF
    for elem, val in comp.items():
        df_gas[f"comp_{elem}"] = val
        df_cond[f"comp_{elem}"] = val

    return df_gas, df_cond, flag_arr

##############################################################################
# 4) Find top-10 species for gas-phase
##############################################################################

def find_top_10_species(df_gas):
    """
    From the gas-phase DataFrame, find the top-10 species by max abundance.
    Ignores 'temperature', 'pressure', 'comp_...' columns.
    Returns a list of top-10 species names.
    """
    non_sp = {'temperature','pressure'}
    non_sp.update(c for c in df_gas.columns if c.startswith('comp_'))
    species_cols = [c for c in df_gas.columns if c not in non_sp]

    maxvals = {}
    for sp in species_cols:
        maxvals[sp] = df_gas[sp].max().item()
    # for each species, find the max abundance across all grid points
    # save the abundance to maxvals

    sorted_sp = sorted(maxvals.items(), key=lambda x: x[1], reverse=True)
    # sort maxvals and find 10 highest element abundancies
    top10 = [s for s, val in sorted_sp[:10]]
    return top10

##############################################################################
# 5) Plot each top species vs. T,P
##############################################################################

def plot_species_vs_TP(df_gas, top_species, comp_index):
    """
    Make tri-contour plots of log10(abundance) vs. (T, log10(P)) for each species.
    """
    from matplotlib.tri import Triangulation

    T = df_gas['temperature'].values
    P = df_gas['pressure'].values
    tri = Triangulation(T, np.log10(P))  # map Pressure -> log10(P)

    for sp in top_species:
        abun = df_gas[sp].replace(0.0, 1e-99)
        log_abun = np.log10(abun)

        plt.figure()
        contour = plt.tricontourf(tri, log_abun, levels=20)
        cb = plt.colorbar(contour)
        cb.set_label("log10(number_density)")
        plt.xlabel("Temperature [K]")
        plt.ylabel("log10(P / bar)")
        plt.title(f"Comp {comp_index} - {sp}")
        plt.savefig(f"plot_comp{comp_index}_{sp}.png", dpi=150)
        plt.close()

##############################################################################
# Main Orchestration
##############################################################################

def main():
    print("\n===== Starting pyFastChem run =====")

    # (1) Generate random compositions
    compositions = generate_random_compositions(n_compositions=10, seed=42)
    composition = compositions[0]
    print("Initial Composition: " + str(composition))
    print(f"Generated {len(compositions)} random compositions (H,O,C,N,S).")

    # (2) Sample T,P: 20x20 => 400 points
    T_array, P_array = sample_temperature_pressure(
        n_temp=20, n_press=20,
        T_min=100, T_max=3000,
        logP_min=-10, logP_max=5,
        seed=123
    )
    print(f"Sampled {len(T_array)} total (T,P) points for each composition.")

    big_gas_list = []
    big_cond_list = []

    overall_start = time.time()

    # (3) Run pyFastChem
    out_dir = "results_single"
    df_gas, df_cond, flags = run_pyfastchem_for_composition(
        composition,
        T_array,
        P_array,
        out_dir=out_dir,
        condensates=False  # or False if you want pure gas-phase
    )


    # (4) Find top-10 gas species
    top10 = find_top_10_species(df_gas)
    print(f"    -> Top-10 gas-phase species for comp: {top10}")

    # (5) Make plots
    #plot_species_vs_TP(df_gas, top10, comp_index=0)

    overall_end = time.time()

    # Merge all gas & cond data
    #df_all_gas = pd.concat(big_gas_list, ignore_index=True)
    #df_all_cond = pd.concat(big_cond_list, ignore_index=True)

    print("\n=== All compositions complete! ===")
    print(f"Total run time: {overall_end - overall_start:.2f} seconds.")

    # If you want, you can save them to CSV or pickle:
    # df_all_gas.to_csv("all_gas.csv", index=False)
    # df_all_cond.to_csv("all_cond.csv", index=False)

    #print("Gas-phase DataFrame shape:", df_all_gas.shape)
    #print("Cond-phase DataFrame shape:", df_all_cond.shape)
    print("Check 'plot_compX_*.png' for the saved plots.\n")

if __name__ == "__main__":
    main()
