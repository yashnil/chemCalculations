
#!/usr/bin/env python3
# step 0 -> generate_ds.py

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyfastchem

N_GROUPS        = 20          # 20 k samples total
N_PER_GROUP     = 1000        
STRATIFY_GRID   = True        # False ⇒ keep pure-random
N_T_BINS        = 20          # only used if STRATIFY_GRID = True
N_P_BINS        = 20

# 1) Generate a single random composition (H, O, C, N, S) in log scale

def generate_single_composition():
    """
    Returns a random composition for H, O, C, N, S.
    Each element's log10 value is sampled uniformly between -9 and 0 (i.e. values from 1e-9 to 1, with 1 exclusive).
    The resulting linear values are then normalized so that their sum equals 1.
    """
    rng = np.random.default_rng()  # each call gets independent randomness
    lower_log, upper_log = -9, 0
    log_vals = rng.uniform(lower_log, upper_log, 5)
    vals = 10 ** log_vals   # convert from log space to linear values
    vals /= vals.sum()      # normalize so that the sum equals 1
    return {
        'H': vals[0],
        'O': vals[1],
        'C': vals[2],
        'N': vals[3],
        'S': vals[4],
    }

# 2) Write an abundance file for each composition

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
            return -999.0  # zero abundance
        return 12.0 + np.log10(elem_ratio / h_ratio)

    lines = [
        "# Custom abundance file for pyFastChem",
        "e-  0.0",
        f"H   {ratio_to_logX(comp['H']):.4f}",
        f"O   {ratio_to_logX(comp['O']):.4f}",
        f"C   {ratio_to_logX(comp['C']):.4f}",
        f"N   {ratio_to_logX(comp['N']):.4f}",
        f"S   {ratio_to_logX(comp['S']):.4f}",
    ]
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")

# 3) Run pyFastChem for a single independent evaluation point

def run_pyfastchem_for_composition(comp, T_array, P_array, out_dir="results", condensates=True):
    """
    1) Writes an abundance file for the given `comp`.
    2) Constructs a pyFastChem object using that file + logK data.
    3) Calls calcDensities for the given T,P array (which is of length 1 in this case).
    4) Returns two DataFrames: (df_gas, df_cond)
       - df_gas has columns for all *gas-phase* species, plus temperature and pressure.
       - df_cond has columns for *condensed* species, plus temperature and pressure.
    """
    os.makedirs(out_dir, exist_ok=True)
    abundance_file = os.path.join(out_dir, "abund.dat")
    write_custom_abundance_file(comp, abundance_file)

    logK_path = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK.dat"
    cond_path = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK_condensates.dat"

    # Convert paths to absolute
    abund_path = os.path.abspath(abundance_file)
    logK_path  = os.path.abspath(logK_path)
    cond_path  = os.path.abspath(cond_path)

    # Create FastChem object (with or without condensates)
    if condensates:
        fastchem_obj = pyfastchem.FastChem(
            abund_path,
            logK_path,
            cond_path,
            1
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

    # Prepare output
    output_data = pyfastchem.FastChemOutput()

    # Call the solver
    start_t = time.time()
    fastchem_flag = fastchem_obj.calcDensities(input_data, output_data)
    end_t = time.time()

    n_points = len(T_array)
    print(f"    -> calcDensities() finished in {end_t - start_t:.2f} s for {n_points} point(s).")

    flag_arr = np.array(output_data.fastchem_flag)
    n_fail = np.count_nonzero(flag_arr != 0)
    n_ok   = n_points - n_fail
    print(f"    -> {n_ok}/{n_points} points converged; {n_fail} failed with non-zero flag.")

    if fastchem_flag != 0:
        msg = pyfastchem.FASTCHEM_MSG[fastchem_flag]
        print(f"    [Warning] fastchem_flag={fastchem_flag} => {msg} for composition {comp}")
    
    nd_gas = np.array(output_data.number_densities, dtype=np.float64)
    nd_gas = np.nan_to_num(nd_gas, nan=0.0, posinf=0.0, neginf=0.0)

    n_gas = fastchem_obj.getGasSpeciesNumber()
    gas_species_names = [fastchem_obj.getGasSpeciesSymbol(i) for i in range(n_gas)]
    df_gas = pd.DataFrame(nd_gas, columns=gas_species_names)
    df_gas['temperature'] = T_array
    df_gas['pressure'] = P_array

    # Build the condensed-phase DataFrame
    nd_cond = np.array(output_data.number_densities_cond)  # shape: (n_points, nCondSpecies)
    n_cond = fastchem_obj.getCondSpeciesNumber()
    cond_species_names = [fastchem_obj.getCondSpeciesName(i) for i in range(n_cond)]
    df_cond = pd.DataFrame(nd_cond, columns=cond_species_names)
    df_cond['temperature'] = T_array
    df_cond['pressure'] = P_array

    # Add composition info to each DataFrame
    for elem, val in comp.items():
        df_gas[f"comp_{elem}"] = val
        df_cond[f"comp_{elem}"] = val

    return df_gas, df_cond, flag_arr

# Normalize the gas-phase abundances for each evaluation point

def normalize_gas_abundances(df_gas):
    """
    For the gas-phase DataFrame, normalize the species abundances so that for each row
    (each T,P point) the sum of all species (excluding non-species columns) is 1.
    """
    df_norm = df_gas.copy()
    non_species = {'temperature', 'pressure'}
    non_species.update(c for c in df_norm.columns if c.startswith('comp_'))
    species_cols = [c for c in df_norm.columns if c not in non_species]
    row_sum = df_norm[species_cols].sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    df_norm[species_cols] = df_norm[species_cols].div(row_sum, axis=0)
    return df_norm

# 4) Find top-10 species for gas-phase from normalized data

def find_top_10_species(df_gas):
    """
    From the normalized gas-phase DataFrame, find the top-10 species by maximum abundance.
    Ignores 'temperature', 'pressure', 'comp_...' columns.
    Returns a list of top-10 species names.
    """
    non_sp = {'temperature', 'pressure'}
    non_sp.update(c for c in df_gas.columns if c.startswith('comp_'))
    species_cols = [c for c in df_gas.columns if c not in non_sp]
    
    maxvals = {sp: df_gas[sp].max().item() for sp in species_cols}
    sorted_sp = sorted(maxvals.items(), key=lambda x: x[1], reverse=True)
    top10 = [s for s, val in sorted_sp[:10]]
    return top10

# 5) Plot each top species vs. T,P

def plot_species_vs_TP(df_gas, top_species, group_index):
    """
    Make tri-contour plots of log10(abundance) vs. (T, log10(P)) for each species.
    Before triangulation, filter out data points with non-finite values.
    """
    from matplotlib.tri import Triangulation

    for sp in top_species:
        # Replace zeros with a small positive number to avoid -inf from log10(0)
        abun = df_gas[sp].replace(0.0, 1e-99)
        log_abun = np.log10(abun)
        T = df_gas['temperature'].values
        logP = np.log10(df_gas['pressure'].values)
        
        # Create a mask for finite values
        valid_mask = np.isfinite(T) & np.isfinite(logP) & np.isfinite(log_abun)
        if np.sum(valid_mask) < 3:
            print(f"Not enough valid points to plot species {sp} for group {group_index}. Skipping.")
            continue
        
        T_valid = T[valid_mask]
        logP_valid = logP[valid_mask]
        z_valid = log_abun[valid_mask]

        tri = Triangulation(T_valid, logP_valid)

        plt.figure()
        contour = plt.tricontourf(tri, z_valid, levels=20)
        cb = plt.colorbar(contour)
        cb.set_label("log10(normalized number_density)")
        plt.xlabel("Temperature [K]")
        plt.ylabel("log10(P / bar)")
        plt.title(f"Group {group_index} - {sp}")
        save_path = f"/Users/yashnilmohanty/Desktop/FastChem-Materials/graphs/plot_group{group_index}_{sp}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

# 4000 total runs

def main() -> None:
    print("\n===== Starting independent pyFastChem runs =====")

    all_gas, all_cond = [], []
    rng              = np.random.default_rng(seed=12345)
    overall_start    = time.time()

    for grp in range(N_GROUPS):
        print(f"\n=== Group {grp}  ({N_PER_GROUP} evaluations) ===")
        grp_gas, grp_cond = [], []
        t0 = time.time()

        # ------------------------------------------------------------------
        # (A)  choose between “pure-random” and “stratified grid” sampling
        # ------------------------------------------------------------------
        if STRATIFY_GRID:
            # how many random compositions per (T-bin, P-bin) cell ?
            reps = max(1, N_PER_GROUP // (N_T_BINS * N_P_BINS))
            point_idx = 0
            for i_t in range(N_T_BINS):
                for i_p in range(N_P_BINS):
                    # jitter within each cell so we do not land on the centre
                    T_val   = 100.0 + (i_t + rng.random()) * (2900.0 / N_T_BINS)
                    logP    = -10.0 + (i_p + rng.random()) * (15.0  / N_P_BINS)
                    P_val   = 10.0**logP

                    for _ in range(reps):
                        comp   = generate_single_composition()
                        T_arr  = np.array([T_val], dtype=np.float64)
                        P_arr  = np.array([P_val], dtype=np.float64)
                        outdir = os.path.join("results", f"group{grp}",
                                              f"T{i_t}_P{i_p}_{point_idx}")
                        df_g, df_c, _ = run_pyfastchem_for_composition(
                            comp, T_arr, P_arr, out_dir=outdir,
                            condensates=False)

                        df_g["group_index"]  = grp
                        df_g["point_index"]  = point_idx
                        df_c["group_index"]  = grp
                        df_c["point_index"]  = point_idx
                        grp_gas.append(df_g);  grp_cond.append(df_c)
                        point_idx += 1
        else:
            # ------------------------------------------------------------------
            # (B)  old behaviour – completely random T, P, composition
            # ------------------------------------------------------------------
            for pt in range(N_PER_GROUP):
                T_val  = rng.uniform(100.0, 3000.0)
                logP   = rng.uniform(-10.0, 5.0)
                P_val  = 10.0**logP
                comp   = generate_single_composition()
                T_arr  = np.array([T_val])
                P_arr  = np.array([P_val])
                outdir = os.path.join("results", f"group{grp}", f"pt{pt}")

                df_g, df_c, _ = run_pyfastchem_for_composition(
                    comp, T_arr, P_arr, out_dir=outdir, condensates=False)

                df_g["group_index"] = grp;  df_g["point_index"] = pt
                df_c["group_index"] = grp;  df_c["point_index"] = pt
                grp_gas.append(df_g);  grp_cond.append(df_c)

        # ------------------------------------------------------------------
        # normalise, plot, and stash this group
        # ------------------------------------------------------------------
        df_gas  = pd.concat(grp_gas,  ignore_index=True)
        df_cond = pd.concat(grp_cond, ignore_index=True)

        df_gas_norm = normalize_gas_abundances(df_gas)

        top10 = find_top_10_species(df_gas_norm)
        print(f"    → top-10 species: {top10}")
        plot_species_vs_TP(df_gas_norm, top10, group_index=grp)

        all_gas.append(df_gas_norm);  all_cond.append(df_cond)
        print(f"=== Group {grp} done in {time.time() - t0:.1f} s")

    # ----------------------------------------------------------------------
    # concatenate & persist
    # ----------------------------------------------------------------------
    df_all_gas  = pd.concat(all_gas,  ignore_index=True)
    df_all_cond = pd.concat(all_cond, ignore_index=True)

    tables_dir = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables"
    os.makedirs(tables_dir, exist_ok=True)
    df_all_gas.to_csv(os.path.join(tables_dir, "all_gas.csv"),  index=False)
    df_all_cond.to_csv(os.path.join(tables_dir, "all_cond.csv"), index=False)

    print("\n=== Finished!  total run time:",
          f"{time.time() - overall_start:.1f} s ===")
    print("Gas-phase dataframe:", df_all_gas.shape,
          " | Condensed-phase:", df_all_cond.shape)

if __name__ == "__main__":
    main()
