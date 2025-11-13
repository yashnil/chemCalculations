# FastChem Data Generation Pipeline

This folder recreates the helper scripts we used earlier to expand the FastChem dataset.
The workflow mirrors the one described during the first resolution study:

1. `prepare_fastchem_jobs.py` – sample new `(T, P, abundances)` states and split them into shard folders.
2. `run_fastchem_batch.py` – run FastChem for one shard (uses the Python bindings).
3. `run_fastchem_all.py` – convenience wrapper that calls `run_fastchem_batch.py` for every shard under a root.
4. `merge_fastchem_outputs.py` – stitch the shard results back into a single CSV that matches the reference schema.

The scripts assume you have:

- The FastChem Python package installed (bindings built locally).
- `FASTCHEM_LOGK` and `FASTCHEM_COND` pointing to the species data tables, or you pass the paths explicitly.
- A “reference” CSV (e.g. `NEW_VERS/all_gas_v10_no_stripe_clean.csv`) whose columns define the desired ordering.

Typical usage for a 40k / 60k sweep (see main README for details):

```bash
python prepare_fastchem_jobs.py --total-samples 40000 --shard-size 2000 \
  --output-root fastchem_jobs_x32 --reference-csv ../NEW_VERS/all_gas_v10_no_stripe_clean.csv

python run_fastchem_all.py --jobs-root fastchem_jobs_x32 \
  --logk $FASTCHEM_LOGK --logk-cond $FASTCHEM_COND --chunksize 128

python merge_fastchem_outputs.py --jobs-root fastchem_jobs_x32 \
  --reference-csv ../NEW_VERS/all_gas_v10_no_stripe_clean.csv \
  --output-csv ../NEW_VERS/all_gas_v10_x32.csv
```

All scripts accept `--help` for the full set of options. Adjust paths as needed. After merging, run the low-temperature filter (`v10/fix_stripe.py`) and the usual cleaning pass to obtain the final `*_no_stripe_clean.csv` files.


