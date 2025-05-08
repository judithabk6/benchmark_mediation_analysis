# Benchmark of estimators for causal mediation analysis

This repository contains all the scripts used to reproduce the results for the article : Judith Abécassis, Houssam Zenati, Sami Boumaïza, Julie Josse, Bertrand Thirion (2025). **Causal mediation analysis with one or multiple mediators: a comparative study.** [pdf](https://judithabk6.github.io/files/article_mediation_benchmark.pdf)

## Organization
The `src` directory contains all the scripts. They are written to run on a computing cluster with a slurm scheduler, but can easily be adapted to any other system. Some paths need changing as well.

The `results` directory contains a table with all the simulation results. It allows to reproduce the figures of the article, and new results and explorations if one wans to represent other aspects of the results.

Regarding UK Biobank results, the scripts to prepare the data are available, but not the data. High level results to create figures are also available, in the files named `results/20250315-EXPOSURE-MEDIATOR-ncpX-g_factor.csv`, with `EXPOSURE` the considered exposure among the ones explored (diabetes, hypertension etc), `MEDIATOR` the brain imaging variables considered (sMRI, dMRI or both) and `X` the number of latent factors used to account for missing values and dimension reduction, as described in the article.



## Technical aspects
This work has been achieved using Python 3.11.

The causal mediation estimators used are implemented in the package [`med_bench`](https://github.com/judithabk6/med_bench), which can be directly installed from the GitHub repository using
```
pip install git+git://github.com/judithabk6/med_bench.git
```

Installation time is a few minutes on a standard personal computer.

The figures were generated with seaborn version `0.12.2`.


## Instructions to reproduce the results
relative paths are to the root of this repository

### main simulation and estimator run, with 100 boostrap repetitions
generate simulated data
```bash
./src/get_simulated_data.py
```

Then, run all estimators in parallel through the slurm cluster. In my case there was a limit on the number of jobs, so this requires adjusting the `VALUES` parameters to go through the 21k jobs to run.
```bash
# prepare list of experiments
rm -f 20250221_simu2025.csv
j=1
for nrep in {0..199}; 
do for n in {500,1000,10000}; 
do for dim_setting in {0..8}; 
do for yset in {True,False};
do for mset in {True,False};
do echo $j,"/scratch/jabecass/results/simulations/20250215_simulations/rep${nrep}_n${n}_setting${dim_setting}_misspecM${mset}_misspecY${yset}" >> 20250221_simu2025.csv; 
j=$((j+1)); 
done; done; done; done; done


sbatch src/test_simulation_settings_slurm.sh
```

gather all results in a table
```bash
./src/gather_simulation_results.py
```

This yields the table `results/20250313_big_table_bootstrap.csv`, that is then used to produce figures for the article, through the jupyter notebook `src/generate_simulation_figures.ipynb`.

### experiment on the number of bootstrap repetitions
The previous scripts were adapted to run 1000 boostrap repetitions for 100 simulated samples. There results are gathered in table `results/20250221_big_table_bootstrap.csv`, and the figures for this analysis are in the notebook `src/2025_bootstrap_test.ipynb`.

### experiments on UK Biobank data
The data was first prepared using the script
```bash
./src/prepare_mental_health_lifestyle.py
```

Then, the dimension reduction was performed in `R` using the `missMDA` package.
```bash
./src/pca_missing_data_mental_health_lifestyle.R
```

And finally, the table preparation was finalized in Python. This script also generates the supplementary tables (table one)
```bash
./src/final_preparation_ukbb.py
```

and the mediation analysis was performed.
```bash
sbatch src/get_ukbb_estimation_slurm.sh

```

The results were then analyzed with the notebook `src/2025_ukbb.ipynb` to generate the figures from the article.


## Contact
If you need further information, don't hesitate to open an issue, or to reach out to Judith Abécassis by email (judith.abecassis[at]inria.fr)