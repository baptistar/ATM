clear all; close all; clc
addpath(genpath('../../../src'))

cd MoG_Exp1
mog_linearizedbasis
mog_comparebasis
cd ..

cd MoG_Exp2
compare_polynomials_vs_wavelets
cd ..

cd TwoD_Datasets
run_adaptive
plot_PDF
process_basis
cd ..

cd MoGAdaptive
run_example
cd ..

cd StochasticVolatility
run_datasets
post_process
plot_results
cd ..

cd TabularDatasets
run_tabular_ATM
run_tabular_nonadaptive
compute_loglik
cd ..

cd TabularDatasetsConditionals
run_tabular_cv
post_process_loglik
run_timing
post_process_timing
cd ..

