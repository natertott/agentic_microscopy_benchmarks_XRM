# agentic_microscopy_benchmarks_XRM
This repository contains benchmark test results for an agentic self-driving X-ray microscopy system. It also contains supplemental information for the paper associated with the benchmarks.

FILES IN THIS REPOSITORY:
agent_eval_core_master_updated.py - This is the core Python file that is used to load, process, model, and plot the benchmark test results.
agent_eval_master_analysis_updated.ipynb - This Jupyter notebook comprises the main plotting and analysis functions from the study. The majority of the figures that appear in the paper are created through this notebook. This includes the training of the surrogate models and in-depth RAG analysis. Crucially, this notebook compares ALL CONFIGS across ALL BENCHMARK TESTS. 
framework_comparison_updated.ipynb - This notebook is used to compare the one-, two-, and three-agent frameworks across their consistent benchmark tests.
optimal_config_comparisons_updated.ipynb - This notebook is used to compare the optimal agent configurations, as identified by the analysis in agent_eval_master_analysis_updated. Please note that this notebook requires importing specific benchmark test results, which can be found inside the sweep_results.zip compressed folder.

sweep_results.zip - This compressed file contains all configurations and all benchmark test results from the study. The folder is split up into two sub-folders: surrogate_modeling_benchmark_suites and full_suite_optimal_configs. The subfolder surrogate_modeling_benchmark_suites contains all test results across all configs tested in the study. The subfolder full_suite_optimal_configs only contains test results for the three optimal and one baseline agents tested. 
