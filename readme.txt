# To run all scenarios marked in model_file with the flag run_Scenario as True
snakemake --cores all solve_all_scenarios --resources solver_slots=1
snakemake --cores all solve_all_dispatch_scenarios