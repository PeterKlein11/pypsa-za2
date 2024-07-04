configfile: "config.yaml"

from os.path import normpath, exists, isdir

wildcard_constraints:
    model_type="[a-zA-Z0-9_]+",
    scenarios_to_run="[-a-zA-Z0-9_]+",

import pandas as pd
import os

scenarios = pd.read_excel(
    os.path.join("scenarios", config["scenarios"]["folder"], config["scenarios"]["setup"]),
    sheet_name="scenario_definition", 
    index_col=0
)
scenarios_to_run = scenarios[scenarios["run_scenario"]==1]


scenarios_dispatch_years={}
for scenario in scenarios_to_run.index:
    years = scenarios_to_run.loc[scenario,"dispatch_years"]
    scenarios_dispatch_years[scenario] = re.split(",\s*",years) if isinstance(years, str) else [str(int(years))]

############################################################################################################
# Rules to run through all scenarios specified in the scenarios_to_run.xlsx file
############################################################################################################

rule solve_all:
    input:
        "results/solve_all_capacity_scenarios"
        "results/solve_all_dispatch_scenarios"

rule solve_all_capacity_scenarios:
    input:
        expand(
            "results/" + config["scenarios"]["folder"] + "/{scenario}/capacity-solved.nc",
            scenario=scenarios_to_run.index,
        )
    output:
        touch("results/solve_all_capacity_scenarios")


# Function to generate the input files based on the scenario and its respective years
def generate_dispatch_inputs():
    inputs = []
    for scenario, years in scenarios_dispatch_years.items():
        for year in years:
            inputs.append(f"results/{config['scenarios']['folder']}/{scenario}/dispatch-{year}-solved.nc")
    return inputs

rule solve_all_dispatch_scenarios:
    input:
        generate_dispatch_inputs()
    output:
        touch("results/solve_all_dispatch_scenarios")

############################################################################################################
# Rules to produce network topology
############################################################################################################

rule build_topology:
    input:
        supply_regions=config["gis"]["path"] + "/supply_regions/rsa_supply_regions.gpkg",
        existing_lines=config["gis"]["path"] + "/transmission_grid/eskom_gcca_2022/Existing_Lines.shp",
        planned_lines=config["gis"]["path"] + "/transmission_grid/tdp_digitised/TDP_2023_32.shp",
        gdp_pop_data=config["gis"]["path"] + "/CSIR/Mesozones/Mesozones.shp",        
    output:
        buses="resources/"+config["scenarios"]["folder"]+"/buses-{scenario}.geojson",
        lines="resources/"+config["scenarios"]["folder"]+"/lines-{scenario}.geojson",
    script: "scripts/build_topology.py"


############################################################################################################
# Rules to produce system dcapacity expansion planning model
############################################################################################################

rule base_network_capacity:
    input:
        buses="resources/" + config["scenarios"]["folder"] + "/buses-{scenario}.geojson",
        lines="resources/" + config["scenarios"]["folder"] + "/lines-{scenario}.geojson",
    output: 
        "networks/" + config["scenarios"]["folder"] + "/{scenario}/capacity-base.nc",
    resources: mem_mb=1000
    script: "scripts/base_network.py"



rule add_electricity_capacity:
    input:
        base_network="networks/" + config["scenarios"]["folder"] + "/{scenario}/capacity-base.nc",
        supply_regions="resources/" + config["scenarios"]["folder"] + "/buses-{scenario}.geojson",
        load="data/bundle/SystemEnergy2009_22.csv",
        eskom_profiles="data/eskom_pu_profiles.csv",
        renewable_profiles="resources/renewable_profiles_updated.nc",
    output: "networks/" + config["scenarios"]["folder"] + "/{scenario}/capacity-elec.nc",
    script: "scripts/add_electricity.py"



rule prepare_and_solve_network:
    input:
        network="networks/"+ config["scenarios"]["folder"] + "/{scenario}/capacity-elec.nc",
    output: 
        network="results/" + config["scenarios"]["folder"] + "/{scenario}/capacity-solved.nc",
        network_stats="results/" + config["scenarios"]["folder"] +"/{scenario}/network_stats_{scenario}.csv",
        emissions="results/" + config["scenarios"]["folder"] +"/{scenario}/emissions_{scenario}.csv",
    resources:
        solver_slots=1
    script:
        "scripts/prepare_and_solve_network.py"


############################################################################################################
# Rules to produce system dispatch model
############################################################################################################
rule base_network_dispatch:
    input:
        buses="resources/" + config["scenarios"]["folder"] + "/buses-{scenario}.geojson",
        lines="resources/" + config["scenarios"]["folder"] + "/lines-{scenario}.geojson",
    output: 
        "networks/" + config["scenarios"]["folder"] + "/{scenario}/dispatch-{year}-base.nc",
    resources: mem_mb=1000
    script: "scripts/base_network.py"

rule add_electricity_dispatch:
    input:
        base_network="networks/" + config["scenarios"]["folder"] + "/{scenario}/dispatch-{year}-base.nc",
        supply_regions="resources/" + config["scenarios"]["folder"] + "/buses-{scenario}.geojson",
        load="data/bundle/SystemEnergy2009_22.csv",
        eskom_profiles="data/eskom_pu_profiles.csv",
        renewable_profiles="resources/renewable_profiles_updated.nc",
    output: "networks/" + config["scenarios"]["folder"] + "/{scenario}/dispatch-{year}-elec.nc",
    script: "scripts/add_electricity.py"

rule solve_network_dispatch:
    input:
        network="networks/" + config["scenarios"]["folder"] + "/{scenario}/dispatch-{year}-elec.nc",
        optimised_network_stats="results/" + config["scenarios"]["folder"] +"/{scenario}/network_stats_{scenario}.csv",
    output: "results/" + config["scenarios"]["folder"]+"/{scenario}/dispatch-{year}-solved.nc",
    resources:
        solver_slots=1
    script:
        "scripts/solve_network_dispatch.py"
