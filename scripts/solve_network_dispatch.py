# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch in hourly resolution using the capacities of
previous capacity expansion in rule :mod:`solve_network`.
"""


import logging
import pandas as pd
import numpy as np
import pypsa
import os
from xarray import DataArray
#from _helpers import configure_logging, update_config_with_sector_opts
#from solve_network import prepare_network, solve_network
from pypsa.descriptors import get_switchable_as_dense as get_as_dense, get_activity_mask

logger = logging.getLogger(__name__)


from add_electricity import load_scenario_definition
from custom_constraints import set_operational_limits
from prepare_and_solve_network import ccgt_steam_constraints, set_max_status



def add_missing_year_and_interp(n_stats, category):
    n_stats = n_stats[category]
    return n_stats.reindex(range(n_stats.columns[0], n_stats.columns[-1]+1),axis=1).interpolate(axis=1)

def set_optimal_capacities(n):
    optimised_network_stats = pd.read_csv(snakemake.input.optimised_network_stats, index_col=[0,1],header=[0,1])
    optimised_network_stats.columns = pd.MultiIndex.from_tuples([(x[0], int(x[1])) for x in optimised_network_stats.columns])

    p_nom_opt = (
        add_missing_year_and_interp(optimised_network_stats, "Optimal Capacity")
        - add_missing_year_and_interp(optimised_network_stats, "Installed Capacity")
    )
    for c in p_nom_opt.index.levels[0].drop("Load"):
        for bus in n.buses.index:
            for carrier in p_nom_opt.loc[c].index:
                plant = f"{bus}-{carrier}-{snakemake.wildcards.year}"
                if plant in n.df(c).index:
                    n.df(c).loc[plant, "p_nom"] = p_nom_opt.loc[(c,carrier), int(snakemake.wildcards.year)]


def aggregate_coal_unit(n):
    # In the capacity expansion model the coal generators are split into 2/3 sets of units represented by * or **
    # This function aggregates the capacity of these units back into a single unit before the linear unit commitment

    gen_param = n.generators.iloc[0]
    sum_param = ["p_nom","p_nom_opt"]
    top_param = ["p_nom_extendable", "committable"]
    avg_param = [idx for idx in gen_param.index if (not isinstance(gen_param[idx], str) and idx not in sum_param+top_param)]

    coal_units = n.generators[n.generators.index.str.contains("\*")].index
    n.generators.loc["RSA-load_shedding", "plant_name"] = "RSA-load_shedding"
    n.generators.index = n.generators.plant_name.values
    index_series = pd.Series(n.generators.index)
    duplicates = index_series[index_series.duplicated(keep=False)]
    for g in duplicates.unique().tolist():
        n.generators.loc[g, sum_param] = n.generators.loc[g, sum_param].sum().values
        n.generators.loc[g, avg_param] = n.generators.loc[g, avg_param].mean().values
    n.generators.drop_duplicates(inplace=True)
    n.generators.index.name = "Generator"

    n.generators_t.p_max_pu.rename({cu: cu.replace('*', '').strip() for cu in coal_units},axis=1, inplace=True)
    n.generators_t.p_min_pu.rename({cu: cu.replace('*', '').strip() for cu in coal_units},axis=1, inplace=True)

    for lim in ["p_max_pu", "p_min_pu"]:
        pu = n.generators_t[lim].copy()
        n.generators_t[lim] = pu.loc[:, ~pu.columns.duplicated()]


def set_reserves(n, sns, scenario_setup, status_max):
   
    reserves = pd.read_excel(
        os.path.join(scenario_setup["sub_path"],"operational_constraints.xlsx"), 
        sheet_name = "operational_reserves",
        index_col = [0,1],
    ).loc[scenario_setup["operational_reserves"], int(snakemake.wildcards.year)]

    #### Dispatchable generators
    com_i = n.generators.query("committable == True").index
    p_nom_com = n.generators.loc[com_i, "p_nom"]
    p_com = n.model.variables["Generator-p"].sel(Generator=com_i, snapshot=sns)
    com_status = n.model.variables["Generator-status"].sel({"Generator-com":com_i, "snapshot":sns}).rename({"Generator-com":"Generator"})
    n.model.add_variables(lower=0, coords=n.model.variables["Generator-p"].sel(Generator=com_i, snapshot=sns).coords, name="Generator-spin_res")
    n.model.add_variables(lower=0, coords=n.model.variables["Generator-p"].sel(Generator=com_i, snapshot=sns).coords, name="Generator-total_res")
    
    # Add dispatchable generator spinning reserves
    gen_spin_res_lhs = n.model.variables["Generator-spin_res"].sel(snapshot=sns)
    gen_spin_res_rhs =  com_status * p_nom_com - p_com
    n.model.add_constraints(gen_spin_res_lhs <= gen_spin_res_rhs, name="Generator-spin_res")
    
    # Add dispatchable generator total reserves
    gen_total_res_lhs = n.model.variables["Generator-total_res"].sel(snapshot=sns) + p_com
    gen_total_res_rhs =  DataArray(status_max*p_nom_com)
    gen_total_res_rhs = gen_total_res_rhs.rename({"Generator-com":"Generator"}) if "Generator" in gen_total_res_rhs else gen_total_res_rhs
    n.model.add_constraints(gen_total_res_lhs <= gen_total_res_rhs, name="Generator-total_res")

    #### Energy storage
    n.model.add_variables(lower=0, coords=n.model.variables["StorageUnit-p_store"].sel(snapshot=sns).coords, name="StorageUnit-res")
    p_store = n.model.variables["StorageUnit-p_store"].sel(snapshot=sns)
    p_dispatch = n.model.variables["StorageUnit-p_dispatch"].sel(snapshot=sns)
    st_soc = n.model.variables["StorageUnit-state_of_charge"].sel(snapshot=sns)

    st_res_lhs1 = n.model.variables["StorageUnit-res"].sel(snapshot=sns) + p_dispatch - p_store
    st_res_rhs1 = DataArray(n.get_switchable_as_dense("StorageUnit", "p_max_pu")*n.storage_units.p_nom)
    n.model.add_constraints(st_res_lhs1 <= st_res_rhs1, name="Storage_unit-res1")

    st_res_lhs2 = n.model.variables["StorageUnit-res"].sel(snapshot=sns)
    st_res_rhs2 = n.model.variables["StorageUnit-state_of_charge"].sel(snapshot=sns) + p_store
    n.model.add_constraints(st_res_lhs2 <= st_res_rhs2, name="Storage_unit-res2")

    #### Total reserves
    tot_spin_res = n.model.variables["Generator-spin_res"].sum("Generator") + n.model.variables["StorageUnit-res"].sum("StorageUnit")
    tot_res = n.model.variables["Generator-total_res"].sum("Generator") + n.model.variables["StorageUnit-res"].sum("StorageUnit")

    n.model.add_constraints(tot_spin_res >= reserves["spinning"], name="Reserves-spin_res")
    n.model.add_constraints(tot_res >= reserves["total"], name="Reserves-total_res")

    return

def custom_constraints_wrapper(scenario_setup):
    def custom_constraints(n, sns):

        #status_max = set_max_status(n, sns)
        ccgt_steam_constraints(n, sns, snakemake)
        set_operational_limits(n, sns, scenario_setup, snakemake, exclude_flag=["year"], model_type = "dispatch")
        set_reserves(n, sns, scenario_setup, n.get_switchable_as_dense("Generator", "p_max_pu"))
    return custom_constraints

def optimize_with_monthly_rolling_horizon(n, snapshots=None, overlap=0, **kwargs):
    
    snapshots = n.snapshots if snapshots is None else snapshots
    extra_func = custom_constraints_wrapper(kwargs["scenario_setup"])
    sns_cnt = 0
    for m in range(1,13):
        mnth_sns = snapshots[snapshots.month == m]
        sns_overlap = overlap if m<12 else 0
        sns = snapshots[sns_cnt: (sns_cnt + len(mnth_sns) + sns_overlap)]

        print(f"Optimizing month {m} from snapshot {sns[0]} to {sns[-1]}")

        if m>1:
            if not n.stores.empty:
                n.stores.e_initial = n.stores_t.e.loc[snapshots[sns_cnt - 1]]
            if not n.storage_units.empty:
                n.storage_units.state_of_charge_initial = (
                    n.storage_units_t.state_of_charge.loc[snapshots[sns_cnt - 1]]
                )
        n.optimize(snapshots = sns, linearized_unit_commitment=True, extra_functionality=extra_func, solver_name = kwargs["solver_name"], solver_options = kwargs["solver_options"])
 
        sns_cnt += len(mnth_sns)
    return n


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network_dispatch', 
            **{
                'scenario':'CNS_G_RNZ_CB_UC',
                'year':'2030',
                'model_type':'dispatch'
            }
        )
    n = pypsa.Network(snakemake.input.network)

    solver_name = snakemake.config["dispatch_solver"]["solver"].pop("name")
    solver_options = snakemake.config["dispatch_solver"]["solver"].copy()

    scenario_setup = load_scenario_definition(snakemake)
    set_optimal_capacities(n)
    aggregate_coal_unit(n)

    extra_func = custom_constraints_wrapper(scenario_setup)
    #n.optimize(snapshots=n.snapshots[0:500], linearized_unit_commitment=True, extra_functionality=extra_func, solver_name="xpress", solver_options={"lpflags":0,"crossover":0})
    #n.optimize.optimize_with_rolling_horizon(horizon=48, overlap=24,linearized_unit_commitment=True, extra_functionality=extra_func, solver_name="xpress", solver_options={"threads":20})
    optimize_with_monthly_rolling_horizon(n, overlap=48, solver_name=solver_name, solver_options=solver_options, scenario_setup = scenario_setup)
    n.export_to_netcdf(snakemake.output[0])