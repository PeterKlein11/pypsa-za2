import logging
import pandas as pd
import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense, expand_series, get_activity_mask
from pypsa.optimization.common import reindex
import os

from _helpers import get_investment_periods
# from add_electricity import load_costs, update_transmission_costs

import xarray as xr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


"""
********************************************************************************
    Operational limits
********************************************************************************
"""

def calc_max_gen_potential(n, sns, gens, incl_pu, weightings, active, cf_limit, extendable=False):
    suffix = "" if extendable == False else "-ext"
    p_max_pu = get_as_dense(n, 'Generator', "p_max_pu", sns)[gens] if incl_pu else pd.DataFrame(1, index=sns, columns=gens)
    p_max_pu.columns.name = f'Generator{suffix}'
    
    if n.multi_invest:
        cf_limit_h = pd.DataFrame(0, index=sns, columns=gens)
        for y in cf_limit_h.index.get_level_values(0).unique():
            cf_limit_h.loc[y] = cf_limit[y]
    else:
        cf_limit_h = pd.DataFrame(cf_limit[sns[0].year], index=sns, columns=gens) * weightings[gens]
    
    if not extendable:
        return cf_limit_h[gens] * active[gens] * p_max_pu * weightings[gens] * n.generators.loc[gens, "p_nom"]
    p_nom = n.model.variables["Generator-p_nom"].sel({f"Generator{suffix}": gens})
    potential = xr.DataArray(cf_limit_h[gens] * active[gens] * p_max_pu * weightings[gens])
    potential = potential.rename({"Generator":"Generator-ext"}) if "Generator" in potential.dims else potential
    
    return (potential * p_nom).sum(f'Generator{suffix}')

def group_and_sum(data, groupby_func):
    grouped_data = data.groupby(groupby_func).sum()
    return grouped_data.sum(axis=1) if len(grouped_data) > 1 else grouped_data
    
def apply_operational_constraints(n, sns, **kwargs):
    energy_unit_conversion = {"GW":1e3, "GJ": 1/3.6, "TJ": 1000/3.6, "PJ": 1e6/3.6, "GWh": 1e3, "TWh": 1e6}
    apply_to = kwargs["apply_to"]

    carrier = [c.strip() for c in kwargs["carrier"].split("+")]

    #if carrier in ["gas", "diesel"]:
    #    carrier = n.carriers[n.carriers.index.str.contains(carrier)].index
    bus = kwargs["bus"]
    period = kwargs["period"]

    type_ = "energy_power" if kwargs["type"] in ["primary_energy", "output_energy", "output_power"] else "capacity_factor"

    if (period  == "week") & (max(n.snapshot_weightings["generators"])>1):
        logger.warning(
            "Applying weekly operational limits and time segmentation should be used with caution as the snapshot weightings might not align with the weekly grouping."
        )
    incl_pu = kwargs["incl_pu"]
    limit = kwargs["limit"]

    sense = "<=" if limit == "max" else ">="

    cf_limit = 0 * kwargs["values"] if type_ == "energy_power" else kwargs["values"]
    en_pow_limit = 0 * kwargs["values"] if type_ == "capacity_factor" else kwargs["values"]

    if ((kwargs["type"] in ["primary_energy", "output_energy"]) & (kwargs["units"] != "MWh")) or ((kwargs["type"] == "output_power") & (kwargs["units"] != "MW")):
        en_pow_limit *= energy_unit_conversion[kwargs["units"]]

    years = get_investment_periods(sns, n.multi_invest)

    filtered_gens = n.generators.query("carrier in @carrier") if len(carrier)>1 else n.generators.query("carrier == @carrier")
    if bus != "global":
        filtered_gens = filtered_gens.query("bus == @bus")
    fix_i = filtered_gens.query("not p_nom_extendable").index if apply_to in ["fixed", "all"] else []
    ext_i = filtered_gens.query("p_nom_extendable").index if apply_to in ["extendable", "all"] else []
    filtered_gens = filtered_gens.loc[list(fix_i) + list(ext_i)]

    if len(filtered_gens) == 0:
        return

    efficiency = get_as_dense(n, "Generator", "efficiency", inds=filtered_gens.index) if kwargs["type"] == "primary_energy" else pd.DataFrame(1, index=n.snapshots, columns = filtered_gens.index)
    weightings = (1/efficiency).multiply(n.snapshot_weightings.generators, axis=0)

    # if only extendable generators only select snapshots where generators are active
    min_year = n.generators.loc[filtered_gens.index, "build_year"].min()
    sns_active = sns[sns.get_level_values(0) >= min_year] if n.multi_invest else sns[sns.year >= min_year]
    act_gen = (n.model.variables['Generator-p'].loc[sns_active, filtered_gens.index] * weightings.loc[sns_active]).sel(Generator=filtered_gens.index).sum('Generator')
    act_gen_pow = (n.model.variables['Generator-p'].loc[sns_active, filtered_gens.index]).sel(Generator=filtered_gens.index).sum('Generator')

    timestep = "timestep" if n.multi_invest else "snapshot"
    groupby_dict = {
        "year": f"{timestep}.year",
        "month": f"{timestep}.month",
        "week": f"{timestep}.week",
        "hour": None
    }

    active = get_activity_mask(n, "Generator", sns).astype(int)
    if type_ != "energy_power":
        max_gen_fix = calc_max_gen_potential(n, sns, fix_i, incl_pu, weightings, active, cf_limit, extendable=False) if len(fix_i)>0 else 0
        max_gen_ext = calc_max_gen_potential(n, sns, ext_i, incl_pu, weightings, active, cf_limit, extendable=True) if len(ext_i)>0 else 0

    if groupby := groupby_dict[period]:
        for y in years:
            year_sns = sns_active[sns_active.get_level_values(0)==y] if n.multi_invest else sns_active
            if len(year_sns) > 0:
                if type_ == "capacity_factor":
                    lhs = (act_gen - max_gen_ext) 
                    if (isinstance(max_gen_fix, int)) | (isinstance(max_gen_fix, float)):
                        rhs = max_gen_fix
                    else:
                        rhs = max_gen_fix.loc[y] if n.multi_invest else max_gen_fix.loc[year_sns]
                else:
                    lhs = act_gen
                    rhs = en_pow_limit[y]
                lhs = lhs.sel(snapshot=year_sns)
                lhs_p = lhs.sum() if period == "year" else lhs.groupby(groupby).sum()

                # if period == "month":
                #     if isinstance(rhs, int) or isinstance(rhs, float):
                #         rhs_p = rhs
                #     else:
                #         rhs_p = group_and_sum(rhs, lambda x: x.index.month)
                #         rhs_p.index.name = period
                # elif period == "week":
                #     if isinstance(rhs, int) or isinstance(rhs, float):
                #         rhs_p = rhs
                #     else:
                #         rhs_p = group_and_sum(rhs, lambda x: x.index.isocalendar().week)
                #         rhs_p.index.name = period
                # else:  # period == "year"
                #     rhs_p = rhs if isinstance(rhs, int) or isinstance(rhs, float) else rhs.sum().sum()
                rhs_p = rhs if isinstance(rhs, int) else rhs.sum().sum()
                n.model.add_constraints(lhs_p, sense, rhs_p, name=f'{limit}-{kwargs["carrier"]}-{period}-{kwargs["apply_to"][:3]}-{y}')

    else:

        lhs = (act_gen - max_gen_ext).sel(snapshot = sns_active) if type_ == "capacity_factor" else act_gen_pow.sel(snapshot = sns_active)
        if kwargs["type"] == "output_energy":
            logging.warning("Energy limits are not yet implemented for hourly operational limits.")
            return
    
        if type_ == "capacity_factor":
            rhs = (
                max_gen_fix
                if isinstance(max_gen_fix, int)
                else xr.DataArray(
                    max_gen_fix.loc[sns_active].sum(axis=1)
                ).rename({"dim_0": "snapshot"})
            )
        else:
            rhs = pd.Series(index = sns)
            for y in years:
                rhs.loc[y] = en_pow_limit[y]
                
        n.model.add_constraints(lhs, sense, rhs, name = f'{limit}-{kwargs["carrier"]}-hour-{kwargs["apply_to"][:3]}')

def set_operational_limits(n, sns, scenario_setup):

    op_limits = pd.read_excel(
        os.path.join(scenario_setup["sub_path"], "operational_constraints.xlsx"),
        sheet_name='operational_constraints',
        index_col=list(range(9)),
    )
    
    if scenario_setup["operational_limits"] not in op_limits.index.get_level_values(0).unique():
        return
    op_limits = op_limits.loc[scenario_setup["operational_limits"]]

    #drop rows where all NaN
    op_limits = op_limits.loc[~(op_limits.isna().all(axis=1))]
    for idx, row in op_limits.iterrows():
        apply_operational_constraints(
            n, sns, 
            bus = idx[0], carrier = idx[1], 
            type = idx[2], values = row, 
            period = idx[3], incl_pu = idx[4],
            limit = idx[5], apply_to = idx[6],
            units = idx[7],
        )


def ccgt_steam_constraints(n, sns, snakemake):
    # At each bus HRSG power is limited by what OCGT power production
    config = snakemake.config["electricity"]["conventional_generators"]
    p_nom_ratio = config["ccgt_st_to_gt_ratio"]
    ocgt_carriers = n.generators.carrier[(n.generators.carrier.str.contains("ocgt"))].unique()

    # remove ocgt_diesel_emg from ocgt_carriers
    ocgt_carriers = [c for c in ocgt_carriers if c in config["allowable_ocgt_st_carriers"]]

    #ocgt_carriers = ["ocgt_gas", "ocgt_diesel"]
    for bus in n.buses.index:
        ocgt_gens = n.generators.query("bus == bus & carrier in @ocgt_carriers").index
        ccgt_hrsg = n.generators.query("bus == bus & carrier == 'ccgt_steam'").index
        
        lhs = (n.model.variables['Generator-p'].loc[sns, ccgt_hrsg] - p_nom_ratio*n.model.variables['Generator-p'].loc[sns, ocgt_gens]).sum("Generator")
        rhs = 0
        n.model.add_constraints(lhs, "<=", rhs, name = f'ccgt_steam_limit-{bus}')

"""
********************************************************************************
    Reserve margin
********************************************************************************
"""
def check_active(n, c, y, list):
    active = n.df(c).index[n.get_active_assets(c, y)] if n.multi_invest else list
    return list.intersection(active)

def reserve_margin_constraints(n, sns, scenario_setup, snakemake):
    ###################################################################################
    # Reserve margin above maximum peak demand in each year
    # The sum of res_margin_carriers multiplied by their assumed constribution factors 
    # must be higher than the maximum peak demand in each year by the reserve_margin value

    res_margin = pd.read_excel(
        os.path.join(scenario_setup["sub_path"], "reserve_margin.xlsx"), 
        sheet_name="reserve_margin",
        index_col=[0,1]).loc[scenario_setup["reserve_margin"]].drop("unit", axis=1)

    capacity_credit = pd.read_excel(
            os.path.join(scenario_setup["sub_path"], "reserve_margin.xlsx"), 
            sheet_name="capacity_credits",
            index_col=[0])[scenario_setup["capacity_credits"]]

    res_mrgn_active = res_margin.loc["reserve_margin_active"]
    res_mrgn = res_margin.loc["reserve_margin"]

    peak = n.loads_t.p_set.loc[sns].sum(axis=1).groupby(sns.get_level_values(0)).max() if n.multi_invest else n.loads_t.p_set.loc[sns].sum(axis=1).max()
    peak = peak if n.multi_invest else pd.Series(peak, index = sns.year.unique())
    #capacity_credit = snakemake.config["electricity"]["reserves"]["capacity_credit"]

    for y in peak.index:
        if res_mrgn_active[y]:    

            fix_i = n.generators.query("not p_nom_extendable & carrier in @capacity_credit").index
            ext_i = n.generators.query("p_nom_extendable & carrier in @capacity_credit").index
    
            fix_cap = 0
            lhs = 0
            for c in ["Generator", "StorageUnit"]:
                fix_i = n.df(c).query("not p_nom_extendable & carrier in @capacity_credit.index").index
                fix_i = check_active(n, c, y, fix_i)

                fix_cap += (
                    n.df(c).loc[fix_i, "carrier"].map(capacity_credit)
                    * n.df(c).loc[fix_i, "p_nom"]
                ).sum()
            
                ext_i = n.df(c).query("p_nom_extendable & carrier in @capacity_credit.index").index
                ext_i = check_active(n, c, y, ext_i)
    
                lhs += (
                    n.model.variables[f"{c}-p_nom"].sel({f"{c}-ext":ext_i}) 
                    *xr.DataArray(n.df(c).loc[ext_i, "carrier"].map(capacity_credit)).rename({f"{c}":f"{c}-ext"})
                ).sum(f"{c}-ext")

            rhs = peak.loc[y]*(1+res_mrgn[y]) - fix_cap 

            n.model.add_constraints(lhs, ">=", rhs, name = f"reserve_margin_{y}")    

def annual_co2_constraints(n, sns, param, scenario_setup):

    if scenario_setup["co2_constraints"] in ["None", "none", ""]:
        return

    gen_emissions = param.loc["co2_emissions"].drop("unit", axis=1)
    gen_emissions = gen_emissions[gen_emissions.mean(axis=1) >0]

    gen_p = n.model.variables['Generator-p']
    gen_p = gen_p.sel(Generator = n.generators.query("carrier in @gen_emissions.index").index)

    co2_emissions = xr.DataArray(coords=gen_p.coords, dims=gen_p.dims)

    for gen in gen_p.coords["Generator"].values:
        for y in n.investment_periods:
            co2_emissions.loc[dict(period=y, Generator=gen)] = (
                gen_emissions.loc[n.generators.loc[gen, "carrier"], y] / n.generators.loc[gen, "efficiency"]
            )
    lhs = (gen_p * co2_emissions).sum("Generator").groupby("period").sum()

    annual_limits = pd.read_excel(
        os.path.join(scenario_setup["sub_path"], "carbon_constraints.xlsx"), 
        sheet_name="annual_carbon_constraint",
        index_col=[0]).loc[scenario_setup["co2_constraints"]]

    conv = 1
    if annual_limits.unit.split("/")[0] == "Mt":
        conv = 1e6
    elif annual_limits.units.split("/")[0] == "Gt":
        conv=1e9

    rhs = (annual_limits.loc[n.investment_periods] * conv)
    rhs.index.name = "period"
    n.model.add_constraints(lhs, "<=", rhs, name = 'annual_co2_limits')
