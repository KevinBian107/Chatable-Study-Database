import pandas as pd
import numpy as np
from pathlib import Path

pd.options.plotting.backend = "plotly"

GROUP_COLOR_MAP = {
    "ds": "blue",
    "cogs": "purple",
    "math": "orange",
    "pp": "red",
    "ge": "yellow",
    "research": "green",
}

COL_GROUP = {
    "training_spring25": "pp",
    "salk_spring25": "research",
    "linderman_lab_spring25": "research",
    "mpi_spring25": "research",
    "dsc214": "research",
    "dsc102": "ds",
    "math214": "math",
    "psyc179": "cogs",
    "psyc132": "cogs",
    "ece171a": "ds",
    "chem11": "ge",
    "startup": "research",
    "pp_spring25": "pp",
    "training_winter25": "pp",
    "salk_winter25": "research",
    "linderman_lab_winter25": "research",
    "rplh_winter25": "research",
    "mpi_winter25": "research",
    "math190a": "math",
    "math189": "math",
    "cse257_winter25": "ds",
    "cse251b": "ds",
    "cse234": "ds",
    "cogs164": "cogs",
    "cogs101a": "cogs",
    "pp_winter25": "pp",
    "training_fall24": "pp",
    "salk_fall24": "research",
    "mpi_fall24": "research",
    "math173a": "math",
    "math142a": "math",
    "cse258": "ds",
    "dsc190_fall24": "ds",
    "dsc100": "ds",
    "math20d": "math",
    "pp_fall24": "pp",
    "doc1": "ge",
    "cogs9": "cogs",
    "pp_fall22": "pp",
    "reading_winter23": "pp",
    "pp_winter23": "pp",
    "math20c": "math",
    "dsc20": "ds",
    "bild1": "cogs",
    "bild22": "cogs",
    "cogs87": "cogs",
    "doc2": "ge",
    "training_spring23": "pp",
    "reading_spring23": "pp",
    "pp_spring23": "pp",
    "cogs14a": "cogs",
    "cogs17": "cogs",
    "dsc30": "ds",
    "math20e": "math",
    "doc3": "ge",
    "training_summer23": "pp",
    "reading_summer23": "pp",
    "pp_summer23": "pp",
    "ds_summer_project": "ds",
    "dsc40a": "ds",
    "training_fall23": "pp",
    "ex_phys": "cogs",
    "fmp_fall23": "research",
    "cogs153": "cogs",
    "cogs107a": "cogs",
    "cogs14b": "cogs",
    "dsc40b": "ds",
    "math180a": "math",
    "kdd/ds3/tnt": "pp",
    "pp_fall23": "pp",
    "training_winter24": "pp",
    "salk_winter24": "research",
    "cse257_winter24": "ds",
    "dsc80": "ds",
    "cogs107b": "cogs",
    "math180b": "math",
    "fmp_winter24": "research",
    "pp_winter24": "pp",
    "salk_spring24": "research",
    "dsc140a": "ds",
    "dsc140b": "ds",
    "math180c": "math",
    "math181a": "math",
    "cogs107c": "cogs",
    "psyc137": "cogs",
    "dsc106": "ds",
    "cse150b": "ds",
    "fmp_spring24": "research",
    "pp_spring24": "pp",
    "training_summer24": "pp",
    "salk_summer24": "research",
    "mpi_summer24": "research",
    "cogs180": "cogs",
    "pp_sumer24": "pp",
    "ds_project_summer24": "pp",
    "reading_summer24": "pp",
    "driving": "pp",
    "math18": "math",
    "dsc10": "ds",
    "math20b": "math",
}


def transform_study(study):
    """First step neccesssary conversions, specific to my dataset"""

    # Timestamp Conversion
    format = "%m/%d/%y"
    study["date"] = pd.to_datetime(study["date"], format=format)

    # Group by date to get unique data set on dates
    grouped_study = study.groupby("date").mean().fillna(0).reset_index()

    # add columns
    grouped_study = grouped_study.assign(
        math18=grouped_study["math18hw"]
        + grouped_study["math18review"]
        + grouped_study["math18matlab"]
    )
    grouped_study = grouped_study.assign(
        dsc10=grouped_study["dsc10review"] + grouped_study["dsc10hw"]
    )
    grouped_study = grouped_study.assign(
        math20b=grouped_study["math20breview"] + grouped_study["math20bhw"]
    )

    # drop and rename
    grouped_study = grouped_study.drop(
        columns=[
            "math18hw",
            "math18review",
            "math18matlab",
            "dsc10review",
            "dsc10hw",
            "math20breview",
            "math20bhw",
        ]
    ).rename(columns={"ds": "ds_summer_project"})

    # convert
    number_col = grouped_study.select_dtypes(include="number").columns
    grouped_study[number_col] = grouped_study[number_col] / 60

    grouped_study["week"] = grouped_study["week"] * 60

    return grouped_study
