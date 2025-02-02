import logging
from os import listdir
from pathlib import Path

import pandas as pd

from utils.general import setup_logging
from utils.ichilov_parsing import add_all_measurements_ichilov, merge_pat_details

setup_logging()

base_path = Path(__file__).parent / 'ichilov_data'

dataframes = {
    file_name: pd.read_csv(base_path / file_name, low_memory=False)
    for file_name in listdir(base_path)
    if file_name.endswith('.csv')
}

logging.info(f"Dataframes: { {name: df.shape for name, df in dataframes.items()} }")

surgeries_with_pat_details = merge_pat_details(
    dataframes['sweat_shop_outcome_merged.csv'], dataframes['surgery.csv']
)

logging.info(f"Surgeries with patient details: {surgeries_with_pat_details.shape}")

surgeries_with_measurements = add_all_measurements_ichilov(
    surgeries_with_pat_details, dataframes
)

logging.info(f"Surgeries with measurements: {surgeries_with_measurements.shape}")

output_path = base_path / 'surgeries_with_measurements_ichilov.csv'
surgeries_with_measurements.to_csv(output_path)

logging.info(f"Output saved to {output_path}")
