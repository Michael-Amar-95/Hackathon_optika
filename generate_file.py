import logging
from os import listdir
from pathlib import Path

import pandas as pd

from utils.parsing import complaints_to_surgeries, merge_ids, parse_surgeries, add_all_measurements
from utils.general import setup_logging

# Produced files using commands of the form:
# Get list of tables:
# > mdb-tables WinOptika.mdb
# For each table:
# > mdb-export WinOptika.mdb VisualAcuityNcc > VisualAcuityNcc.csv


setup_logging()

base_path = Path(__file__).parent / 'haim_data'

dataframes = {
    file_name: pd.read_csv(base_path / file_name, low_memory=False)
    for file_name in listdir(base_path)
    if file_name.endswith('.csv')
}

logging.info(f"Dataframes: { {name: df.shape for name, df in dataframes.items()} }")

surgeries = complaints_to_surgeries(dataframes['Chief complaint_surg dates.csv'])

logging.info(f"Surgeries: {surgeries.shape}")

surgeries = merge_ids(surgeries, dataframes['Lakoah.csv'])

logging.info(f"Surgeries with ids: {surgeries.shape}")

surgeries = parse_surgeries(surgeries)

logging.info(f"Parsed surgeries: {surgeries.shape}")


surgeries = add_all_measurements(surgeries, dataframes)

logging.info(f"Surgeries with measurements: {surgeries.shape}")

output_path = base_path / 'surgeries_with_measurements.csv'
surgeries.to_csv(output_path)

logging.info(f"Output saved to {output_path}")
