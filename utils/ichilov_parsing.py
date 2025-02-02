import logging
from typing import cast, Callable

import numpy as np
import pandas as pd
import re

from utils.parsing import add_deviations_after, add_measurements_before_and_after, process_refraction, to_date, apply_log_mar

# missing left right eye info
# missing flick
# ALT XT25

ORTHO = ["ortho", r"\b0\b", 'flick', 'flixk']

XSTR = ["x", r"\(t\)", "xt", "x"]
ESTR = ["et", r"e\(t\)", "e"]


DEV_TYPE_PART = re.compile('|'.join(ESTR+XSTR+ORTHO))
NUM_OR_RANGE = re.compile(r"\d+(-\d+)?")


def extract_type_mag(dev: str) -> float:
    if pd.isna(dev):
        return np.nan

    magnitude = None
    for parts in dev.lower().split('+'):
        dev_type_part_match = DEV_TYPE_PART.search(parts)
        if dev_type_part_match:
            dev_type = dev_type_part_match.group(0)
            if re.compile('|'.join(ORTHO)).search(dev_type):
                return 0
            elif re.compile('|'.join(XSTR)).search(dev_type):
                dev_type = 1
            elif re.compile('|'.join(ESTR)).search(dev_type):
                dev_type = -1
            num_or_range_match = NUM_OR_RANGE.search(parts)
            if num_or_range_match:
                dev_range = num_or_range_match.group(0)
                if '-' in dev_range:
                    low, high = map(int, dev_range.split('-'))
                    magnitude = (low + high) / 2  # Compute the midpoint as a float
                elif dev_range:
                    magnitude = int(dev_range)
                return magnitude * dev_type


def parse_deviations(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{
            f'Resolved_{ND[0]}{is_correction.lower()}c': df[f'{ND}_{is_correction}C_Center'].apply(extract_type_mag) for is_correction in ['S', 'C'] for ND in ['Near', 'Distance']
        }
    )


acuity_rename_dict = {
    # Refraction SE
    'Cyclopegic_Refraction_Sphere': 'SPHLE',
    'Cyclopegic_Refraction_Left_Cyl': 'CylLE',
    'Cyclopegic_Refraction_Right_Sphere': 'SPHRE',
    'Cyclopegic_Refraction_Right_Cyl': 'CylRE',

    # Acuity
    'Hadut_with_glasses_left': 'DxcLE',
    'Hadut_with_glasses_left_glasses': 'is_corrected_left',
    'Hadut_with_glasses_right': 'DxcRE',
    'Hadut_with_glasses_right_glasses': 'is_corrected_right',
}


def merge_pat_details(processed_surgeries: pd.DataFrame, original_surgeries: pd.DataFrame) -> pd.DataFrame:
    merged_surgeries = (
        processed_surgeries.drop(columns=['Birth_Date']).merge(
            original_surgeries[[
                'pat_id', 'Gender_Text', 'Birth_Date', 'Entry_Date'
            ]].drop_duplicates(subset=(['pat_id', 'Entry_Date'])),
            how='left', on=['pat_id', 'Entry_Date'],
        )
    )
    merged_surgeries['Age_Years'] = (merged_surgeries['Entry_Date'].map(to_date) - merged_surgeries['Birth_Date'].map(to_date)).dt.days / 365
    merged_surgeries['MinN'] = merged_surgeries['Gender_Text'].map({'זכר': 0, 'נקבה': 1})
    return merged_surgeries


def add_all_measurements_ichilov(surgeries: pd.DataFrame, dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    column_names = dict(
        base_date_column_name='Entry_Date',
        measurement_date_column_name='Entry_Date',
        id_column_name='pat_id',
    )
    deviations = parse_deviations(dataframes['COVERTEST.csv'])
    logging.info(f"Processed Deviations: {deviations.shape}")
    surgeries_with_measurements = add_deviations_after(surgeries, deviations, **column_names)
    logging.info(f"Added Deviations after: {surgeries_with_measurements.shape}")
    surgeries_with_measurements = add_measurements_before_and_after(surgeries_with_measurements, 'Deviations', deviations, add_after=False,  **column_names)
    logging.info(f"Added Deviations before: {surgeries_with_measurements.shape}")

    # TODO: Fix acuity: only one column and then another which says if it's SC or CC. If both exist use CC.
    for measurement_name, file_name, preprocessor in (
            ('Streopsis', 'SENSORYTEST.csv', None),
            ('Diagnosis', 'EYE DIAGNOSIS.csv', None),
            (
                'Acuity',
                'ACUITY TEST.csv',
                lambda df: apply_log_mar(
                    process_refraction(
                        df.rename(columns=acuity_rename_dict)
                    ),
                    cols=frozenset(['DxcLE', 'DxcRE']),
                )
            ),
    ):
        measurements_df = dataframes[file_name]
        if preprocessor is not None:
            try:
                measurements_df = cast(Callable, preprocessor)(measurements_df)
            except Exception as e:
                raise ValueError(f"Error processing {measurement_name}") from e
            logging.info(f"Processed {file_name}: {measurements_df.shape}")
        surgeries_with_measurements = add_measurements_before_and_after(surgeries_with_measurements, measurement_name, measurements_df,  **column_names)
        logging.info(f"Added {measurement_name}: {surgeries_with_measurements.shape}")

    return surgeries_with_measurements


# TODO: Use this to convert ichilov columns to haim columns
rename_dict = {
    'pat_id': 'LakoahCode',
    'Entry_Date': 'Taarich',
    'RE_MR': 'RE_MR_num_mm',
    'LE_MR': 'LE_MR_num_mm',
    'RE_LR': 'RE_LR_num_mm',
    'LE_LR': 'LE_LR_num_mm',
    'AcuityBefore_LE_SE': 'RefractionBefore_LE_SE',
    'AcuityBefore_RE_SE': 'RefractionBefore_RE_SE',
    'StreopsisBefore_stereopsis': 'StreopsisBefore_StereoCC',  # Also copy to 'StreopsisBefore_StereoSC'
}
