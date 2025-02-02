import logging
import re
from collections import ChainMap, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

DATETIME_FORMAT = '%Y-%m-%d'
DATETIME_LENGTH = len('YYYY-MM-DD')


def print_all_rows_with_value_selected(dataframes: dict[str, pd.DataFrame], col: str, vals: set[Any]) -> None:
    from IPython.core.display import display
    for file_name, df in dataframes.items():
        if col in df.columns:
            if len(vals & set(df[col])) > 0:
                print(file_name)
                display(df[df[col].isin(list(vals))])
                print()


TO_EXCLUDE = {'EUA', 'ECCE', 'Sling', 'silicone tube', 'Conjunctival nevus', 'Excision', 'P&I', 'irrigation', 'probing', 'RE P & I'}


def complaints_to_surgeries(complaints: pd.DataFrame) -> pd.DataFrame:
    surgeries = complaints[
        complaints['Surgeries'].fillna('').str.contains('[a-zA-Z]') &
        (~pd.DataFrame({s: complaints['Surgeries'].str.contains(s, case=False, regex=False) for s in TO_EXCLUDE}).any(axis=1)) &
        # TODO: Generates "FutureWarning: Downcasting object dtype arrays on .fillna is deprecated"
        (~complaints['ChiefComplaint'].fillna("").str.contains('טסט'))
    ].reset_index(drop=True)
    surgeries['Taarich'] = surgeries['Taarich'].astype('string')
    return surgeries


DEV_TYPES = {'Dsc', 'Nsc', 'AddSC', 'Dcc', 'Ncc', 'AddCC'}
# noinspection SpellCheckingInspection
CATEGORY_PER_FACTOR = {
    -1: {'E', 'ET', 'E(T)', 'LET', 'RET', 'L ET', 'R ET', 'LE(T)'},
    1: {'X', 'XT', 'X(T)', 'LXT', 'RXT', 'R XT', 'L XT', 'LX(T)'},
    0: {'ORTHO', '0'},
    -999: {'RHT', 'RH', 'LHT', 'LH', 'LH(T)', '<MISSING>', '', 'NAN', 'DVD', 'LHYPO', 'RHYPO', 'HYPO', 'DVD', 'F DVD', 'RDVD', 'LDVD', 'DVD R'},
}
FACTOR_PER_CATEGORY = {category: factor for factor, categories in CATEGORY_PER_FACTOR.items() for category in categories}


def process_value_string(s: str) -> str:
    # noinspection SpellCheckingInspection
    return s.strip().upper().replace('FLICK', '2').replace('FLCK', '2').replace('FLIK', '2').replace('DEG', '')


def process_deviation(dev: pd.Series, ignored: dict[str, list]) -> pd.Series:
    code = dev['Code']
    try:
        result = {}
        for dev_type in DEV_TYPES:
            for suffix in ('', 'Line2', 'Line3'):
                for n in ('1', '2'):
                    value_column = f'{dev_type}{n}Free{suffix}'
                    category_column = f'{dev_type}{n}{suffix}'
                    from_category_column = process_value_string(str(dev.get(category_column, '<MISSING>')))
                    value_from_category_column, resolved_category = extract_pattern(from_category_column, re.compile(r'[0-9]+'))
                    factor = FACTOR_PER_CATEGORY.get(resolved_category)

                    from_value_column = process_value_string(str(dev.get(value_column, '<MISSING>')))
                    value_and_category = from_value_column + ' ' + from_category_column
                    num_value = None

                    try:
                        num_value = int(from_value_column)
                    except ValueError:
                        if value_from_category_column is not None:
                            num_value = int(value_from_category_column)

                    if factor is None and from_value_column != 'NAN':
                        category = from_value_column
                        value_from_value_column, resolved_category = extract_pattern(category, re.compile(r'[0-9]+'))
                        factor = FACTOR_PER_CATEGORY.get(resolved_category)
                        if value_from_value_column is not None:
                            num_value_from_value_column = int(value_from_value_column)
                            if num_value is None:
                                num_value = num_value_from_value_column
                            else:
                                assert num_value == num_value_from_value_column, f"Values don't match: {num_value} != {num_value_from_value_column}"

                    if factor is None:
                        ignored[str(value_and_category)].append(code)
                        continue
                    if factor == 0:
                        final_value = 0
                    else:
                        if factor == -999 and num_value != 0:
                            continue
                        if num_value is None:
                            ignored[str(value_and_category)].append(code)
                            continue
                        final_value = factor * num_value

                    if result.get(dev_type, final_value) != final_value:
                        del result[dev_type]
                        ignored['conflict'].append(code)
                        continue
                    result[dev_type] = final_value
        return pd.Series({f'Resolved_{dev_type}': value for dev_type, value in result.items()})
    except Exception as e:
        raise ValueError(f"Error in deviation: {dev.to_dict()}") from e


def process_deviations(dev_df: pd.DataFrame) -> pd.DataFrame:
    ignored = defaultdict(list)
    new_columns = dev_df.apply(partial(process_deviation, ignored=ignored), axis=1)
    conflicting_columns = set(dev_df.columns.intersection(new_columns.columns))
    assert len(conflicting_columns) == 0, f"Conflicting columns: {conflicting_columns}"
    single_items = []
    for key, codes in ignored.items():
        if len(codes) == 1:
            single_items.append(key)
        else:
            logging.warning(f"Ignored {len(codes)} items with key {key}: {codes}")
    if len(single_items) > 0:
        logging.warning(f"Ignored {len(single_items)} items with various keys: {single_items}")
    return pd.concat([dev_df, new_columns], axis=1)


EYES = ('LE', 'RE')
MUSCLES = ('LR', 'MR', 'IO', 'SR', 'IR', 'SO')
DELIM_PAT = re.compile(r'\+|\r\n|\n|:')
MM_PAT = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*(?:mm)?', flags=re.IGNORECASE)
BE_MM_PAT_S = re.compile(r'[0-9]+(?:\.[0-9]+)?\s*/\s*[0-9]+(?:\.[0-9]+)?')
BE_MM_PAT_L = re.compile(r'\bL\s*([0-9]+(?:\.[0-9]+)?)\s*(?:[mM]{2})?')
BE_MM_PAT_R = re.compile(r'\bR\s*([0-9]+(?:\.[0-9]+)?)\s*(?:[mM]{2})?')
RECESS_PAT = re.compile(r'\b(:?recess|recesws|ecess|recerss|rerecess)', flags=re.IGNORECASE)
ADVANCE_PAT = re.compile(r'\b(:?advance|rsect|resect|advancement)\b', flags=re.IGNORECASE)
EYE_PAT = re.compile(r'^(:?RE|[rR][eE]\b|LE|[lL][eE]\b|BE|[bB][eE]\b)')
MUSCLE_PAT = re.compile(r'\b(:?' + '|'.join(MUSCLES) + r')\b', flags=re.IGNORECASE)
DATE_PAT = re.compile(r'[0-9]{1,2}/[0-9]{1,2}/(?:[0-9]{2}|[0-9]{4})')
BAD_DATE_PAT = re.compile(r'\b(:?19|20)[0-9]{2}\b')


SNELLEN_PAT = re.compile(r'([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)([+-]*)')


def log_mar(x: str) -> float:
    if x is None:
        return np.nan
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    assert isinstance(x, str), f"Not a string: {x}"
    mat = SNELLEN_PAT.match(x.strip())
    if mat is None:
        return np.nan
    res = -np.log10(float(mat[1]) / float(mat[2]))
    for sign in mat[3]:
        res += 0.02 if sign == '+' else -0.02
    return res


def apply_log_mar(df: pd.DataFrame, cols: frozenset[str]) -> pd.DataFrame:
    new_columns = df[cols].map(log_mar, na_action='ignore').rename(columns={c: f'{c}_LogMar' for c in cols})
    conflicting_columns = set(df.columns.intersection(new_columns.columns))
    assert len(conflicting_columns) == 0, f"Conflicting columns: {conflicting_columns}"
    return pd.concat([df, new_columns], axis=1)


def calc_se(ref: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        index=ref.index,
        data={
            'LE_SE': pd.to_numeric(ref['SPHLE'], 'coerce') + 0.5 * pd.to_numeric(ref['CylLE'], 'coerce').fillna(0),
            'RE_SE': pd.to_numeric(ref['SPHRE'], 'coerce') + 0.5 * pd.to_numeric(ref['CylRE'], 'coerce').fillna(0),
        },
    )


def process_refraction(ref_df: pd.DataFrame) -> pd.DataFrame:
    new_columns = calc_se(ref_df)
    conflicting_columns = set(ref_df.columns.intersection(new_columns.columns))
    assert len(conflicting_columns) == 0, f"Conflicting columns: {conflicting_columns}"
    return pd.concat([
        ref_df,
        new_columns,
    ], axis=1)


def extract_pattern(s: str, pat: re.Pattern) -> tuple[str | None, str]:
    match = pat.search(s)
    if match is None:
        return None, s
    extracted = match[0] if match.lastindex is None else match[match.lastindex]
    return extracted, (s[:match.start()].strip() + ' ' + s[match.end():].strip()).strip()


# noinspection SpellCheckingInspection
CATEGORY_ALIASES = {
    'Myectomy': {'Myectomy of', 'meyctomy', },
    'Myectomy of adhesions': {'Myectomy of adhesions of', },
    'Y splitting': {'Y spliting of', 'with Y splitting'},
    'Supraplacement': {'Supraplacement of'},
    'Synechiolysis': {'Synechiolysis of', },
    'Lasso operation': {'Lasso operation on'},
    'Plication': {'Plication of'},
}

CATEGORY_ALIAS_INDEX = dict(ChainMap(*(
    {category.lower(): category, **{alias.lower(): category for alias in aliases}}
    for category, aliases in CATEGORY_ALIASES.items()
)))


def process_surgery_part(part: str) -> dict[str, Any]:
    result = {}
    part = part.strip()
    eye, part = extract_pattern(part, EYE_PAT)
    if eye is not None:
        eye = eye.upper()
        result['eyes'] = EYES if eye == 'BE' else (eye, )

    be_num_mm, part = extract_pattern(part, BE_MM_PAT_S)
    if be_num_mm is not None:
        assert eye is None or eye == 'BE'
        num_mm = tuple(reversed(tuple(float(x) for x in be_num_mm.split('/'))))
        result['num_mm'] = num_mm
    else:
        be_num_mm_l, part = extract_pattern(part, BE_MM_PAT_L)
        be_num_mm_r, part = extract_pattern(part, BE_MM_PAT_R)
        assert (be_num_mm_l is None and be_num_mm_r is None) or (be_num_mm_l is not None and be_num_mm_r is not None)
        if be_num_mm_l is not None:
            assert eye is None or eye == 'BE'
            num_mm = (float(be_num_mm_l), float(be_num_mm_r))
            result['num_mm'] = num_mm
        else:
            num_mm, part = extract_pattern(part, MM_PAT)
            if num_mm is not None:
                num_mm = (float(num_mm), ) * len(result.get('eyes', ('dummy', )))
                result['num_mm'] = num_mm

    recess, part = extract_pattern(part, RECESS_PAT)
    if recess is not None and num_mm is not None:
        result['num_mm'] = tuple(-x for x in num_mm)
    else:
        _, part = extract_pattern(part, ADVANCE_PAT)

    muscle, part = extract_pattern(part, MUSCLE_PAT)
    if muscle is not None:
        result['muscle'] = muscle.upper()

    if len(part) > 0:
        result['category'] = CATEGORY_ALIAS_INDEX.get(part.lower(), part)

    return result


def process_surgery_part_recursive(part: str) -> tuple[dict[str, Any], ...]:
    part_dict = process_surgery_part(part)
    part_dict_list = [part_dict]
    while True:
        category = part_dict.get('category')
        if category is None:
            break
        new_part_dict = process_surgery_part(category)
        if {'muscle', 'num_mm'} <= set(new_part_dict.keys()):
            part_dict.pop('category')
            part_dict_list.append(new_part_dict)
        else:
            break
        part_dict = new_part_dict
    return tuple(part_dict_list)


def merge_muscle(d: dict[str, Any], other_category: str | None, other_num_mm: float = 0.0) -> dict[str, Any]:
    result = {}
    num_mm = d.get('num_mm', 0.0) + other_num_mm
    if num_mm != 0.0:
        result['num_mm'] = num_mm

    category = d.get('category')
    if category is None:
        category = other_category
    else:
        if other_category is not None:
            category = f"{category} {other_category}"
    if category is not None:
        result['category'] = category

    return result


# noinspection SpellCheckingInspection
GLOBAL_CATEGORY_ALIASES = {
    'Infraplacement': {'infraplacement Half width', 'Down shift infraplacement', 'infraplacement half width BE', 'Ifraplacement'},
    'Kestenbaum': {'Kestenbaum procedure', 'Kestenbaum procediure'},
    'Knapp': {'Knapp procedure', 'Knapp Procedure RE', 'Knapp procidure', 'LE Knapp procedure'},
    'Post. Fixation suture': {
        'BE Post. Fixation suture', 'BE Post. fixation suture 14 mm', 'BE Post. fixation suture 15 mm', 'Posterior fixation suture',
        'Post fixation suture on RMR',
    },
    'Supraplacement': {
        'Supraplacement Half muscle', 'Supraplacement half width', 'supraplacement half width', 'supra half', 'supraplaxement',
        'Supraplacemnt', 'Supra placement', 'Supra',
    },
    'Y splitting': {'Y spliting', 'LLR Y splitting', 'Y split of RLR'},
}

GLOBAL_CATEGORY_ALIAS_INDEX = dict(ChainMap(*(
    {category.lower(): category, **{alias.lower(): category for alias in aliases}}
    for category, aliases in GLOBAL_CATEGORY_ALIASES.items()
)))


def process_surgery(sur: str) -> dict[str, Any]:
    result: dict[str, Any] = {}

    resolved_bad_date, s = extract_pattern(sur, BAD_DATE_PAT)
    if resolved_bad_date is not None:
        # Bad dates not supported
        return {'unprocessed': sur}

    resolved_date, s = extract_pattern(sur, DATE_PAT)
    if resolved_date is not None:
        result['date'] = datetime.strptime(resolved_date, '%d/%m/%y').strftime(DATETIME_FORMAT)
        other_resolved_date, _ = extract_pattern(s, DATE_PAT)
        if other_resolved_date is not None:
            # Multiple dates not supported
            return {'unprocessed': sur}

    unprocessed = []
    eyes = None
    muscle = None
    for part in DELIM_PAT.split(s):
        try:
            part_dicts = process_surgery_part_recursive(part)
        except Exception as e:
            raise ValueError(f"Error processing part: {part}") from e
        for part_dict in part_dicts:
            if 'num_mm' not in part_dict:
                muscle = None
            eyes = part_dict.pop('eyes', eyes)
            muscle = part_dict.pop('muscle', muscle)
            if eyes is None or muscle is None:
                new_unprocessed = part.strip()
                if len(new_unprocessed) > 0:
                    unprocessed.append(new_unprocessed)
                continue
            num_mms = part_dict.get('num_mm', (0.0, ) * len(eyes))
            if len(num_mms) == 1 and len(eyes) == 2:
                num_mms = num_mms * 2

            for eye, num_mm in zip(eyes, num_mms):
                # noinspection PyTypeChecker
                eye_result = result.setdefault(eye, {})
                eye_result[muscle] = merge_muscle(eye_result.get(muscle, {}), part_dict.get('category'), num_mm)

    if len(unprocessed) > 0:
        remaining_unprocessed = []
        for unprocessed_part in unprocessed:

            processed = False
            part_dict = process_surgery_part(unprocessed_part)
            eyes = part_dict.get('eyes')
            num_mms = part_dict.get('num_mm')
            if eyes is not None and num_mms is not None and 'muscle' not in part_dict:
                for eye, num_mm in zip(eyes, num_mms):
                    eye_result = result.get(eye)
                    if eye_result is not None:
                        mr_num_mm = eye_result.get('MR', {}).get('num_mm')
                        if mr_num_mm is not None and np.sign(mr_num_mm) == (-1 * np.sign(num_mm)):
                            eye_result['LR'] = merge_muscle(eye_result.get('LR', {}), part_dict.get('category'), num_mm)
                            processed = True
                        else:
                            lr_num_mm = eye_result.get('LR', {}).get('num_mm')
                            if lr_num_mm is not None and np.sign(lr_num_mm) == (-1 * np.sign(num_mm)):
                                eye_result['MR'] = merge_muscle(eye_result.get('MR', {}), part_dict.get('category'), num_mm)
                                processed = True
            if processed:
                continue

            global_category = GLOBAL_CATEGORY_ALIAS_INDEX.get(unprocessed_part.lower())
            if global_category is not None:
                result[global_category] = 1
            else:
                remaining_unprocessed.append(unprocessed_part)
        if len(remaining_unprocessed) > 0:
            result['unprocessed'] = ' '.join(remaining_unprocessed)

    return result


def parse_surgery_to_series(s: str) -> pd.Series:
    try:
        surgery_dict = process_surgery(s)
    except (ValueError , RuntimeError, AssertionError) as e:
        logging.error(f"Error processing surgery: {s}", exc_info=e)
        surgery_dict = {'unprocessed': s}
    return pd.Series(dict(ChainMap(
        {'Resolved_Taarich': surgery_dict.get('date')},
        *(
            {
                f'{eye}_{muscle}_num_mm': (cur_d := surgery_dict.get(eye, {}).get(muscle, {})).get('num_mm'),
                f'{eye}_{muscle}_category': cur_d.get('category'),
            }
            for muscle in MUSCLES
            for eye in EYES
        ),
        {global_category: surgery_dict.get(global_category, 0) for global_category in GLOBAL_CATEGORY_ALIASES.keys()},
        {'unprocessed': surgery_dict.get('unprocessed')},
    )))


def merge_ids(surgeries: pd.DataFrame, lakoah: pd.DataFrame) -> pd.DataFrame:
    return (
        surgeries.drop(columns=[
            'Allergies', 'ReferingPhysician', 'Description', 'DescriptionFlag', 'UserID', 'UserIDDate',
        ]).merge(
            lakoah[['KodReshuma', 'TaarichLeida', 'MinN']],
            how='left', left_on='LakoahCode', right_on='KodReshuma'
        )
        .drop(columns=['KodReshuma'])
    )


def to_date(s: str) -> datetime | float:
    if isinstance(s, str):
        return datetime.strptime(s[:DATETIME_LENGTH], DATETIME_FORMAT)
    return np.nan


def parse_surgeries(surgeries: pd.DataFrame) -> pd.DataFrame:
    parsed_surgeries = surgeries['Surgeries'].apply(parse_surgery_to_series)
    global_category_columns = list(set(parsed_surgeries.columns) & set(GLOBAL_CATEGORY_ALIAS_INDEX.keys()))
    parsed_surgeries[global_category_columns] = parsed_surgeries[global_category_columns].fillna(0)
    parsed_surgeries['Resolved_Taarich'] = parsed_surgeries['Resolved_Taarich'].fillna(surgeries['Taarich'])
    parsed_surgeries['Age_Years'] = (parsed_surgeries['Resolved_Taarich'].map(to_date) - surgeries['TaarichLeida'].map(to_date)).dt.days / 365
    surgeries = surgeries.drop(columns=parsed_surgeries.columns, errors='ignore')
    return pd.concat([
        surgeries.loc[:, :'Surgeries'],
        parsed_surgeries,
        surgeries.loc[:, 'Surgeries':].iloc[:, 1:],
    ], axis=1)


@dataclass
class Bracket:
    start_days_exclusive: int
    target_days: int
    end_days_inclusive: int

    def __repr__(self) -> str:
        return f"({self.start_days_exclusive}, {self.end_days_inclusive}]"


BRACKETS = (
    Bracket(21, 42, 90),
    Bracket(90, 105, 140),
    Bracket(140, 180, 220),
    Bracket(220, 270, 320),
    Bracket(320, 365, 420),
    Bracket(420, 480, 548),
    Bracket(548, 820, 1095),
    Bracket(1095, 1095, 50_000),
)


def add_deviations_after(
    surgeries: pd.DataFrame, deviations: pd.DataFrame,
    base_date_column_name: str = 'Resolved_Taarich',
    measurement_date_column_name: str = 'Taarich',
    id_column_name='LakoahCode',
) -> pd.DataFrame:
    assert deviations.columns.is_unique, f"Columns not unique in deviations"
    deviations[measurement_date_column_name] = deviations[measurement_date_column_name].astype('string')
    deviations.replace(re.compile(r'^\s*$'), np.nan, inplace=True)

    is_na_per_col = deviations.isna().all()
    deviations.drop(columns=list(is_na_per_col[is_na_per_col].index), inplace=True)

    new_columns = tuple(f'DeviationsAfter_{c}' for c in ('DaysDifference', 'Bracket', *deviations.columns))

    conflicting_columns = set(surgeries.columns) & set(new_columns)
    assert len(conflicting_columns) == 0, f"Conflicting columns: {conflicting_columns}"

    surgeries_with_deviations = pd.DataFrame(columns=(
        *surgeries.columns,
        *new_columns,
    ))

    # TODO: Inefficient, find a way to do this with vector operations.
    for _, row in surgeries.iterrows():
        patient_measurements = deviations[(deviations[id_column_name] == row[id_column_name]) & (deviations[measurement_date_column_name] > row[base_date_column_name])].copy()
        if len(patient_measurements) == 0:
            continue
        patient_measurements['days_difference'] = (
            patient_measurements[measurement_date_column_name].map(to_date) - to_date(row[base_date_column_name])
        ).dt.days
        for bracket in BRACKETS:
            patient_measurements_in_bracket = patient_measurements[
                (patient_measurements['days_difference'] > bracket.start_days_exclusive)
                & (patient_measurements['days_difference'] <= bracket.end_days_inclusive)
            ].copy()
            if len(patient_measurements_in_bracket) == 0:
                continue
            patient_measurements_in_bracket['diff_from_target'] = abs(patient_measurements_in_bracket['days_difference'] - bracket.target_days)
            target_measurement = patient_measurements_in_bracket.loc[patient_measurements_in_bracket['diff_from_target'].idxmin()]
            next_i = len(surgeries_with_deviations)
            surgeries_with_deviations.loc[next_i, row.index] = row.values
            surgeries_with_deviations.loc[next_i, 'DeviationsAfter_DaysDifference'] = target_measurement['days_difference']
            surgeries_with_deviations.loc[next_i, 'DeviationsAfter_Bracket'] = str(bracket)
            surgeries_with_deviations.loc[next_i, [f'DeviationsAfter_{c}' for c in target_measurement.index]] = target_measurement.values

    return surgeries_with_deviations


TARGET_DAYS_AFTER_FALLBACK = 42
MAX_DAYS_BEFORE = 365


def add_measurements_before_and_after(
        base_df: pd.DataFrame, measurement_name: str, measurements_df: pd.DataFrame, add_after: bool = True,
        base_date_column_name: str = 'Resolved_Taarich',
        measurement_date_column_name: str = 'Taarich',
        id_column_name='LakoahCode',
) -> pd.DataFrame:
    assert measurements_df.columns.is_unique, f"Columns not unique in {measurement_name}"
    measurements_df[measurement_date_column_name] = measurements_df[measurement_date_column_name].astype('string')
    measurements_df.replace(re.compile(r'^\s*$'), np.nan, inplace=True)

    is_na_per_col = measurements_df.isna().all()
    measurements_df.drop(columns=list(is_na_per_col[is_na_per_col].index), inplace=True)

    new_columns = pd.DataFrame({
        col: pd.Series(index=base_df.index, dtype=dtype)
        for col, dtype in (
            (f'{measurement_name}NumMeasures', 'int'),
            (f'{measurement_name}EarliestMeasure', 'string'),
            (f'{measurement_name}LatestMeasure', 'string'),
            *(
                (f'{measurement_name}Before_{c}', 'object') for c in measurements_df.columns
            ),
            *((
                (f'{measurement_name}After_{c}', 'object') for c in measurements_df.columns
            ) if add_after else tuple()),
        )
    })
    conflicting_columns = set(base_df.columns.intersection(new_columns.columns))
    assert len(conflicting_columns) == 0, f"Conflicting columns: {conflicting_columns}"

    df_with_measurements = pd.concat([base_df, new_columns], axis=1)

    # TODO: Inefficient, find a way to do this with vector operations.
    for i, row in base_df.iterrows():
        patient_measurements = measurements_df[measurements_df[id_column_name] == row[id_column_name]]
        df_with_measurements.loc[i, f'{measurement_name}NumMeasures'] = len(patient_measurements)
        df_with_measurements.loc[i, f'{measurement_name}EarliestMeasure'], df_with_measurements.loc[i, f'{measurement_name}LatestMeasure'] = (
            patient_measurements[measurement_date_column_name].agg(['min', 'max'])
        )

        measurements_before = patient_measurements[patient_measurements[measurement_date_column_name] <= row[base_date_column_name]].copy()
        if len(measurements_before) > 0:
            measurements_before['days_difference'] = (
                    to_date(row[base_date_column_name]) - measurements_before[measurement_date_column_name].map(to_date)
            ).dt.days
            measurements_before = measurements_before[measurements_before['days_difference'] <= MAX_DAYS_BEFORE].copy()
            if len(measurements_before) > 0:
                latest_measurement_before = measurements_before.loc[measurements_before[measurement_date_column_name].idxmax()]
                df_with_measurements.loc[i, [f'{measurement_name}Before_{c}' for c in latest_measurement_before.index]] = latest_measurement_before.values

        if add_after:
            measurements_after = patient_measurements[patient_measurements[measurement_date_column_name] > row[base_date_column_name]].copy()
            if len(measurements_after) > 0:
                measurements_after['days_difference'] = (
                        measurements_after[measurement_date_column_name].map(to_date) - to_date(row[base_date_column_name])
                ).dt.days

                if len(measurements_after) > 0:
                    target_after = to_date(row[f'DeviationsAfter_{measurement_date_column_name}'])
                    if target_after is None or (isinstance(target_after, float) and np.isnan(target_after)):
                        target_after = to_date(row[base_date_column_name]) + timedelta(days=TARGET_DAYS_AFTER_FALLBACK)
                    measurements_after['diff_from_target'] = abs((measurements_after[measurement_date_column_name].map(to_date) - target_after).dt.days)
                    target_measurement_after = measurements_after.loc[measurements_after['diff_from_target'].idxmin()]
                    df_with_measurements.loc[i, [f'{measurement_name}After_{c}' for c in target_measurement_after.index]] = target_measurement_after.values

    is_na_per_col = df_with_measurements.isna().all()
    df_with_measurements.drop(columns=list(is_na_per_col[is_na_per_col].index), inplace=True)

    return df_with_measurements


def add_all_measurements(surgeries: pd.DataFrame, dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    deviations = process_deviations(dataframes['Deviations_measures.csv'])
    logging.info(f"Processed Deviations: {deviations.shape}")
    surgeries_with_measurements = add_deviations_after(surgeries, deviations)
    logging.info(f"Added Deviations after: {surgeries_with_measurements.shape}")
    surgeries_with_measurements = add_measurements_before_and_after(surgeries_with_measurements, 'Deviations', deviations, add_after=False)
    logging.info(f"Added Deviations before: {surgeries_with_measurements.shape}")

    for measurement_name, file_name, preprocessor in (
            ('NCVA', 'NCVA.csv', partial(apply_log_mar, cols=frozenset(['NccLE', 'NccRE', 'NccBE']))),
            ('DCVA', 'DCVA.csv', partial(apply_log_mar, cols=frozenset(['DccLE', 'DccRE', 'DccBE']))),
            ('D_SC_VA', 'D SC VA.csv', partial(apply_log_mar, cols=frozenset(['DscLE', 'DscRE', 'DscBE']))),
            ('Streopsis', 'Sensory tests streopsis.csv', None),
            ('Refraction', 'Refraction.csv', process_refraction),
    ):
        measurements_df = dataframes[file_name]
        if preprocessor is not None:
            try:
                measurements_df = preprocessor(measurements_df)
            except Exception as e:
                raise ValueError(f"Error processing {measurement_name}") from e
            logging.info(f"Processed {file_name}: {measurements_df.shape}")
        surgeries_with_measurements = add_measurements_before_and_after(surgeries_with_measurements, measurement_name, measurements_df)
        logging.info(f"Added {measurement_name}: {surgeries_with_measurements.shape}")
    surgeries_with_measurements = combine_acuity(surgeries_with_measurements)
    return surgeries_with_measurements


def combine_acuity(df: pd.DataFrame) -> pd.DataFrame:
    df['AcuityBefore_DxcLE_LogMar'] = df['DCVABefore_DccLE_LogMar'].fillna(df['D_SC_VABefore_DscLE_LogMar'])
    df['AcuityBefore_is_corrected_left'] = df['DCVABefore_DccLE_LogMar'].notna()
    df['AcuityBefore_DxcRE_LogMar'] = df['DCVABefore_DccRE_LogMar'].fillna(df['D_SC_VABefore_DscRE_LogMar'])
    df['AcuityBefore_is_corrected_right'] = df['DCVABefore_DccRE_LogMar'].notna()
    df['AcuityBefore_DxcBE_LogMar'] = df['DCVABefore_DccBE_LogMar'].fillna(df['D_SC_VABefore_DscBE_LogMar'])
    df['AcuityBefore_is_corrected_be'] = df['DCVABefore_DccBE_LogMar'].notna()
    return df
