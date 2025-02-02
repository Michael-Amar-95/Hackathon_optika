import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel, fisher_exact
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from predictor import Predictor
from utils.baseline import MONO_BINO, MONO, BINO, BASELINES
from utils.general import setup_logging
from utils.parsing import to_date
from utils.plotting import subplots

setup_logging()

# TODO: For missing features: retrain after imputing with coefficient of corresponding one-hot encoded feature.
# TODO: Review unprocessed surgery text and categories
# TODO: Review largest prediction errors in train set.
# TODO: Use non-linear fit?
# TODO: Add more Interaction terms?
# TODO: Cross-validation
# TODO: Check for multicollinearity?
# TODO: Feature selection: remove feature with highest p-value, retrain, repeat until all p-values are below threshold.
# TODO: Add feature for the type of surgery (LR both eyes, MR both eyes, LR+MR one eye)?
# TODO: Add feature for positive/negative initial deviation?
# TODO: Try reverting to measuring deviation after, with initial deviation as a feature. Maybe add interaction term for initial deviation + surgery mm.
# TODO: Add measurement time difference as feature?
# TODO: Trick for estimating overfit for small data sets: train with random shuffles of the targets, and get the distribution of errors. If the model is memorizing the targets, it would be able to do it also with random targets.


@dataclass
class TrainerConfig:
    filepath: str
    # If a feature is not-NaN on less than this fraction of rows, it will be dropped
    remove_rows_with_multiple_dates: bool = True
    feature_present_threshold: float = 0.2
    test_fraction: float = 0.2
    use_kfold: bool = True
    kfold_splits: int = 5
    random_state: int = 42
    with_intercept: bool = True
    # If this is true all non-surgery features are only used in interactions with surgery features
    only_interactions_with_surgery_features: bool = True
    # If this is true, add one-hot encoded features that indicate if a feature was missing
    one_hot_encode_missing_feature: bool = True  # TODO: Is this a good idea?
    impute_missing_feature: bool = False  # TODO: Do this on each k fold
    # If this is true, flip negative num_mm values to positive so the model can train on both simultaneously
    fix_num_mm_sign: bool = True
    success_threshold: float = 10
    use_engineered_and_extra_features: bool = False
    train_several_models: bool = False
    surgery_features_: tuple[str, ...] = (
        # 'RE_SO_num_mm', 'LE_SO_num_mm', 'RE_IR_num_mm', 'LE_IR_num_mm',
        # 'RE_SR_num_mm', 'LE_SR_num_mm', 'RE_IO_num_mm', 'LE_IO_num_mm',
        'RE_MR_num_mm', 'LE_MR_num_mm', 'RE_LR_num_mm', 'LE_LR_num_mm',
    )
    category_features_: tuple[str, ...] = (
        'RE_SO_category', 'LE_SO_category', 'RE_IR_category', 'LE_IR_category',
        'RE_SR_category', 'LE_SR_category', 'RE_IO_category', 'LE_IO_category',
        'RE_MR_category', 'LE_MR_category', 'RE_LR_category', 'LE_LR_category',
    )
    deviation_features_: tuple[str, ...] = (
        'DeviationsBefore_Resolved_Dsc',
        'DeviationsBefore_Resolved_Nsc',
        'DeviationsBefore_Resolved_Dcc',
        'DeviationsBefore_Resolved_Ncc',
        # 'DeviationsBefore_Resolved_AddCC',  # Decided not to use for now
        # 'DeviationsBefore_Resolved_AddSC',  # Missing
    )
    engineered_features_: tuple[str, ...] = (
        'StreopsisBefore_DWorth4dotCC_fusion',
        'StreopsisBefore_NWorth4dotCC_fusion',
        'Dcc_LogMar_eyes_difference',
        'is_both_eyes',
    )
    other_features_: tuple[str, ...] = (
        # Mostly missing
        # Requires separating the zeros to a new one-hot encoded feature
        # 'StreopsisBefore_StereoCC',

        # Requires converting string values to LogMar
        # 'NCVABefore_NccLE_LogMar',
        # 'NCVABefore_NccRE_LogMar',
        # 'NCVABefore_NccBE_LogMar',

        'DCVABefore_DccLE_LogMar',
        'DCVABefore_DccRE_LogMar',
        'DCVABefore_DccBE_LogMar',

        'D_SC_VABefore_DscLE_LogMar',
        'D_SC_VABefore_DscRE_LogMar',
        'D_SC_VABefore_DscBE_LogMar',

        'RefractionBefore_LE_SE',
        'RefractionBefore_RE_SE',

        'MinN',
        'Age_Years',

        'DeviationsAfter_DaysDifference',
    )
    extra_other_features_: tuple[str, ...] = (
        'StreopsisBefore_StereoSC',
        'StreopsisBefore_StereoCC',
    )
    raw_targets_: tuple[str, ...] = (
        'DeviationsAfter_Resolved_Dsc',
        'DeviationsAfter_Resolved_Nsc',
        # 'DeviationsAfter_Resolved_AddSC',
        'DeviationsAfter_Resolved_Dcc',
        'DeviationsAfter_Resolved_Ncc',
        # 'DeviationsAfter_Resolved_AddCC',
    )
    targets_: tuple[str, ...] = (
        'DeviationsDiff_Dsc',
        'DeviationsDiff_Nsc',
        'DeviationsDiff_Dcc',
        'DeviationsDiff_Ncc',
    )
    # Keep these columns in the final DataFrame, even if they are not features or targets
    extra_columns_: tuple[str, ...] = (
        'DeviationsAfter_Bracket',
        'StreopsisBefore_DWorth4dotCC',
        'StreopsisBefore_NWorth4dotCC',
    )
    baseline_prefix: str = 'Baseline_'
    final_before: str = 'DeviationBefore'
    final_target: str = 'DeviationDiff'
    prediction: str = 'DeviationDiff_Pred'

    @property
    def surgery_features(self) -> list[str]:
        return list(self.surgery_features_)

    @property
    def category_features(self) -> list[str]:
        return list(self.category_features_)

    @property
    def deviation_features(self) -> list[str]:
        return list(self.deviation_features_)

    @property
    def non_deviation_features(self) -> list[str]:
        if self.use_engineered_and_extra_features:
            return list(self.other_features_) + self.surgery_features + list(self.engineered_features_) + list(self.extra_other_features_)
        else:
            return list(self.other_features_) + self.surgery_features

    @property
    def features(self) -> list[str]:
        return self.deviation_features + list(self.other_features_) + self.surgery_features

    @property
    def raw_targets(self) -> list[str]:
        return list(self.raw_targets_)

    @property
    def targets(self) -> list[str]:
        return list(self.targets_)

    @property
    def extra_columns(self) -> list[str]:
        return list(self.extra_columns_)


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        logging.info("File date: " + datetime.fromtimestamp(os.stat(self.config.filepath).st_mtime).strftime('%Y-%m-%d %H:%M:%S'))
        self.base_df = pd.read_csv(self.config.filepath)
        logging.info(f"Dataframe shape: {self.base_df.shape}")

    def date_diffs(self, date_col1: str, date_col2: str, df: pd.DataFrame = None, value_counts: bool = True) -> pd.Series:
        if df is None:
            df = self.base_df
        res = df[date_col2].apply(to_date) - df[date_col1].apply(to_date)
        if value_counts:
            return res.value_counts()
        else:
            return res

    def filter_df(self) -> None:
        # TODO: Should we filter out rows with categories?
        self.filtered_df = self.base_df.copy()
        self._single_filter(
            pd.concat([
                (self.filtered_df[deviation].notna() & self.filtered_df[deviation.replace('Before', 'After')].notna())
                for deviation in self.config.deviation_features
            ], axis=1).any(axis=1),
            "all deviation features missing",
        )
        self._single_filter(
            (
                self.filtered_df[self.config.surgery_features].notna()
                & (self.filtered_df[self.config.surgery_features] != 0)
            ).any(axis=1),
            "all surgery features missing or zero",
        )
        if 'Kestenbaum' in self.filtered_df:
            self._single_filter(
                self.filtered_df['Kestenbaum'].eq(0),
                "Kestenbaum procedure",
            )
        if 'Knapp' in self.filtered_df:
            self._single_filter(
                self.filtered_df['Knapp'].eq(0),
                "Knapp procedure",
            )
        self.filtered_df = self.filtered_df.reset_index(drop=True)
        logging.info(f"Filtered dataframe shape: {self.filtered_df.shape}")

    # Apply a single filter to the dataframe
    def _single_filter(self, row_filter: pd.Series, reason: str):
        num_removed = (~row_filter).sum()
        logging.info(f"Removed {num_removed} ({num_removed / len(self.base_df):.1%}) rows - {reason}")
        self.filtered_df = self.filtered_df[row_filter]

    def preprocess_df(self) -> None:
        # Convert features and targets to numeric types.
        # Non-numeric values are replaced with NaN.
        self.preprocessed_df = pd.DataFrame(
            index=self.filtered_df.index,
            data={
                col: (
                    pd.to_numeric(self.filtered_df[col], errors='coerce')
                    if col in (self.config.features + self.config.raw_targets)
                    else self.filtered_df[col]
                )
                for col in self.filtered_df.columns
            }
        )

        logging.info(f"Filling {self.preprocessed_df[self.config.surgery_features].isna().sum().sum()} missing values for surgery features with 0")
        self.preprocessed_df[self.config.surgery_features] = self.preprocessed_df[self.config.surgery_features].fillna(0)
        logging.info(f"Preprocessed dataframe shape: {self.preprocessed_df.shape}")

        # TODO: Fix NCVA - convert string values to LogMar
        # TODO: For all columns with 'VA' - separate C-S-M to separate categorical feature
        # TODO: For Streopsis separate the zero values to a separate one-hot encoded feature

    SURGERY_TYPES = {
        #  RE_MR_num_mm  LE_MR_num_mm  RE_LR_num_mm  LE_LR_num_mm      Type
           (True,        True,         True,         True):            'Both eyes, both muscles',
           (True,        True,         True,        False):            'Both eyes, both muscles',
           (True,        True,        False,         True):            'Both eyes, both muscles',
           (True,        True,        False,        False):            'Both eyes, MR',
           (True,       False,         True,         True):            'Both eyes, both muscles',
           (True,       False,         True,        False):            'One eye, both muscles',
           (True,       False,        False,         True):            'Both eyes, both muscles',
           (True,       False,        False,        False):            'One eye, MR',
           (False,       True,         True,         True):            'Both eyes, both muscles',
           (False,       True,         True,        False):            'Both eyes, both muscles',
           (False,       True,        False,         True):            'One eye, both muscles',
           (False,       True,        False,        False):            'One eye, MR',
           (False,      False,         True,         True):            'Both eyes, LR',
           (False,      False,         True,        False):            'One eye, LR',
           (False,      False,        False,         True):            'One eye, LR',
    }

    def surgery_types(self) -> dict[str, int]:
        res = self.preprocessed_df[self.config.surgery_features].ne(0).groupby(self.config.surgery_features, dropna=False).size().to_frame('row_count')
        res.index = res.index.to_series().map(self.SURGERY_TYPES)
        return res.groupby(level=0).sum().to_dict()['row_count']

    SC_CC = 'sc_cc'
    D_N = 'd_n'

    L_R_MAPPING = {
        'DCVABefore_DccLE_LogMar': 'DCVABefore_DccRE_LogMar',
        'D_SC_VABefore_DscLE_LogMar': 'D_SC_VABefore_DscRE_LogMar',
        'RefractionBefore_LE_SE': 'RefractionBefore_RE_SE',
        'LE_MR_num_mm': 'RE_MR_num_mm',
        'LE_LR_num_mm': 'RE_LR_num_mm',
    }

    def feature_engineering(self) -> None:
        self.eng_df = self.preprocessed_df.copy()
        logging.info(f"Present values per feature:\n{self.eng_df[self.config.features].notna().mean().to_string()}")
        # Engineering and fixing features:
        if self.config.use_engineered_and_extra_features:
            for ftr in ['StreopsisBefore_StereoSC', 'StreopsisBefore_StereoCC']:
                self.eng_df[ftr] = np.where(self.eng_df[ftr].eq(0), 6000, self.eng_df[ftr])
            fusion_strings = [
                "fragile fusion",
                "fusion",
            ]
            for ftr in ['StreopsisBefore_DWorth4dotCC', 'StreopsisBefore_NWorth4dotCC']:
                self.eng_df[ftr + '_fusion'] = self.eng_df[ftr].where(self.eng_df[ftr].isna(), self.eng_df[ftr].str.lower().isin(fusion_strings))
            self.eng_df["Dcc_LogMar_eyes_difference"] = (self.eng_df["DCVABefore_DccRE_LogMar"] - self.eng_df["DCVABefore_DccLE_LogMar"]).fillna(self.eng_df["D_SC_VABefore_DscLE_LogMar"] - self.eng_df["D_SC_VABefore_DscRE_LogMar"])
            self.eng_df['is_both_eyes'] = (self.eng_df[['RE_MR_num_mm', 'RE_LR_num_mm']].ne(0).any(axis=1) & self.eng_df[['LE_MR_num_mm', 'LE_LR_num_mm']].ne(0).any(axis=1)).astype(float)


        self.features_to_use = list(
            self.eng_df[self.config.non_deviation_features].notna().mean().to_frame('not_na')
            .query(f'not_na >= {self.config.feature_present_threshold}').index
        ) + [self.SC_CC, self.D_N]
        logging.info(f"Dropping features due to missing values: {set(self.config.non_deviation_features) - set(self.features_to_use)}")
        self.eng_df = (
            self.eng_df
            .assign(**{self.SC_CC: np.nan, self.D_N: np.nan})
            [['Code'] + self.config.deviation_features + self.features_to_use + self.config.raw_targets + self.config.extra_columns]
            .copy()
        )

        # Calculate target as DeviationAfter - DeviationBefore
        for target in self.config.targets:
            assert target not in self.eng_df
            self.eng_df[target] = self.eng_df[target.replace('Diff', 'After_Resolved')] - self.eng_df[target.replace('Diff', 'Before_Resolved')]

        # Split the data into 4 targets: Dsc, Nsc, Dcc, Ncc
        self.eng_df = pd.concat([
            self.eng_df[self.eng_df[target].notna()]
            .assign(**{
                **{c: np.nan for c in set(self.config.targets) - {target}},
                self.SC_CC: sc_cc, self.D_N: d_n,
                self.config.final_target: lambda df: df[target],
                self.config.final_before: lambda df: df[target.replace('Diff', 'Before_Resolved')],
            })
            for target, sc_cc, d_n in (
                ('DeviationsDiff_Dsc', 0, 0),
                ('DeviationsDiff_Nsc', 0, 1),
                ('DeviationsDiff_Dcc', 1, 0),
                ('DeviationsDiff_Ncc', 1, 1),
            )
        ]).reset_index(drop=True)
        rows_per_target = self.eng_df.groupby([self.SC_CC, self.D_N], dropna=False).size().reset_index()
        logging.info(f"Separated rows per target:\n{rows_per_target.to_string()}\nNew total is {len(self.eng_df)} rows.")

        # Filling missing values by copying features from one eye to the other
        # TODO: What about the BE values? Can they also fill or get filled by the others?
        for prefix, pair in (
            ('RefractionBefore_', ('LE_SE', 'RE_SE')),
            ('DCVABefore_Dcc', ('LE_LogMar', 'RE_LogMar')),
            ('D_SC_VABefore_Dsc', ('LE_LogMar', 'RE_LogMar')),
        ):
            for source, target in (pair, reversed(pair)):
                source_col = f'{prefix}{source}'
                target_col = f'{prefix}{target}'
                logging.info(f"Filling {(self.eng_df[target_col].isna() & self.eng_df[source_col].notna()).sum()} missing values for {target_col} with {source_col}")
                self.eng_df[target_col] = self.eng_df[target_col].fillna(self.eng_df[source_col])

        # Force L-R symmetry in model, by creating BE surgery features which are the sum of LE and RE features
        self.symmetric_features_to_use = [
            (f.replace('LE', 'BE') if f in self.L_R_MAPPING.keys() else f)
            for f in self.features_to_use
            if (f not in self.L_R_MAPPING.values())
        ]
        self.symmetric_surgery_features = [
            f for f in self.symmetric_features_to_use
            if f.endswith('_num_mm')
        ]
        for le_feature, re_feature in self.L_R_MAPPING.items():
            be_feature = le_feature.replace('LE', 'BE')
            assert be_feature not in self.eng_df
            self.eng_df[be_feature] = self.eng_df[le_feature] + self.eng_df[re_feature]

        self.one_hot_encoded_missing_features = []
        for feature in self.symmetric_features_to_use:
            num_to_fill = self.eng_df[feature].isna().sum()
            if num_to_fill > 0:
                # TODO: Alternative approach: impute with a linear regression model based on other features.
                if self.config.one_hot_encode_missing_feature:
                    # TODO: Could there be a "Missing Not At Random" issue here?
                    feature_missing_col = feature + '_missing'
                    logging.info(f"{feature}: filling {num_to_fill} missing values with 0, and creating {feature_missing_col} one-hot encoded feature")
                    self.eng_df[feature_missing_col] = self.eng_df[feature].isna().astype(int)
                    self.one_hot_encoded_missing_features.append(feature_missing_col)
                    self.eng_df[feature] = self.eng_df[feature].fillna(0)
                if self.config.impute_missing_feature:
                    fill_value = self.eng_df[feature].mean()
                    logging.info(f"{feature}: filling {num_to_fill} missing values with the mean {fill_value}")
                    self.eng_df[feature] = self.eng_df[feature].fillna(fill_value)

        self.symmetric_features_to_use.extend(self.one_hot_encoded_missing_features)

        if self.config.fix_num_mm_sign:
            row_filter = self.eng_df['BE_MR_num_mm'] > 0
            columns_to_invert = self.config.surgery_features + self.symmetric_surgery_features + [self.config.final_before, self.config.final_target]
            self.eng_df.loc[row_filter, columns_to_invert] = -self.eng_df.loc[row_filter, columns_to_invert]

        self.eng_df = self.eng_df[
            ['Code'] + self.config.deviation_features + list(chain(*self.L_R_MAPPING.items()))
            + self.symmetric_features_to_use + self.config.raw_targets + self.config.targets + [self.config.final_before, self.config.final_target]
             + self.config.extra_columns
        ].copy()
        self.eng_df[MONO_BINO] = self.eng_df[self.config.surgery_features].apply(
            lambda row: {1: MONO, 2: BINO}[len(set(col[:2] for col, val in row.items() if val != 0))],
            axis=1,
        )

        for name, baseline in BASELINES.items():
            self.eng_df[self.config.baseline_prefix + name] = baseline.generate_prediction(self.eng_df)

        logging.info(f"Engineered dataframe shape: {self.eng_df.shape}")

    def keep_one_date_per_group(self, days_diff: int = 180, groupby_cols: list = ['Code', 'd_n', 'sc_cc']) -> None:
        logging.info(f"Removing Multiple dates per: {groupby_cols}")
        logging.info(f"Prefer closest to {days_diff} days after the surgery")
        logging.info(f"Shape before {self.eng_df.shape}")
        closest_row = self.eng_df.groupby(groupby_cols)['DeviationsAfter_DaysDifference'].apply(lambda x: (x - days_diff).abs().idxmin())
        self.eng_df = self.eng_df.loc[closest_row]
        logging.info(f"Shape after {self.eng_df.shape}")

    def correlations(self, df: pd.DataFrame = None, columns: list[str] | dict[str, list[str]] = None, threshold: float = 0.1, sort=True) -> pd.DataFrame:
        if columns is None:
            columns = self.config.features
        if isinstance(columns, list):
            columns = {'feature1': columns, 'feature2': columns}
        assert len(columns) == 2
        name_1, name_2 = columns.keys()
        if df is None:
            df = self.preprocessed_df
        columns_for_correlation = {
            name: list(df[column_list].notna().mean().to_frame('not_na').query(f'not_na >= {threshold}').index)
            for name, column_list in columns.items()
        }

        corr_list = []
        for col_1 in columns_for_correlation[name_1]:
            for col_2 in columns_for_correlation[name_2]:
                if col_1 == col_2:
                    continue
                if (df.loc[df[col_1].notna(), col_2].nunique() == 1) or (df.loc[df[col_2].notna(), col_1].nunique() == 1):
                    continue
                corr_list.append({
                    name_1: col_1,
                    name_2: col_2,
                    'correlation': (df[col_1]).corr(df[col_2])
                })
        corr_df = pd.DataFrame(corr_list)
        if sort:
            corr_df.sort_values('correlation', key=abs, ascending=False, inplace=True)
        return corr_df

    def hist_subplots(self, columns: tuple[str, ...]) -> None:
        subplots(
            columns,
            data_generator=lambda feature: self.preprocessed_df[feature],
            plot_generator=lambda feature, f_series, ax: f_series.plot.hist(bins=10, ax=ax, density=True, color='teal'),
            title_generator=lambda feature, f_series: feature +
                                                      f'\npresent: {f_series.notna().mean():.1%};\n'
                                                      f'avg: {float(f_series.mean()):.1f}; std: {float(f_series.std()):.1f}\n'
                                                      f'min: {float(f_series.min()):.1f}; max: {float(f_series.max()):.1f}'
        )

    def train_test_split(self) -> None:
        # Maybe stratify by missing targets? Though the stats turn out alright anyway.
        surgery_codes = list(set(self.eng_df['Code']))

        train_codes, test_codes = train_test_split(
            surgery_codes, test_size=self.config.test_fraction,
            random_state=self.config.random_state,
            shuffle=True,  # Rows are approximately ordered by date, and we don't want that bias.
        )

        self.train_df = self.eng_df[self.eng_df['Code'].isin(train_codes)].copy()
        self.test_df = self.eng_df[self.eng_df['Code'].isin(test_codes)].copy()

        # Even though we deliberately shuffled the data, it's convenient to re-sort it after the split.
        self.train_df.sort_index(inplace=True)
        self.test_df.sort_index(inplace=True)
        logging.info(f"Train dataframe shape: {self.train_df.shape}")
        logging.info(f"Test dataframe shape: {self.test_df.shape}")

        # TODO: Feature scaling and centering?
        # TODO: Probably for num_mm features do not center

    def get_regression_formula(self) -> str:
        if self.config.only_interactions_with_surgery_features:
            formula = self.config.final_target + ' ~ ' + ' + '.join(
                self.symmetric_surgery_features + [
                    f'{surgery_feature}:{other_feature}'
                    for surgery_feature in self.symmetric_surgery_features
                    for other_feature in (set(self.symmetric_features_to_use) - set(self.symmetric_surgery_features))
                ]
            )
        else:
            formula = self.config.final_target + ' ~ ' + ' + '.join(self.symmetric_features_to_use)
        if not self.config.with_intercept:
            formula += ' -1'
        return formula

    def train_model_mse(self) -> None:
        self.model = smf.ols(
            formula=self.get_regression_formula(),
            data=self.train_df,
        ).fit()
        # TODO: To use regularization we need feature scaling
        # .fit_regularized(alpha=0.1)

    def train_model_mae(self) -> None:
        self.model = smf.quantreg(
            formula=self.get_regression_formula(),
            data=self.train_df,
        ).fit(q=0.5, max_iter=10_000)

    def train_several_models(self) -> None:
        self.models = {
            'Linear Regression': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression()),
            ]),
            'Ridge Regression': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=1.0)),
            ]),
            'Lasso Regression': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('regressor', Lasso(alpha=1.0)),
            ]),
            'Elastic Net': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('regressor', ElasticNet(alpha=1.0, l1_ratio=0.5)),
            ]),
            'Decision Tree': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('regressor', DecisionTreeRegressor(random_state=42)),
            ]),
            'Random Forest': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('regressor', RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.random_state,
                ))
            ]),
            'Gradient Boosting': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=self.config.random_state,
                ))
            ]),
            'Support Vector Regression': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('regressor', SVR(kernel='rbf')),
            ])
        }
        for name, model in self.models.items():
            # Fit the model
            model.fit(self.train_df[self.symmetric_features_to_use], self.train_df[self.config.final_target])

    def train_model_kfold(self) -> None:
        kfold = KFold(n_splits=self.config.kfold_splits, shuffle=True, random_state=self.config.random_state)
        surgery_codes = self.eng_df['Code'].unique()
        all_tests = []

        for fold, (train_codes_idx, test_codes_idx) in enumerate(kfold.split(surgery_codes)):
            self.train_df = self.eng_df[self.eng_df['Code'].isin(surgery_codes[train_codes_idx])].sort_index()
            test_df = self.eng_df[self.eng_df['Code'].isin(surgery_codes[test_codes_idx])].sort_index()

            if self.config.train_several_models:
                self.train_several_models()
                self._predict_several_models(test_df)
            else:
                self.train_model_mae()
                self._predict(test_df)
            all_tests.append(test_df.assign(fold=fold))

        self.train_df = self.eng_df
        self.test_df = pd.concat(all_tests)
        if self.config.train_several_models:
            self.train_several_models()
            self._predict_several_models(self.train_df)
        else:
            self.train_model_mae()
            self._predict(self.train_df)

    def get_std_for_term(self, term: str) -> pd.Series:
        if term == 'Intercept':
            return pd.Series()
        if ':' in term:
            f1, f2 = term.split(':')
            term_series = self.train_df[f1] * self.train_df[f2]
        else:
            term_series = self.train_df[term]
        return term_series

    def coefficient_df(self) -> pd.DataFrame:
        # TODO: Calculate correct effective importance of features that have an associated one-hot encoded feature for missing values.
        coefficient_df = self.model.params.to_frame('coefficient').join(self.model.pvalues.to_frame('p-value'))
        feature_std = coefficient_df.index.to_series().apply(lambda term: self.get_std_for_term(term).std())
        coefficient_df['importance'] = abs(coefficient_df['coefficient'] * feature_std)
        # TODO: The below is a work in progress
        # if self.config.one_hot_encode_missing_feature:
        #     for feature_missing_col in self.one_hot_encoded_missing_features:
        #         feature = feature_missing_col.replace('_missing', '')

        coefficient_df.sort_values('importance', ascending=False, inplace=True)
        return coefficient_df

    def success_failure_counts(self, errors: pd.Series) -> list[int]:
        return list((abs(errors) <= self.config.success_threshold).value_counts().sort_index(ascending=False))

    def model_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        baseline_values = {
            'Model': df[self.config.prediction],
            'Reality': -df[self.config.final_before],
            'Naive': df[self.config.final_target].mean(),
            **{
                baseline_name: df[self.config.baseline_prefix + baseline_name]
                for baseline_name in BASELINES.keys()
            },
        }

        results = {}

        for name, baseline_value in baseline_values.items():
            baseline_absolute_error = abs(df[self.config.final_target] - baseline_value)
            baseline_mae = baseline_absolute_error.mean()
            # TODO: Is it right to use the t-test? Maybe a sign test instead?
            mae_diff_from_baseline_p_value = np.nan if name in ('Model', 'Reality') else ttest_rel(
                baseline_absolute_error,
                abs(df[self.config.final_target] - baseline_values['Model']),
                alternative='greater',
            ).pvalue
            baseline_success_rate = (baseline_absolute_error <= self.config.success_threshold).mean()
            # TODO: Maybe Fisher is not the correct test because the data is paired.
            #  A test designed for paired data is McNemar's test, but I could only find a two-sided version of it, while we need a one-sided version.
            success_rate_diff_from_baseline_p_value = np.nan if name in ('Model', 'Reality') else fisher_exact(
                [
                    self.success_failure_counts(df[self.config.final_target] - baseline_value),
                    self.success_failure_counts(df[self.config.final_target] - baseline_values['Model']),
                ],
                alternative='less',
            ).pvalue
            results[name] = {
                'MAE': baseline_mae,
                'MAE p-value': mae_diff_from_baseline_p_value,
                'Success rate': baseline_success_rate,
                'Success rate p-value': success_rate_diff_from_baseline_p_value,
            }

        return pd.DataFrame(results).T

    def several_model_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        baseline_values = {
            **{
                name: df[self.config.prediction + name]
                for name in self.models.keys()
            },
            'Reality': -df[self.config.final_before],
            'Naive': df[self.config.final_target].mean(),
            **{
                baseline_name: df[self.config.baseline_prefix + baseline_name]
                for baseline_name in BASELINES.keys()
            },
        }

        results = {}
        for name, baseline_value in baseline_values.items():
            baseline_absolute_error = abs(df[self.config.final_target] - baseline_value)
            baseline_mae = baseline_absolute_error.mean()
            # TODO: Is it right to use the t-test? Maybe a sign test instead?
            baseline_success_rate = (baseline_absolute_error <= self.config.success_threshold).mean()
            # TODO: Maybe Fisher is not the correct test because the data is paired.
            #  A test designed for paired data is McNemar's test, but I could only find a two-sided version of it, while we need a one-sided version.
            results[name] = {
                'MAE': baseline_mae,
                'Success rate': baseline_success_rate,
            }

        return pd.DataFrame(results).T

    def calibration_plot(self, name: str, df: pd.DataFrame) -> None:
        line_lims = [-40, 40]
        ax = df.plot.scatter(self.config.final_target, self.config.prediction, s=2, color='teal')
        ax.set_title(name)
        ax.plot(line_lims, line_lims, '--r')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')

    # noinspection PyTypeChecker
    def _predict(self, df: pd.DataFrame) -> None:
        predictor = Predictor(model=self.model)

        df[self.config.prediction] = self.model.predict(df[self.symmetric_features_to_use]).values

        reproduce_num_mm_pred = df.apply(
            lambda row: pd.Series(asdict(predictor.get_num_mm(
                row[list(set(self.symmetric_features_to_use) - set(self.symmetric_surgery_features))],
                row[self.config.prediction],
            ))), axis=1,
        ).rename(columns=lambda col: 'Reproduce_Pred_' + col)
        df[reproduce_num_mm_pred.columns] = reproduce_num_mm_pred

        num_mm_pred = df.apply(
            lambda row: pd.Series(asdict(predictor.get_num_mm(
                row[list(set(self.symmetric_features_to_use) - set(self.symmetric_surgery_features))],
                -row[self.config.final_before],
            ))), axis=1,
        ).rename(columns=lambda col: 'Pred_' + col)
        df[num_mm_pred.columns] = num_mm_pred

    # noinspection PyTypeChecker
    def _predict_several_models(self, df: pd.DataFrame) -> None:
        for name, model in self.models.items():
            df[self.config.prediction + name] = model.predict(df[self.symmetric_features_to_use])


    def predictions(self) -> None:
        self._predict(self.train_df)
        self._predict(self.test_df)

    def top_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([
            df['Code'],
            df[self.SC_CC],
            df[self.D_N],
            df[self.config.final_target],
            df[self.config.prediction],
            abs(df[self.config.final_target] - df[self.config.prediction]).rename('Error'),
        ], axis=1).sort_values('Error', ascending=False)

    def num_mm_deviation_plot(self, df: pd.DataFrame, surgery_feature: str) -> None:
        assert surgery_feature in self.symmetric_surgery_features
        muscle = surgery_feature[3:5]
        other_surgery_feature = (set(self.symmetric_surgery_features) - {surgery_feature}).pop()
        le_surgery_feature = surgery_feature.replace('BE', 'LE')
        re_surgery_feature = surgery_feature.replace('BE', 'RE')
        df_to_plot = df[
            (df[other_surgery_feature] == 0)
            & (df[le_surgery_feature] == df[re_surgery_feature])
        ].copy()
        df_to_plot['abs_num_mm'] = abs(df_to_plot[le_surgery_feature])
        df_to_plot['sign_fixed_target'] = np.sign(df_to_plot[le_surgery_feature]) * df_to_plot[self.config.final_target]
        ax = df_to_plot.plot.scatter('abs_num_mm', 'sign_fixed_target', s=2, color='teal')

        ax.set_title(f'{muscle} ({len(df_to_plot)} rows)')

        ax.set_xlabel('Surgery (mm)')
        ax.set_ylabel('Deviation difference')

    def prescribed_num_mm_deviation_plot(self, df: pd.DataFrame, surgery_feature: str) -> None:
        assert surgery_feature in self.symmetric_surgery_features
        muscle = surgery_feature[3:5]
        other_surgery_feature = (set(self.symmetric_surgery_features) - {surgery_feature}).pop()
        le_surgery_feature = surgery_feature.replace('BE', 'LE')
        re_surgery_feature = surgery_feature.replace('BE', 'RE')
        df_to_plot = df[
            (df[other_surgery_feature] == 0)
            & (df[le_surgery_feature] == df[re_surgery_feature])
        ].copy()
        df_to_plot['num_mm_fix'] = df[f'Pred_{muscle}_both_eyes'] - df[le_surgery_feature]
        df_to_plot['DeviationAfter'] = df[self.config.final_before] + df[self.config.final_target]

        ax = df_to_plot.plot.scatter('num_mm_fix', 'DeviationAfter', s=2, color='teal')

        ax.set_title(f'{muscle} ({len(df_to_plot)} rows)')

        ax.set_xlabel('Prescribed modification (mm)')
        ax.set_ylabel('Existing deviation after')

    def run_everything(self, no_training: bool = False) -> None:
        logging.info("Filtering")
        self.filter_df()
        logging.info("Preprocessing")
        self.preprocess_df()
        logging.info("Feature engineering")
        self.feature_engineering()
        if self.config.remove_rows_with_multiple_dates:
            self.keep_one_date_per_group()
        if not self.config.use_kfold:
            logging.info("Train-test split")
            self.train_test_split()
        if not no_training:
            if self.config.use_kfold:
                logging.info("K-fold training")
                self.train_model_kfold()
            else:
                logging.info("Training model")
                self.train_model_mae()
                logging.info("Predictions")
                self.predictions()
        logging.info("Done")
