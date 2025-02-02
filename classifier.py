from typing import Dict, List, Tuple, Optional
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score


class ClassifierConfig:
    pos_label_threshold: float = 10.0
    kfold_splits: int = 5
    random_state: int = 42
    

    features: tuple[str, ...] = (
        'RE_MR_num_mm', 'LE_MR_num_mm', 'RE_LR_num_mm', 'LE_LR_num_mm',
        'MinN',
        'Age_Years',
        'DeviationBefore',
        # 'sc_cc',
        # 'd_n',
        # 'RefractionBefore_BE_SE'
    )

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
    final_before: str = 'DeviationBefore'
    final_target: str = 'DeviationDiff'
    id_col: str = 'Code'


class OptikaClassifier:
    # train = pd.DataFrame
    # test = pd.DataFrame

    def __init__(self, config: ClassifierConfig):
        self.config = config

    def get_X_Y(self, df):
        X = df[list(self.config.features)].values
        y = ((df[self.config.final_before] + df[
            self.config.final_target]).abs() <= self.config.pos_label_threshold).astype(int)
        return X, y

    def train_binary_classifier(self, df):
        X_train, y_train = self.get_X_Y(df)
        # model = LogisticRegression()
        model = RandomForestClassifier(random_state=42)  # (n_estimators=50, min_samples_split=10, random_state=42)
        model.fit(X_train, y_train)
        self.model = model

    def train_binary_classifier_cv(self, df, k_folds=None):
        if k_folds is None:
            k_folds = self.config.kfold_splits
        

        # Initialize the model
        model = RandomForestClassifier(random_state=self.config.random_state, n_estimators=50, min_samples_split=10)

        # Initialize StratifiedKFold for splitting the data
        # skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        # kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.config.random_state)
        # surgery_codes = df['Code'].unique()

        # # Build label for stratification
        # df4folds = df[["Code", self.config.final_before, self.config.final_target]].copy()
        # df4folds["label"] = ((df4folds[self.config.final_before] + df4folds[
        #     self.config.final_target]).abs() <= self.config.pos_label_threshold).astype(int)
        
        # # Find majority "category" for each "code"
        # grouped = df4folds.groupby("Code")["label"].max()
        # grouped_df = grouped.reset_index().rename(columns={"label": "stratify_label"})

        # Step 2: Create GroupKFold splits
        group_kfold = GroupKFold(n_splits=k_folds)

        # List to store scores for each fold
        # fold_scores = []
        fold_proba = []  # To store predicted probabilities
        fold_truth = []  # To store true labels

        # for train_index, test_index in skf.split(x, y):
        # for fold, (train_codes_idx, test_codes_idx) in enumerate(kfold.split(surgery_codes, y)):
        # for fold, (train_codes_idx, test_codes_idx) in enumerate(group_kfold.split(grouped_df, grouped_df["stratify_label"], groups=grouped_df["Code"])):
        #     train_codes = grouped_df.iloc[train_codes_idx]["Code"]
        #     test_codes = grouped_df.iloc[test_codes_idx]["Code"]
        for fold, (train_codes_idx, test_codes_idx) in enumerate(group_kfold.split(df, groups=df["Code"])):
            train_codes = df.iloc[train_codes_idx]["Code"]
            test_codes = df.iloc[test_codes_idx]["Code"]
            
            # Split the data into training and test sets
            train_df = df[df['Code'].isin(train_codes)].sort_index()
            test_df = df[df['Code'].isin(test_codes)].sort_index()

            # Get x, y
            X_train, y_train = self.get_X_Y(train_df)
            X_test, y_test = self.get_X_Y(test_df)

            # Convert x, y to pd.DataFrames
            x_train = pd.DataFrame(X_train).reset_index(drop=True)
            y_train = pd.DataFrame(y_train).reset_index(drop=True).values
            x_test = pd.DataFrame(X_test).reset_index(drop=True)
            y_test = pd.DataFrame(y_test).reset_index(drop=True).values

            # x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            # y_train, y_test = y[train_index], y[test_index]  # Use numpy-style indexing for y

            # Train the model on the training data
            model.fit(x_train, y_train)

            # Get predicted probabilities for the test data
            probas = model.predict_proba(x_test)[:, 1]  # Get probability of the positive class

            # Append the probabilities and true labels for later evaluation
            fold_proba.extend(probas)
            fold_truth.extend(y_test)

            # Evaluate the model on the test data (accuracy in this case)
            # fold_scores.append(model.score(x_test, y_test))  # Or use other metrics like accuracy, etc.

        # Calculate the mean score across all folds
        # mean_score = np.mean(fold_scores)

        # Convert probabilities to binary predictions (0 or 1)
        predicted_labels = (np.array(fold_proba) >= 0.5).astype(int)

        # Calculate accuracy based on predicted labels and true labels
        accuracy = accuracy_score(fold_truth, predicted_labels)

        # After cross-validation, fit the model on the entire dataset
        self.x, self.y = self.get_X_Y(df)
        model.fit(self.x, self.y)  # Now fit the model on all data

        # Return both the trained model and the overall accuracy score
        self.model = model
        self.cv_accuracy = accuracy
        self.fold_truth = fold_truth
        self.fold_proba = fold_proba


    def save(self, filename):
        try:
            self.x = None
            self.y = None
        except:
            pass
        joblib.dump(self, filename)


