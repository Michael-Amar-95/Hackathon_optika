from dataclasses import dataclass
from pickle import UnpicklingError
from typing import Literal, Dict, List, Tuple, Optional


# from classifier import Optimizer
import numpy as np
from statsmodels.iolib import load_pickle
import joblib

from scipy.optimize import minimize

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

@dataclass
class SurgeryPrescription:
    MR_both_eyes: float
    LR_both_eyes: float
    MR_single_eye: float
    LR_single_eye: float


class Predictor:
    def __init__(self, model_path: str=None, model=None):
        if model is None:
            try:
                self.model = load_pickle(model_path)
            except UnpicklingError:
                self.model = joblib.load(model_path)
        else:
            assert model_path is None, "model_path must be None if model is not None"
            self.model = model

    # Some of these can be missing
    def predict(
        self,
        Deviation: float,
        DCVA_Dcc_LogMar: float,
        D_SC_VA_Dsc_LogMar: float,
        Refraction_SE: float,
        Age_Years: float,
        Gender: float, # 1 = Female
        sc_cc: Literal[0, 1] = 1,  # CC
        d_n: Literal[0, 1] = 1,  # N
        days_to_measure: int = 365,
    ) -> SurgeryPrescription:
        data = {
            'RefractionBefore_BE_SE': 0 if np.isnan(Refraction_SE) else Refraction_SE,
            'RefractionBefore_BE_SE_missing': 1 if np.isnan(Refraction_SE) else 0,
            'D_SC_VABefore_DscBE_LogMar': 0 if np.isnan(D_SC_VA_Dsc_LogMar) else D_SC_VA_Dsc_LogMar,
            'D_SC_VABefore_DscBE_LogMar_missing': 1 if np.isnan(D_SC_VA_Dsc_LogMar) else 0,
            'DCVABefore_DccBE_LogMar': 0 if np.isnan(DCVA_Dcc_LogMar) else DCVA_Dcc_LogMar,
            'DCVABefore_DccBE_LogMar_missing': 1 if np.isnan(DCVA_Dcc_LogMar) else 0,
            'sc_cc': sc_cc,
            'd_n': d_n,
            'Age_Years': Age_Years,
            'MinN': Gender,
            'DeviationsAfter_DaysDifference': days_to_measure,
        }

        return self.get_num_mm(
            data, -Deviation,
            only_interactions_with_surgery_features=True,
            with_intercept=True,
        )

    def get_num_mm(
        self, data: dict[str, float], prediction: float,
        only_interactions_with_surgery_features: bool = True,
        with_intercept: bool = True,
    ) -> SurgeryPrescription:
        # Equation to inverse:
        # c = mr_coefficient * MR_prediction + lr_coefficient * LR_prediction

        mr_coefficient = self.model.params['BE_MR_num_mm']
        lr_coefficient = self.model.params['BE_LR_num_mm']
        c = prediction

        if only_interactions_with_surgery_features:
            mr_coefficient += sum(value * self.model.params[f'BE_MR_num_mm:{key}'] for key, value in data.items())
            lr_coefficient += sum(value * self.model.params[f'BE_LR_num_mm:{key}'] for key, value in data.items())
        else:
            c -= sum(value * self.model.params[key] for key, value in data.items())
        if with_intercept:
            c -= self.model.params['Intercept']

        norm_c = c / (mr_coefficient ** 2 + lr_coefficient ** 2)
        return SurgeryPrescription(
            # Solve for MR assuming LR is zero
            MR_both_eyes=c / mr_coefficient / 2,
            # Solve for LR assuming MR is zero
            LR_both_eyes=c / lr_coefficient / 2,
            # Solve for both, choosing the solution which minimizes the sum of squares
            # TODO: Is this the best way to choose the solution?
            MR_single_eye=mr_coefficient * norm_c,
            LR_single_eye=lr_coefficient * norm_c,
        )

    def predict_proba(
        self,
        Deviation: float,
        # DCVA_Dcc_LogMar: float,
        # D_SC_VA_Dsc_LogMar: float,
        # Refraction_SE: float,
        Age_Years: float,
        Gender: float, # 1 = Female
        # sc_cc: Literal[0, 1] = 1,  # CC
        # d_n: Literal[0, 1] = 1,  # N
        # days_to_measure: int = 365,
    ) -> SurgeryPrescription:
        optimized_features = self.model.config.surgery_features_

        optimizer = Optimizer(
            classifier=self.model.model, 
            all_features=self.model.config.features, 
            optimized_features=optimized_features,
            random_state=42
        )

        fixed_features = {
            'MinN': Gender,
            'Age_Years': Age_Years,
            'DeviationBefore': Deviation,
            # 'sc_cc': sc_cc,
            # 'd_n': d_n,
            # 'RefractionBefore_BE_SE': 0 if np.isnan(Refraction_SE) else Refraction_SE,
            }

        bounds = {
            "LR_both_eyes": [
            (np.float64(0.0), np.float64(0.0)),
            (np.float64(0.0), np.float64(0.0)),
            (np.float64(-8.0), np.float64(9.0)),
            (np.float64(-8.0), np.float64(9.0))
            ],
            "MR_both_eyes": [
            (np.float64(-6.5), np.float64(0.0)),
            (np.float64(-6.5), np.float64(0.0)),
            (np.float64(0.0), np.float64(0.0)),
            (np.float64(0.0), np.float64(0.0))
            ],
            "single_eye": [
            (np.float64(-6.5), np.float64(0.0)),
            (np.float64(0.0), np.float64(0.0)),
            (np.float64(-8.0), np.float64(9.0)),
            (np.float64(0.0), np.float64(0.0))
            ],
        } 

        # Optimize all 3 possibilities:
        optimal_surgery_features = None
        optimal_surgery_name = 'None'
        success_proba = 0.0
        for surgery in list(bounds.keys()):

            try:
                optimal_features, prediction = optimizer.optimize(
                    fixed_features=fixed_features,
                    bounds=bounds[surgery],
                    maximize=True  # Set to False if you want to minimize
                )
                
                print(f"{surgery} - Optimal Features:")
                for feature, value in optimal_features.items():
                    print(f"  {feature}: {value:.4f}")
                print(f"\n{surgery} - Classifier Prediction: {prediction:.4f}")
                if prediction >= success_proba:
                    success_proba = prediction
                    optimal_surgery_name = surgery
                    optimal_surgery_features = optimal_features
            except ValueError as e:
                print(f"{surgery} - Optimization Error: {e}")

        # Enforce symmetric solution and return
        print(f"Optimal surgery is '{optimal_surgery_name}'")
        if optimal_surgery_name == "single_eye":
            return SurgeryPrescription(
                    MR_both_eyes=np.float16(0.0),
                    LR_both_eyes=np.float16(0.0),
                    MR_single_eye=optimal_surgery_features["RE_MR_num_mm"],
                    LR_single_eye=optimal_surgery_features["RE_LR_num_mm"]            
                )
        elif optimal_surgery_name == "MR_both_eyes":
            return SurgeryPrescription(
                    MR_both_eyes=(optimal_surgery_features["RE_MR_num_mm"] + optimal_surgery_features["LE_MR_num_mm"]) / 2,
                    LR_both_eyes=np.float16(0.0),
                    MR_single_eye=np.float16(0.0),
                    LR_single_eye=np.float16(0.0)
                )
        elif optimal_surgery_name == "LR_both_eyes":
            return SurgeryPrescription(
                    MR_both_eyes=np.float16(0.0),
                    LR_both_eyes=(optimal_surgery_features["RE_LR_num_mm"] + optimal_surgery_features["LE_LR_num_mm"]) / 2,
                    MR_single_eye=np.float16(0.0),
                    LR_single_eye=np.float16(0.0)
                )

class Optimizer:
    """
    A class to optimize a subset of features to achieve the best classifier prediction.
    
    Attributes:
        classifier: A trained classifier with a predict_proba or predict method.
        all_features: List of all feature names used by the classifier.
        optimized_features: List of feature names to optimize.
        random_state: An integer seed for reproducibility.
    """
    
    def __init__(
        self, 
        classifier, 
        all_features: List[str], 
        optimized_features: List[str], 
        random_state: Optional[int] = None
    ):
        """
        Initializes the Optimizer.
        
        Args:
            classifier: A trained classifier with a predict_proba or predict method.
            all_features: List of all feature names used by the classifier.
            optimized_features: List of feature names to optimize.
            random_state: An integer seed for reproducibility.
        """
        self.classifier = classifier
        self.all_features = all_features
        self.optimized_features = optimized_features
        self.random_state = random_state
        
        # Validate that optimized_features are part of all_features
        for feature in self.optimized_features:
            if feature not in self.all_features:
                raise ValueError(f"Optimized feature '{feature}' is not in all_features.")
        
        # Determine if classifier has predict_proba or predict
        if hasattr(self.classifier, 'predict_proba'):
            self.has_predict_proba = True
        elif hasattr(self.classifier, 'predict'):
            self.has_predict_proba = False
        else:
            raise ValueError("Classifier must have either 'predict_proba' or 'predict' method.")
    
    def optimize(
        self, 
        fixed_features: Dict[str, float], 
        bounds: List[Tuple[float, float]], 
        maximize: bool = True, 
        initial_guess: Optional[List[float]] = None
    ) -> Tuple[Dict[str, float], float]:
        """
        Performs optimization to find the best combination of optimized features.
        
        Args:
            fixed_features: Dictionary with keys as feature names and values as fixed values.
            bounds: List of tuples specifying (min, max) for each optimized feature.
            maximize: Boolean indicating whether to maximize (True) or minimize (False) the prediction.
            initial_guess: Optional initial guess for the optimizer. If None, uses the midpoint of bounds.
        
        Returns:
            A tuple containing:
                - Dictionary of optimal feature values.
                - Classifier's prediction for the optimal features.
        
        Raises:
            ValueError: If inputs are invalid or optimization fails.
        """
        # Validate fixed_features
        if not isinstance(fixed_features, dict):
            raise ValueError("fixed_features must be a dictionary.")
        
        if len(fixed_features) != len(self.all_features) - len(self.optimized_features):
            raise ValueError(
                f"Number of fixed features ({len(fixed_features)}) does not match expected ({len(self.all_features) - len(self.optimized_features)})."
            )
        
        for feature in fixed_features:
            if feature not in self.all_features:
                raise ValueError(f"Fixed feature '{feature}' is not in all_features.")
            if feature in self.optimized_features:
                raise ValueError(f"Feature '{feature}' is marked as both fixed and optimized.")
        
        # Define the order of all features
        all_feature_order = self.all_features
        
        # Objective function: negative prediction if maximizing
        def objective(x):
            # Combine fixed and optimized features
            feature_values = []
            optimized_dict = dict(zip(self.optimized_features, x))
            for feature in all_feature_order:
                if feature in fixed_features:
                    feature_values.append(fixed_features[feature])
                else:
                    feature_values.append(optimized_dict[feature])
            feature_array = np.array(feature_values).reshape(1, -1)
            
            if self.has_predict_proba:
                # Assume we are interested in the probability of the positive class
                prediction = self.classifier.predict_proba(feature_array)[0][1]
            else:
                # If only predict is available, use it directly
                prediction = self.classifier.predict(feature_array)[0]
            
            return -prediction if maximize else prediction
        
        # Set initial guess
        if initial_guess is None:
            initial_guess = [ (b[0] + b[1])/2 for b in bounds ]
        else:
            if len(initial_guess) != len(self.optimized_features):
                raise ValueError("Initial guess length does not match number of optimized features.")
        
        # Set optimization options
        options = {
            'disp': False,
            'maxiter': 1000
        }
        
        # Perform optimization
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        # Extract optimal feature values
        optimal_values = result.x
        optimal_features = dict(zip(self.optimized_features, optimal_values))
        
        # Combine with fixed features for prediction
        combined_features = fixed_features.copy()
        combined_features.update(optimal_features)
        
        # Prepare feature array in the correct order
        final_feature_array = np.array([combined_features[feature] for feature in all_feature_order]).reshape(1, -1)
        
        if self.has_predict_proba:
            prediction = self.classifier.predict_proba(final_feature_array)[0][1]
        else:
            prediction = self.classifier.predict(final_feature_array)[0]
        
        return optimal_features, prediction