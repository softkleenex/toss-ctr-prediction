# -*- coding: utf-8 -*-
"""
Ensemble Module
다양한 앙상블 전략 및 모델 조합
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class LightGBMEnsemble:
    """LightGBM 앙상블 클래스"""

    def __init__(self, n_seeds: int = 5, n_folds: int = 5):
        self.n_seeds = n_seeds
        self.n_folds = n_folds
        self.models = []
        self.scores = []

    def get_default_params(self) -> dict:
        """기본 LightGBM 파라미터"""
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 200,
            'learning_rate': 0.012,
            'feature_fraction': 0.65,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'lambda_l1': 0.05,
            'lambda_l2': 0.05,
            'min_child_samples': 20,
            'verbose': -1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }

    def train_ensemble(self, X, y, params: dict = None,
                      num_boost_round: int = 2500) -> List[np.ndarray]:
        """
        앙상블 학습

        Returns:
            List of OOF predictions for each seed
        """
        if params is None:
            params = self.get_default_params()

        print(f"\n=== LightGBM Ensemble Training ===")
        print(f"Seeds: {self.n_seeds}, Folds: {self.n_folds}")
        print(f"Total models: {self.n_seeds * self.n_folds}")

        all_oof_preds = []

        for seed_idx, seed in enumerate(range(42, 42 + self.n_seeds)):
            print(f"\n--- Seed {seed} ({seed_idx+1}/{self.n_seeds}) ---")

            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                random_state=seed)

            oof_pred = np.zeros(len(X))

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_val],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100),
                        lgb.log_evaluation(0)
                    ]
                )

                oof_pred[val_idx] = model.predict(X_val)
                self.models.append(model)

                # AUC 계산
                from sklearn.metrics import roc_auc_score
                fold_auc = roc_auc_score(y_val, oof_pred[val_idx])

                print(f"  Fold {fold}: AUC={fold_auc:.6f}, "
                      f"Best iter={model.best_iteration}")

            # Seed의 전체 OOF AUC
            seed_auc = roc_auc_score(y, oof_pred)
            self.scores.append(seed_auc)
            print(f"Seed {seed} OOF AUC: {seed_auc:.6f}")

            all_oof_preds.append(oof_pred)

        # 전체 평균 AUC
        avg_auc = np.mean(self.scores)
        std_auc = np.std(self.scores)
        print(f"\n=== Ensemble Results ===")
        print(f"Average OOF AUC: {avg_auc:.6f} ± {std_auc:.6f}")
        print(f"Total models trained: {len(self.models)}")

        return all_oof_preds

    def predict_ensemble(self, X_test) -> np.ndarray:
        """앙상블 예측"""
        print(f"\nPredicting with {len(self.models)} models...")

        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X_test)
            predictions.append(pred)

            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{len(self.models)} models")

        # 평균
        ensemble_pred = np.mean(predictions, axis=0)

        print(f"\nEnsemble prediction:")
        print(f"  Mean: {ensemble_pred.mean():.6f}")
        print(f"  Std: {ensemble_pred.std():.6f}")

        return ensemble_pred


class XGBoostModel:
    """XGBoost 모델 (다양성 확보용)"""

    def __init__(self):
        self.model = None

    def get_default_params(self) -> dict:
        """기본 XGBoost 파라미터"""
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'gpu_hist',
            'max_depth': 7,
            'learning_rate': 0.01,
            'n_estimators': 2000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'random_state': 42
        }

    def train(self, X_train, y_train, X_val, y_val,
             params: dict = None) -> 'XGBoostModel':
        """XGBoost 학습"""
        if params is None:
            params = self.get_default_params()

        print("\n=== XGBoost Training ===")

        self.model = xgb.XGBClassifier(**params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )

        # Validation AUC
        from sklearn.metrics import roc_auc_score
        val_pred = self.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)

        print(f"Validation AUC: {val_auc:.6f}")
        print(f"Best iteration: {self.model.best_iteration}")

        return self

    def predict(self, X_test) -> np.ndarray:
        """예측"""
        return self.model.predict_proba(X_test)[:, 1]


class StackingEnsemble:
    """스태킹 앙상블 (미사용 - 실험용)"""

    def __init__(self, base_models: list, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or lgb.LGBMClassifier()

    def fit_predict(self, X_train, y_train, X_test):
        """스태킹 앙상블 학습 및 예측"""
        # Base model predictions
        base_preds_train = []
        base_preds_test = []

        for i, model in enumerate(self.base_models):
            print(f"\nBase model {i+1}/{len(self.base_models)}")

            # Train
            model.fit(X_train, y_train)

            # Predict
            pred_train = model.predict_proba(X_train)[:, 1]
            pred_test = model.predict_proba(X_test)[:, 1]

            base_preds_train.append(pred_train)
            base_preds_test.append(pred_test)

        # Stack predictions
        X_train_stack = np.column_stack(base_preds_train)
        X_test_stack = np.column_stack(base_preds_test)

        # Meta model
        print("\nMeta model training...")
        self.meta_model.fit(X_train_stack, y_train)

        final_pred = self.meta_model.predict_proba(X_test_stack)[:, 1]

        return final_pred


def create_weighted_ensemble(predictions_dict: Dict[str, np.ndarray],
                             weights: Dict[str, float] = None) -> np.ndarray:
    """
    가중 평균 앙상블

    Args:
        predictions_dict: {model_name: predictions}
        weights: {model_name: weight}

    Returns:
        Weighted ensemble predictions
    """
    if weights is None:
        # Equal weights
        weights = {name: 1.0 for name in predictions_dict.keys()}

    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}

    print("\n=== Weighted Ensemble ===")
    for name, weight in normalized_weights.items():
        print(f"  {name}: {weight:.4f}")

    # Weighted average
    ensemble = np.zeros_like(list(predictions_dict.values())[0])

    for name, pred in predictions_dict.items():
        ensemble += pred * normalized_weights[name]

    print(f"\nEnsemble:")
    print(f"  Mean: {ensemble.mean():.6f}")
    print(f"  Std: {ensemble.std():.6f}")

    return ensemble


if __name__ == "__main__":
    print("Ensemble Module")
    print("=" * 50)

    # 모듈 구성 설명
    print("\nAvailable ensemble strategies:")
    print("  1. LightGBM Multi-seed Multi-fold (25 models)")
    print("  2. XGBoost for diversity")
    print("  3. Weighted averaging")
    print("  4. Stacking (experimental)")
