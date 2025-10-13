# -*- coding: utf-8 -*-
"""
Configuration Module
프로젝트 전역 설정
"""

from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODEL_DIR = PROJECT_ROOT / 'models'

TRAIN_PATH = DATA_DIR / 'train.parquet'
TEST_PATH = DATA_DIR / 'test.parquet'


# ============================================================
# Features
# ============================================================

BASE_FEATURES = [
    # User attributes
    'gender', 'age_group',

    # Ad attributes
    'inventory_id',

    # Temporal
    'day_of_week', 'hour',

    # Location features (5)
    'l_feat_1', 'l_feat_2', 'l_feat_3', 'l_feat_5', 'l_feat_10',

    # Behavioral features (7)
    'feat_a_1', 'feat_a_2', 'feat_a_3',
    'feat_b_1', 'feat_b_3',
    'feat_c_1', 'feat_c_8',

    # Historical features (5)
    'history_a_1', 'history_a_3',
    'history_b_1', 'history_b_21', 'history_b_30'
]

CATEGORICAL_FEATURES = [
    'gender', 'age_group', 'inventory_id',
    'day_of_week', 'hour'
]

NUMERICAL_FEATURES = [f for f in BASE_FEATURES if f not in CATEGORICAL_FEATURES]

# Feature groups for engineering
HISTORY_FEATURES = [f for f in BASE_FEATURES if 'history' in f]
FEAT_FEATURES = [f for f in BASE_FEATURES if 'feat' in f and 'l_feat' not in f]
L_FEAT_FEATURES = [f for f in BASE_FEATURES if 'l_feat' in f]


# ============================================================
# Model Parameters
# ============================================================

LIGHTGBM_PARAMS = {
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

XGBOOST_PARAMS = {
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


# ============================================================
# Training Configuration
# ============================================================

# Seeds for ensemble
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]

# Cross-validation
N_FOLDS = 5

# Early stopping
EARLY_STOPPING_ROUNDS = 100

# Training rounds
NUM_BOOST_ROUND = 2500


# ============================================================
# Sampling Strategy
# ============================================================

SAMPLING_STRATEGIES = {
    'balanced_1_1': {
        'ratio': '1:1',
        'description': 'Balanced sampling (Pos:Neg = 1:1)'
    },
    'imbalanced_1_1.5': {
        'ratio': '1:1.5',
        'description': 'Slightly imbalanced (Pos:Neg = 1:1.5)'
    },
    'large': {
        'ratio': 'large',
        'n_samples': 3500000,
        'description': 'Large sample with original ratio'
    }
}


# ============================================================
# Calibration Configuration
# ============================================================

CALIBRATION_METHODS = {
    'supreme_optimal': {
        'offset': 0.248,
        'scale': 0.504,
        'description': 'Best performing calibration'
    },
    'refined': {
        'offset': 0.2478,
        'scale': 0.5042,
        'micro_variation': True,
        'description': 'Refined with micro variation'
    },
    'aggressive': {
        'offset': 0.252,
        'scale': 0.496,
        'description': 'Aggressive calibration for higher target'
    }
}

DEFAULT_CALIBRATION = 'refined'


# ============================================================
# Ensemble Weights
# ============================================================

ENSEMBLE_WEIGHTS = {
    'balanced_1_1': 2.0,
    'imbalanced_1_1.5': 1.5,
    'large': 1.0
}


# ============================================================
# Experiment Tracking
# ============================================================

EXPERIMENT_LOG_FILE = OUTPUT_DIR / 'experiments.csv'
SUBMISSION_LOG_FILE = OUTPUT_DIR / 'submissions.csv'


# ============================================================
# Competition Info
# ============================================================

COMPETITION = {
    'name': 'Toss NEXT ML Challenge - CTR Prediction',
    'platform': 'Dacon',
    'metric': 'AUC',
    'target': 'clicked',
    'click_rate': 0.0191  # 1.91%
}


# ============================================================
# Best Scores (for reference)
# ============================================================

BEST_SCORES = {
    'supreme_evolved_refined': 0.3434805649,
    'final_push_v1': 0.3434775373,
    'enhanced_v2': 0.3425593061
}

TARGET_SCORE = 0.35


# ============================================================
# Helper Functions
# ============================================================

def get_config_summary():
    """설정 요약 출력"""
    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)

    print(f"\nCompetition: {COMPETITION['name']}")
    print(f"  Platform: {COMPETITION['platform']}")
    print(f"  Metric: {COMPETITION['metric']}")
    print(f"  Target Score: {TARGET_SCORE}")
    print(f"  Best Score: {BEST_SCORES['supreme_evolved_refined']}")

    print(f"\nFeatures:")
    print(f"  Base: {len(BASE_FEATURES)}")
    print(f"  Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"  Numerical: {len(NUMERICAL_FEATURES)}")

    print(f"\nTraining:")
    print(f"  Seeds: {len(ENSEMBLE_SEEDS)}")
    print(f"  Folds: {N_FOLDS}")
    print(f"  Total models: {len(ENSEMBLE_SEEDS) * N_FOLDS}")

    print(f"\nCalibration:")
    print(f"  Default method: {DEFAULT_CALIBRATION}")
    print(f"  Available methods: {len(CALIBRATION_METHODS)}")

    print("=" * 60)


if __name__ == "__main__":
    get_config_summary()
