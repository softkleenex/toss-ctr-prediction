# -*- coding: utf-8 -*-
"""
Main Training Pipeline
전체 학습 파이프라인 통합 실행
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 모듈들
from preprocessing import load_and_preprocess
from feature_engineering import FeatureEngineer, get_feature_groups
from ensemble import LightGBMEnsemble, XGBoostModel, create_weighted_ensemble
from calibration import CalibrationStrategy, EnsembleCalibrator
from utils import (save_submission, print_memory_usage, seed_everything,
                   timer_decorator, get_feature_importance)
from config import *


@timer_decorator
def main_pipeline(use_advanced_features: bool = True,
                 calibration_method: str = 'refined',
                 save_output: bool = True):
    """
    메인 학습 파이프라인

    Args:
        use_advanced_features: 고급 피처 엔지니어링 사용 여부
        calibration_method: 보정 방법
        save_output: 결과 저장 여부
    """
    print("=" * 70)
    print("TOSS CTR PREDICTION - MAIN TRAINING PIPELINE")
    print("=" * 70)

    # 시드 설정
    seed_everything(42)

    print_memory_usage("Initial")

    # ===== 1. Data Loading & Preprocessing =====
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 70)

    X_train, y_train, X_test, test_ids = load_and_preprocess(
        train_path=str(TRAIN_PATH),
        test_path=str(TEST_PATH),
        features=BASE_FEATURES,
        sample_strategy='1:1',
        preprocessing_mode='advanced'
    )

    print_memory_usage("After Preprocessing")

    # ===== 2. Feature Engineering =====
    if use_advanced_features:
        print("\n" + "=" * 70)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 70)

        engineer = FeatureEngineer()

        # Train set
        train_df = pd.DataFrame(X_train, columns=BASE_FEATURES)
        train_df = engineer.engineer_all_features(train_df, include_advanced=True)

        # Test set
        test_df = pd.DataFrame(X_test, columns=BASE_FEATURES)
        test_df = engineer.engineer_all_features(test_df, include_advanced=True)

        X_train = train_df
        X_test = test_df

        print(f"\nFinal feature count: {len(X_train.columns)}")
        print_memory_usage("After Feature Engineering")

    # ===== 3. Model Training (Ensemble) =====
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)

    # LightGBM Ensemble
    lgb_ensemble = LightGBMEnsemble(n_seeds=5, n_folds=5)
    oof_preds = lgb_ensemble.train_ensemble(
        X_train, y_train,
        params=LIGHTGBM_PARAMS,
        num_boost_round=NUM_BOOST_ROUND
    )

    # Feature Importance
    if save_output:
        fi_df = get_feature_importance(
            lgb_ensemble.models[0],
            X_train.columns.tolist(),
            top_n=30
        )

    print_memory_usage("After Training")

    # ===== 4. Prediction =====
    print("\n" + "=" * 70)
    print("STEP 4: PREDICTION")
    print("=" * 70)

    test_predictions = lgb_ensemble.predict_ensemble(X_test)

    # ===== 5. Calibration =====
    print("\n" + "=" * 70)
    print("STEP 5: CALIBRATION")
    print("=" * 70)

    calibrator = CalibrationStrategy()
    final_predictions = calibrator.calibrate(
        test_predictions,
        method=calibration_method
    )

    # ===== 6. Save Submission =====
    if save_output:
        print("\n" + "=" * 70)
        print("STEP 6: SAVE SUBMISSION")
        print("=" * 70)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'submission_{calibration_method}_{timestamp}.csv'

        save_submission(
            final_predictions,
            test_ids,
            filename=filename
        )

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Advanced Features: {use_advanced_features}")
    print(f"  Calibration: {calibration_method}")
    print(f"  Models Trained: {len(lgb_ensemble.models)}")

    print(f"\nPrediction Summary:")
    print(f"  Mean: {final_predictions.mean():.6f}")
    print(f"  Std: {final_predictions.std():.6f}")
    print(f"  Range: [{final_predictions.min():.6f}, {final_predictions.max():.6f}]")

    print(f"\nTarget Score: {TARGET_SCORE}")
    print(f"Best Historical Score: {BEST_SCORES['supreme_evolved_refined']}")

    print_memory_usage("Final")

    return {
        'predictions': final_predictions,
        'test_ids': test_ids,
        'models': lgb_ensemble.models,
        'oof_scores': lgb_ensemble.scores
    }


def quick_train_pipeline():
    """빠른 학습 파이프라인 (기본 피처만)"""
    print("=" * 70)
    print("QUICK TRAINING PIPELINE (Basic Features)")
    print("=" * 70)

    return main_pipeline(
        use_advanced_features=False,
        calibration_method='refined',
        save_output=True
    )


def full_train_pipeline():
    """전체 학습 파이프라인 (모든 피처)"""
    print("=" * 70)
    print("FULL TRAINING PIPELINE (All Features)")
    print("=" * 70)

    return main_pipeline(
        use_advanced_features=True,
        calibration_method='refined',
        save_output=True
    )


def experiment_calibration_methods():
    """여러 Calibration 방법 실험"""
    print("=" * 70)
    print("CALIBRATION METHODS EXPERIMENT")
    print("=" * 70)

    # 기본 학습
    result = main_pipeline(
        use_advanced_features=True,
        calibration_method='refined',
        save_output=False
    )

    predictions = result['predictions']
    test_ids = result['test_ids']

    # 모든 Calibration 방법 시도
    calibrator = CalibrationStrategy()

    for method in ['supreme_optimal', 'refined', 'aggressive',
                   'quantile', 'power', 'sigmoid']:
        print(f"\n{'='*70}")
        print(f"Testing: {method}")
        print('='*70)

        calibrated = calibrator.calibrate(predictions, method=method)

        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'experiment_{method}_{timestamp}.csv'
        save_submission(calibrated, test_ids, filename=filename)


if __name__ == "__main__":
    import sys

    # Command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == 'quick':
            quick_train_pipeline()
        elif mode == 'full':
            full_train_pipeline()
        elif mode == 'experiment':
            experiment_calibration_methods()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: quick, full, experiment")
    else:
        # 기본: full pipeline
        full_train_pipeline()
