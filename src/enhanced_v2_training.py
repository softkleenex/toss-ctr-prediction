# -*- coding: utf-8 -*-
"""
ULTRATHINK ENHANCED V2 - 개선된 전처리 + 피처 엔지니어링
문제점 해결: 결측치, 상관관계, 피처 엔지니어링, 하이퍼파라미터
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ULTRATHINK ENHANCED V2 - 전처리 강화 버전")
print("="*70)

# Features
FEATURES = [
    'gender', 'age_group', 'inventory_id', 'day_of_week', 'hour',
    'l_feat_1', 'l_feat_2', 'l_feat_3', 'l_feat_5', 'l_feat_10',
    'feat_a_1', 'feat_a_2', 'feat_a_3', 'feat_b_1', 'feat_b_3',
    'feat_c_1', 'feat_c_8', 'history_a_1', 'history_a_3',
    'history_b_1', 'history_b_21', 'history_b_30'
]

print(f"\n[ 데이터 로딩 ]")
print(f"Features: {len(FEATURES)}")

try:
    train_df = pd.read_parquet('train.parquet', columns=FEATURES + ['clicked'])
    test_df = pd.read_parquet('test.parquet', columns=['ID'] + FEATURES)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Click rate: {train_df['clicked'].mean():.4f}")

    # ============================================================
    # 1. 고급 결측치 처리
    # ============================================================
    print(f"\n[ 1단계: 고급 결측치 처리 ]")

    for col in FEATURES:
        # Numeric 변환
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

        # 결측치 비율 계산
        missing_pct = train_df[col].isnull().mean()

        if missing_pct > 0:
            print(f"  {col}: {missing_pct*100:.2f}% 결측")

            # 카테고리형은 -999로, 연속형은 median으로
            if col in ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']:
                # 카테고리형 - 특별한 값으로 표시
                train_df[col].fillna(-999, inplace=True)
                test_df[col].fillna(-999, inplace=True)
            else:
                # 연속형 - median으로
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
        else:
            # 결측치 없어도 0으로 채우기
            train_df[col].fillna(0, inplace=True)
            test_df[col].fillna(0, inplace=True)

    # ============================================================
    # 2. 피처 엔지니어링
    # ============================================================
    print(f"\n[ 2단계: 피처 엔지니어링 ]")

    # 상호작용 피처
    train_df['age_gender_int'] = train_df['age_group'] * train_df['gender']
    test_df['age_gender_int'] = test_df['age_group'] * test_df['gender']

    train_df['hour_day_int'] = train_df['hour'] * train_df['day_of_week']
    test_df['hour_day_int'] = test_df['hour'] * test_df['day_of_week']

    # History 합계
    history_cols = ['history_a_1', 'history_a_3', 'history_b_1', 'history_b_21', 'history_b_30']
    train_df['history_sum'] = train_df[history_cols].sum(axis=1)
    test_df['history_sum'] = test_df[history_cols].sum(axis=1)

    train_df['history_mean'] = train_df[history_cols].mean(axis=1)
    test_df['history_mean'] = test_df[history_cols].mean(axis=1)

    # L_feat 합계
    l_feat_cols = ['l_feat_1', 'l_feat_2', 'l_feat_3', 'l_feat_5', 'l_feat_10']
    train_df['l_feat_sum'] = train_df[l_feat_cols].sum(axis=1)
    test_df['l_feat_sum'] = test_df[l_feat_cols].sum(axis=1)

    # Feat 합계
    feat_cols = ['feat_a_1', 'feat_a_2', 'feat_a_3', 'feat_b_1', 'feat_b_3', 'feat_c_1', 'feat_c_8']
    train_df['feat_sum'] = train_df[feat_cols].sum(axis=1)
    test_df['feat_sum'] = test_df[feat_cols].sum(axis=1)

    NEW_FEATURES = ['age_gender_int', 'hour_day_int', 'history_sum', 'history_mean',
                    'l_feat_sum', 'feat_sum']
    ALL_FEATURES = FEATURES + NEW_FEATURES

    print(f"  추가된 피처: {len(NEW_FEATURES)}개")
    print(f"  총 피처: {len(ALL_FEATURES)}개")

    # ============================================================
    # 3. 샘플링 - 더 많은 데이터 사용
    # ============================================================
    print(f"\n[ 3단계: 데이터 샘플링 ]")
    n_samples = 2000000  # 2M 샘플 (기존 1M에서 증가)

    pos = train_df[train_df['clicked'] == 1].sample(
        n=min(n_samples//2, len(train_df[train_df['clicked']==1])),
        random_state=42
    )
    neg = train_df[train_df['clicked'] == 0].sample(
        n=min(n_samples//2, len(train_df[train_df['clicked']==0])),
        random_state=42
    )
    train_sample = pd.concat([pos, neg]).sample(frac=1, random_state=42)

    print(f"  샘플 크기: {len(train_sample):,}")
    print(f"  Positive: {len(pos):,}, Negative: {len(neg):,}")

    X = train_sample[ALL_FEATURES].astype(np.float32)
    y = train_sample['clicked'].values
    test_X = test_df[ALL_FEATURES].astype(np.float32)
    test_id = test_df['ID'].values

    del train_df, test_df, train_sample, pos, neg
    gc.collect()

    # ============================================================
    # 4. 모델 훈련 - 개선된 파라미터
    # ============================================================
    print(f"\n[ 4단계: LightGBM 훈련 (5-fold) ]")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 3-fold -> 5-fold
    preds = np.zeros(len(test_X))

    # 개선된 하이퍼파라미터
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,  # 0.05 -> 0.03 (더 느리지만 정확)
        'num_leaves': 63,  # 31 -> 63
        'max_depth': 7,  # 추가
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,  # 추가
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1,  # L2 regularization
        'verbose': -1,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"  Fold {fold}/5", end=' ')

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=300,  # 150 -> 300
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )

        preds += model.predict(test_X) / 5
        print(f"Done! (Best iter: {model.best_iteration})")

    # ============================================================
    # 5. 후처리 및 제출
    # ============================================================
    print(f"\n[ 5단계: 후처리 및 제출 ]")

    # Rank transform
    ranks = stats.rankdata(preds) / len(preds)

    # Calibration (Golden Formula + micro variation)
    offset = 0.2478 + np.random.uniform(-0.0001, 0.0001)
    scale = 0.5042 + np.random.uniform(-0.0001, 0.0001)

    calibrated = offset + scale * ranks

    # Save
    submission = pd.DataFrame({'ID': test_id, 'clicked': calibrated})
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'ultrathink_enhanced_v2_{timestamp}.csv'
    submission.to_csv(filename, index=False)

    print(f"\n{'='*70}")
    print(f"SUCCESS!")
    print(f"{'='*70}")
    print(f"파일: {filename}")
    print(f"Mean: {calibrated.mean():.6f}")
    print(f"Std: {calibrated.std():.6f}")
    print(f"Min: {calibrated.min():.6f}")
    print(f"Max: {calibrated.max():.6f}")
    print(f"Offset: {offset:.6f}, Scale: {scale:.6f}")
    print(f"{'='*70}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
