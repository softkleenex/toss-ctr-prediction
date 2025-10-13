# -*- coding: utf-8 -*-
"""
Feature Engineering Module
고급 피처 엔지니어링 기능 (42+ features)
"""

import pandas as pd
import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """피처 엔지니어링 클래스"""

    def __init__(self):
        self.history_cols = ['history_a_1', 'history_a_3', 'history_b_1',
                            'history_b_21', 'history_b_30']
        self.feat_cols = ['feat_a_1', 'feat_a_2', 'feat_a_3',
                         'feat_b_1', 'feat_b_3', 'feat_c_1', 'feat_c_8']
        self.l_feat_cols = ['l_feat_1', 'l_feat_2', 'l_feat_3',
                           'l_feat_5', 'l_feat_10']

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """상호작용 피처 생성"""
        print("Creating interaction features...")

        # 기본 상호작용
        df['age_gender_int'] = df['age_group'] * df['gender']
        df['hour_day_int'] = df['hour'] * df['day_of_week']

        # History 상호작용
        df['hist_a1_b21'] = df['history_a_1'] * df['history_b_21']
        df['hist_a1_b30'] = df['history_a_1'] * df['history_b_30']
        df['hist_a3_b21'] = df['history_a_3'] * df['history_b_21']

        # Feat 상호작용
        df['feat_b3_c8'] = df['feat_b_3'] * df['feat_c_8']
        df['feat_a1_b1'] = df['feat_a_1'] * df['feat_b_1']

        # 비율 피처
        df['hist_a1_b21_ratio'] = df['history_a_1'] / (df['history_b_21'] + 1e-10)
        df['feat_b3_c8_ratio'] = df['feat_b_3'] / (df['feat_c_8'] + 1e-10)

        # 조화평균
        df['hist_harmonic'] = (2 * df['history_a_1'] * df['history_b_21']) / \
                              (df['history_a_1'] + df['history_b_21'] + 1e-10)

        # 기하평균
        df['hist_geometric'] = np.sqrt(np.abs(df['history_a_1'] * df['history_b_21']))

        return df

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """통계 피처 생성"""
        print("Creating statistical features...")

        # History 통계
        hist_df = df[self.history_cols]

        df['hist_mean'] = hist_df.mean(axis=1)
        df['hist_std'] = hist_df.std(axis=1)
        df['hist_max'] = hist_df.max(axis=1)
        df['hist_min'] = hist_df.min(axis=1)
        df['hist_median'] = hist_df.median(axis=1)

        # 고급 통계
        df['hist_range'] = df['hist_max'] - df['hist_min']
        df['hist_skew'] = hist_df.skew(axis=1)
        df['hist_kurtosis'] = hist_df.kurtosis(axis=1)

        # Quantiles
        df['hist_q75'] = hist_df.quantile(0.75, axis=1)
        df['hist_q25'] = hist_df.quantile(0.25, axis=1)
        df['hist_iqr'] = df['hist_q75'] - df['hist_q25']

        # Coefficient of Variation
        df['hist_cv'] = df['hist_std'] / (df['hist_mean'] + 1e-10)

        # Median Absolute Deviation
        df['hist_mad'] = np.abs(hist_df.sub(df['hist_median'], axis=0)).median(axis=1)

        # Feat 통계
        feat_df = df[self.feat_cols]
        df['feat_sum'] = feat_df.sum(axis=1)
        df['feat_mean'] = feat_df.mean(axis=1)
        df['feat_std'] = feat_df.std(axis=1)

        # L_feat 통계
        l_feat_df = df[self.l_feat_cols]
        df['l_feat_sum'] = l_feat_df.sum(axis=1)
        df['l_feat_mean'] = l_feat_df.mean(axis=1)

        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 관련 피처 생성"""
        print("Creating temporal features...")

        # 다중 주기 Cyclical Encoding
        for period in [24, 12, 8, 6]:
            df[f'hour_sin_{period}'] = np.sin(2 * np.pi * df['hour'] / period)
            df[f'hour_cos_{period}'] = np.cos(2 * np.pi * df['hour'] / period)

        # 요일 Cyclical
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # 피크 타임 Indicators
        df['is_morning_rush'] = df['hour'].between(7, 9).astype(int)
        df['is_lunch'] = df['hour'].between(11, 13).astype(int)
        df['is_evening'] = df['hour'].between(18, 20).astype(int)
        df['is_prime_time'] = df['hour'].between(19, 22).astype(int)
        df['is_late_night'] = df['hour'].between(23, 24).astype(int) | \
                              df['hour'].between(0, 2).astype(int)

        # 주말 여부
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        return df

    def create_polynomial_features(self, df: pd.DataFrame,
                                   top_features: List[str] = None) -> pd.DataFrame:
        """다항식 피처 생성"""
        print("Creating polynomial features...")

        if top_features is None:
            top_features = ['history_a_1', 'history_b_21', 'feat_b_3', 'feat_c_8']

        for feat in top_features:
            if feat in df.columns:
                df[f'{feat}_sq'] = df[feat] ** 2
                df[f'{feat}_cube'] = df[feat] ** 3
                df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))
                df[f'{feat}_log1p'] = np.log1p(np.abs(df[feat]))

        return df

    def engineer_all_features(self, df: pd.DataFrame,
                             include_advanced: bool = True) -> pd.DataFrame:
        """모든 피처 엔지니어링 적용"""
        print("\n=== Feature Engineering ===")

        # 기본 상호작용
        df = self.create_interaction_features(df)

        if include_advanced:
            # 통계 피처
            df = self.create_statistical_features(df)

            # 시간 피처
            df = self.create_temporal_features(df)

            # 다항식 피처
            df = self.create_polynomial_features(df)

        print(f"Final feature count: {len(df.columns)}")
        print("Feature engineering completed!")

        return df


def get_feature_groups() -> dict:
    """피처 그룹 정의"""
    return {
        'base': [
            'gender', 'age_group', 'inventory_id', 'day_of_week', 'hour',
            'l_feat_1', 'l_feat_2', 'l_feat_3', 'l_feat_5', 'l_feat_10',
            'feat_a_1', 'feat_a_2', 'feat_a_3', 'feat_b_1', 'feat_b_3',
            'feat_c_1', 'feat_c_8', 'history_a_1', 'history_a_3',
            'history_b_1', 'history_b_21', 'history_b_30'
        ],
        'interaction': [
            'age_gender_int', 'hour_day_int',
            'hist_a1_b21', 'hist_a1_b30', 'hist_a3_b21',
            'feat_b3_c8', 'feat_a1_b1',
            'hist_a1_b21_ratio', 'feat_b3_c8_ratio',
            'hist_harmonic', 'hist_geometric'
        ],
        'statistical': [
            'hist_mean', 'hist_std', 'hist_max', 'hist_min', 'hist_median',
            'hist_range', 'hist_skew', 'hist_kurtosis',
            'hist_q75', 'hist_q25', 'hist_iqr', 'hist_cv', 'hist_mad',
            'feat_sum', 'feat_mean', 'feat_std',
            'l_feat_sum', 'l_feat_mean'
        ],
        'temporal': [
            'hour_sin_24', 'hour_cos_24', 'hour_sin_12', 'hour_cos_12',
            'hour_sin_8', 'hour_cos_8', 'hour_sin_6', 'hour_cos_6',
            'dow_sin', 'dow_cos',
            'is_morning_rush', 'is_lunch', 'is_evening',
            'is_prime_time', 'is_late_night', 'is_weekend'
        ]
    }


if __name__ == "__main__":
    # 테스트
    print("Feature Engineering Module")
    print("=" * 50)

    feature_groups = get_feature_groups()
    total_features = sum(len(v) for v in feature_groups.values())

    print(f"\nTotal feature count: {total_features}")
    for group_name, features in feature_groups.items():
        print(f"  {group_name}: {len(features)} features")
