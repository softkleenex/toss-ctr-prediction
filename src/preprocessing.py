# -*- coding: utf-8 -*-
"""
Data Preprocessing Module
전처리 및 데이터 로딩 기능
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import gc

class DataPreprocessor:
    """데이터 전처리 클래스"""

    def __init__(self, features: List[str]):
        self.features = features
        self.cat_features = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
        self.num_features = [f for f in features if f not in self.cat_features]

    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로딩"""
        print("Loading data...")
        train_df = pd.read_parquet(train_path, columns=self.features + ['clicked'])
        test_df = pd.read_parquet(test_path, columns=['ID'] + self.features)

        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"Click rate: {train_df['clicked'].mean():.4f}")

        return train_df, test_df

    def handle_missing_values(self, df: pd.DataFrame, mode: str = 'advanced') -> pd.DataFrame:
        """
        결측치 처리

        Args:
            df: 처리할 데이터프레임
            mode: 'basic' or 'advanced'
        """
        if mode == 'basic':
            # 기본 전략: 0으로 채우기
            return df.fillna(0)

        elif mode == 'advanced':
            # 고급 전략: 타입별 다른 처리
            for col in self.features:
                # Numeric 변환
                df[col] = pd.to_numeric(df[col], errors='coerce')

                missing_pct = df[col].isnull().mean()

                if missing_pct > 0:
                    if col in self.cat_features:
                        # 카테고리형: -999 (특별값)
                        df[col].fillna(-999, inplace=True)
                    else:
                        # 연속형: median
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                else:
                    # 결측치 없어도 0으로
                    df[col].fillna(0, inplace=True)

        return df

    def create_balanced_sample(self, df: pd.DataFrame, ratio: str = '1:1',
                              n_samples: int = None, random_state: int = 42) -> pd.DataFrame:
        """
        클래스 밸런스 샘플링

        Args:
            df: 전체 데이터
            ratio: '1:1', '1:1.5', 'large' 등
            n_samples: 총 샘플 수 (ratio='large'일 때만)
            random_state: 랜덤 시드
        """
        pos = df[df['clicked'] == 1]
        neg = df[df['clicked'] == 0]

        print(f"\nCreating {ratio} sample...")

        if ratio == '1:1':
            # Balanced sampling
            n_pos = len(pos)
            neg_sample = neg.sample(n=min(n_pos, len(neg)), random_state=random_state)
            pos_sample = pos

        elif ratio == '1:1.5':
            # Slightly imbalanced
            n_pos = len(pos)
            neg_sample = neg.sample(n=min(int(n_pos * 1.5), len(neg)), random_state=random_state)
            pos_sample = pos

        elif ratio == 'large':
            # Large sample with original ratio
            if n_samples is None:
                n_samples = 3500000

            n_pos = min(int(n_samples * 0.019), len(pos))  # 1.9% click rate
            n_neg = n_samples - n_pos

            pos_sample = pos.sample(n=n_pos, random_state=random_state)
            neg_sample = neg.sample(n=min(n_neg, len(neg)), random_state=random_state)

        sample = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=random_state)

        print(f"  Total: {len(sample):,}")
        print(f"  Positive: {len(pos_sample):,}")
        print(f"  Negative: {len(neg_sample):,}")
        print(f"  Ratio: 1:{len(neg_sample)/len(pos_sample):.2f}")

        return sample

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """메모리 최적화를 위한 dtype 변경"""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)

        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype(np.int32)

        return df

    def get_memory_usage(self, df: pd.DataFrame) -> str:
        """메모리 사용량 확인"""
        mem = df.memory_usage(deep=True).sum() / 1024**2
        return f"{mem:.2f} MB"


def load_and_preprocess(train_path: str = 'train.parquet',
                       test_path: str = 'test.parquet',
                       features: List[str] = None,
                       sample_strategy: str = '1:1',
                       preprocessing_mode: str = 'advanced') -> Tuple:
    """
    메인 전처리 함수

    Returns:
        X_train, y_train, X_test, test_ids
    """
    if features is None:
        features = [
            'gender', 'age_group', 'inventory_id', 'day_of_week', 'hour',
            'l_feat_1', 'l_feat_2', 'l_feat_3', 'l_feat_5', 'l_feat_10',
            'feat_a_1', 'feat_a_2', 'feat_a_3', 'feat_b_1', 'feat_b_3',
            'feat_c_1', 'feat_c_8', 'history_a_1', 'history_a_3',
            'history_b_1', 'history_b_21', 'history_b_30'
        ]

    preprocessor = DataPreprocessor(features)

    # 데이터 로딩
    train_df, test_df = preprocessor.load_data(train_path, test_path)

    # 결측치 처리
    train_df = preprocessor.handle_missing_values(train_df, mode=preprocessing_mode)
    test_df = preprocessor.handle_missing_values(test_df, mode=preprocessing_mode)

    # 샘플링
    train_sample = preprocessor.create_balanced_sample(train_df, ratio=sample_strategy)

    # dtype 최적화
    train_sample = preprocessor.optimize_dtypes(train_sample)
    test_df = preprocessor.optimize_dtypes(test_df)

    # 분리
    X_train = train_sample[features]
    y_train = train_sample['clicked'].values
    X_test = test_df[features]
    test_ids = test_df['ID'].values

    # 메모리 정리
    del train_df, test_df, train_sample
    gc.collect()

    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Memory: Train={preprocessor.get_memory_usage(X_train)}, "
          f"Test={preprocessor.get_memory_usage(X_test)}")

    return X_train, y_train, X_test, test_ids


if __name__ == "__main__":
    # 테스트
    X_train, y_train, X_test, test_ids = load_and_preprocess(
        sample_strategy='1:1',
        preprocessing_mode='advanced'
    )
    print("\nPreprocessing completed successfully!")
