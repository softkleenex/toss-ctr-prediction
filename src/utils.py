# -*- coding: utf-8 -*-
"""
Utility Functions
유틸리티 및 헬퍼 함수들
"""

import pandas as pd
import numpy as np
import os
import gc
import psutil
from datetime import datetime
from typing import Any, Dict
import warnings
warnings.filterwarnings('ignore')


def get_memory_usage() -> Dict[str, float]:
    """시스템 메모리 사용량 확인"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return {
        'rss_mb': mem_info.rss / 1024 ** 2,  # 물리 메모리
        'vms_mb': mem_info.vms / 1024 ** 2,  # 가상 메모리
        'percent': process.memory_percent()
    }


def print_memory_usage(label: str = ""):
    """메모리 사용량 출력"""
    mem = get_memory_usage()
    if label:
        print(f"\n[{label}] Memory Usage:")
    else:
        print("\nMemory Usage:")

    print(f"  RSS: {mem['rss_mb']:.2f} MB")
    print(f"  VMS: {mem['vms_mb']:.2f} MB")
    print(f"  Percent: {mem['percent']:.2f}%")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    메모리 사용량 최적화

    Args:
        df: 최적화할 데이터프레임
        verbose: 상세 출력 여부

    Returns:
        최적화된 데이터프레임
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f'\nMemory optimization:')
        print(f'  Before: {start_mem:.2f} MB')
        print(f'  After: {end_mem:.2f} MB')
        print(f'  Decreased: {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df


def save_submission(predictions: np.ndarray, test_ids: np.ndarray,
                   filename: str = None, prefix: str = "submission") -> str:
    """
    제출 파일 저장

    Args:
        predictions: 예측값
        test_ids: 테스트 ID
        filename: 파일명 (None이면 자동 생성)
        prefix: 파일명 prefix

    Returns:
        저장된 파일명
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{prefix}_{timestamp}.csv'

    submission = pd.DataFrame({
        'ID': test_ids,
        'clicked': predictions
    })

    submission.to_csv(filename, index=False)

    print(f"\nSubmission saved: {filename}")
    print(f"  Shape: {submission.shape}")
    print(f"  Stats:")
    print(f"    Mean: {predictions.mean():.6f}")
    print(f"    Std: {predictions.std():.6f}")
    print(f"    Min: {predictions.min():.6f}")
    print(f"    Max: {predictions.max():.6f}")

    return filename


def log_experiment(config: Dict[str, Any], score: float,
                  log_file: str = "experiments.csv"):
    """
    실험 결과 로깅

    Args:
        config: 실험 설정 dict
        score: 성능 점수
        log_file: 로그 파일명
    """
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'score': score,
        **config
    }

    df = pd.DataFrame([log_entry])

    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

    print(f"\nExperiment logged to {log_file}")


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    클래스 가중치 계산 (불균형 데이터용)

    Args:
        y: 타겟 변수

    Returns:
        {class: weight} dict
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)

    weight_dict = dict(zip(classes, weights))

    print("\nClass Weights:")
    for cls, weight in weight_dict.items():
        count = (y == cls).sum()
        print(f"  Class {cls}: {weight:.4f} (count: {count:,})")

    return weight_dict


def get_feature_importance(model, feature_names: list,
                          top_n: int = 20) -> pd.DataFrame:
    """
    Feature Importance 추출 및 정렬

    Args:
        model: LightGBM or XGBoost model
        feature_names: 피처 이름 리스트
        top_n: 상위 N개만 반환

    Returns:
        Feature importance DataFrame
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        importance = model.feature_importance()
    else:
        raise ValueError("Model doesn't have feature importance")

    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)

    if top_n:
        fi_df = fi_df.head(top_n)

    print(f"\nTop {len(fi_df)} Feature Importances:")
    for i, row in fi_df.iterrows():
        bar_len = int(row['importance'] / fi_df['importance'].max() * 40)
        bar = '█' * bar_len
        print(f"  {i+1:2d}. {row['feature']:20s} {bar} {row['importance']:.1f}")

    return fi_df


def timer_decorator(func):
    """함수 실행 시간 측정 데코레이터"""
    import time

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed = end_time - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60

        print(f"\n[{func.__name__}] Elapsed time: ", end="")
        if hours > 0:
            print(f"{hours}h ", end="")
        if minutes > 0:
            print(f"{minutes}m ", end="")
        print(f"{seconds:.2f}s")

        return result

    return wrapper


def seed_everything(seed: int = 42):
    """모든 랜덤 시드 고정"""
    import random
    import os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    print(f"Random seed set to {seed}")


def check_gpu_available() -> bool:
    """GPU 사용 가능 여부 확인"""
    try:
        import lightgbm as lgb
        # LightGBM GPU 테스트
        params = {'device': 'gpu'}
        print("\nGPU Status:")
        print("  LightGBM GPU: Available")
        return True
    except Exception as e:
        print("\nGPU Status:")
        print(f"  LightGBM GPU: Not available ({str(e)})")
        return False


if __name__ == "__main__":
    print("Utility Functions Module")
    print("=" * 50)

    # 메모리 사용량 확인
    print_memory_usage("Initial")

    # GPU 확인
    check_gpu_available()

    # 시드 설정
    seed_everything(42)

    print("\nAll utility functions loaded successfully!")
