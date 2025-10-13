# -*- coding: utf-8 -*-
"""
Calibration Module
다양한 예측값 보정 전략
"""

import numpy as np
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')


class CalibrationStrategy:
    """예측값 보정 전략 클래스"""

    def __init__(self):
        self.methods = {
            'supreme_optimal': self._supreme_optimal,
            'refined': self._refined,
            'aggressive': self._aggressive,
            'quantile': self._quantile_transform,
            'power': self._power_transform,
            'sigmoid': self._sigmoid_stretch,
            'isotonic': self._isotonic_calibration
        }

    def calibrate(self, predictions: np.ndarray, method: str = 'refined') -> np.ndarray:
        """
        예측값 보정

        Args:
            predictions: 원본 예측값
            method: 보정 방법

        Returns:
            보정된 예측값
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. "
                           f"Available: {list(self.methods.keys())}")

        calibrated = self.methods[method](predictions)

        print(f"\nCalibration: {method}")
        print(f"  Original - Mean: {predictions.mean():.6f}, "
              f"Std: {predictions.std():.6f}")
        print(f"  Calibrated - Mean: {calibrated.mean():.6f}, "
              f"Std: {calibrated.std():.6f}")
        print(f"  Range: [{calibrated.min():.6f}, {calibrated.max():.6f}]")

        return calibrated

    def _supreme_optimal(self, predictions: np.ndarray) -> np.ndarray:
        """Supreme Optimal - 최고 성적 보정"""
        ranks = stats.rankdata(predictions) / len(predictions)
        return 0.248 + 0.504 * ranks

    def _refined(self, predictions: np.ndarray) -> np.ndarray:
        """Refined - 미세 조정 버전"""
        ranks = stats.rankdata(predictions) / len(predictions)
        offset = 0.2478 + np.random.uniform(-0.0001, 0.0001)
        scale = 0.5042 + np.random.uniform(-0.0001, 0.0001)
        return offset + scale * ranks

    def _aggressive(self, predictions: np.ndarray) -> np.ndarray:
        """Aggressive - 더 높은 목표"""
        ranks = stats.rankdata(predictions) / len(predictions)
        return 0.252 + 0.496 * ranks

    def _quantile_transform(self, predictions: np.ndarray) -> np.ndarray:
        """Quantile Transform - 균등 분포로 변환"""
        from sklearn.preprocessing import QuantileTransformer

        qt = QuantileTransformer(output_distribution='uniform')
        transformed = qt.fit_transform(predictions.reshape(-1, 1)).ravel()

        # Scale to [0.248, 0.752]
        return 0.248 + 0.504 * transformed

    def _power_transform(self, predictions: np.ndarray) -> np.ndarray:
        """Power Transform"""
        # Rank 기반
        ranks = stats.rankdata(predictions) / len(predictions)

        # Power law
        power = 0.95
        transformed = np.power(ranks, power)

        # Scale
        return 0.248 + 0.504 * transformed

    def _sigmoid_stretch(self, predictions: np.ndarray) -> np.ndarray:
        """Sigmoid Stretch"""
        ranks = stats.rankdata(predictions) / len(predictions)

        # Sigmoid 함수로 중간값 강화
        stretched = 1 / (1 + np.exp(-8 * (ranks - 0.5)))

        return 0.248 + 0.504 * stretched

    def _isotonic_calibration(self, predictions: np.ndarray,
                             y_true: np.ndarray = None) -> np.ndarray:
        """
        Isotonic Regression (학습 데이터 필요)

        Note: y_true가 없으면 rank 기반으로 대체
        """
        if y_true is None:
            # Fallback to rank-based
            return self._refined(predictions)

        ir = IsotonicRegression(out_of_bounds='clip')
        calibrated = ir.fit_transform(predictions, y_true)

        return calibrated

    def compare_methods(self, predictions: np.ndarray) -> dict:
        """모든 보정 방법 비교"""
        results = {}

        print("\n=== Calibration Methods Comparison ===")

        for method_name in ['supreme_optimal', 'refined', 'aggressive',
                           'quantile', 'power', 'sigmoid']:
            calibrated = self.calibrate(predictions, method=method_name)
            results[method_name] = calibrated

        return results


class EnsembleCalibrator:
    """여러 모델 예측값을 앙상블하고 보정"""

    def __init__(self):
        self.calibration = CalibrationStrategy()

    def weighted_ensemble(self, predictions_list: list,
                         weights: list = None) -> np.ndarray:
        """
        가중 평균 앙상블

        Args:
            predictions_list: 여러 모델의 예측값 리스트
            weights: 각 모델의 가중치
        """
        if weights is None:
            weights = [1.0] * len(predictions_list)

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        ensemble = np.average(predictions_list, axis=0, weights=weights)

        print(f"\nEnsemble: {len(predictions_list)} models")
        print(f"  Weights: {weights}")
        print(f"  Mean: {ensemble.mean():.6f}")

        return ensemble

    def rank_average(self, predictions_list: list) -> np.ndarray:
        """Rank 평균 앙상블"""
        ranks_list = []

        for pred in predictions_list:
            ranks = stats.rankdata(pred) / len(pred)
            ranks_list.append(ranks)

        avg_ranks = np.mean(ranks_list, axis=0)

        return avg_ranks

    def create_submission(self, predictions: np.ndarray,
                         test_ids: np.ndarray,
                         calibration_method: str = 'refined',
                         filename: str = None) -> dict:
        """
        제출 파일 생성

        Returns:
            submission dict with 'clicked' predictions
        """
        import pandas as pd
        from datetime import datetime

        # Calibration
        calibrated = self.calibration.calibrate(predictions, method=calibration_method)

        # Create submission
        submission = pd.DataFrame({
            'ID': test_ids,
            'clicked': calibrated
        })

        # Save
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'submission_{calibration_method}_{timestamp}.csv'

        submission.to_csv(filename, index=False)

        print(f"\nSubmission saved: {filename}")
        print(f"  Shape: {submission.shape}")
        print(f"  Stats: Mean={calibrated.mean():.6f}, "
              f"Std={calibrated.std():.6f}")

        return submission.to_dict()


def analyze_calibration_effect(original: np.ndarray,
                               calibrated: np.ndarray) -> None:
    """보정 효과 분석"""
    print("\n=== Calibration Analysis ===")

    print("\nDistribution Shift:")
    print(f"  Original  - Mean: {original.mean():.6f}, Std: {original.std():.6f}")
    print(f"  Calibrated - Mean: {calibrated.mean():.6f}, Std: {calibrated.std():.6f}")

    print("\nQuantiles:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        orig_q = np.quantile(original, q)
        cal_q = np.quantile(calibrated, q)
        print(f"  {q*100:5.1f}%: {orig_q:.6f} → {cal_q:.6f} "
              f"(Δ={cal_q-orig_q:+.6f})")


if __name__ == "__main__":
    # 테스트
    print("Calibration Module Test")
    print("=" * 50)

    # 더미 데이터
    np.random.seed(42)
    predictions = np.random.beta(2, 5, size=10000)

    calibrator = CalibrationStrategy()

    # 모든 방법 비교
    results = calibrator.compare_methods(predictions)

    print("\n\nTest completed!")
