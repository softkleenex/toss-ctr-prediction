# 제출 기록

## 최종 제출 (Top 3)

### 1위: Supreme Evolved Refined
- **Score**: 0.3434805649
- **Date**: 2025-09-20
- **Description**: Supreme 전략 + 고급 FE + 25 모델 앙상블
- **File**: `ultrathink_supreme_evolved_refined_20250918_145412.csv`

**특징**:
- 42+ 피처 (고급 FE)
- LightGBM 25개 (5 seeds × 5 folds)
- XGBoost 추가
- 6가지 Calibration 전략
- Refined calibration 선택

### 2위: Final Push v1
- **Score**: 0.3434775373
- **Date**: 2025-10-01
- **Description**: 최종 Push 버전
- **File**: `ultrathink_final_push_35_v1_20250923_133448.csv`

### 3위: Enhanced V2
- **Score**: 0.3425593061
- **Date**: 2025-10-12
- **Description**: 개선된 전처리 + 기본 FE
- **File**: `ultrathink_enhanced_v2_20251012_133850.csv`

**특징**:
- 28 피처 (기본 FE)
- LightGBM 5-fold
- 2M 샘플링
- 단일 Calibration

## 전체 제출 히스토리

| 날짜 | 파일 | Score | 순위 |
|------|------|-------|------|
| 2025-10-12 | enhanced_v2 | 0.3425593061 | #3 |
| 2025-10-01 | final_push_v1 | 0.3434775373 | #2 |
| 2025-09-20 | supreme_evolved_refined | **0.3434805649** | **#1** |
| 2025-09-20 | supreme_evolved_optimal | 0.3434745079 | #4 |
| 2025-09-20 | supreme_evolved_aggressive | 0.3434062571 | #5 |

## 실패 사례

| 날짜 | 전략 | Score | 원인 |
|------|------|-------|------|
| 2025-10-08 | RADICAL Extreme Rank | 0.296761395 | 과도한 변환 |
| 2025-10-08 | MEGA Ensemble | 0.3102118172 | 잘못된 앙상블 |
| 2025-09-16 | 10x Extreme | 0.2328561672 | 극단적 설정 |

## 학습 내용

1. **Feature Engineering이 핵심**: 22개 → 42개로 증가 시 +0.0025 개선
2. **앙상블의 중요성**: 단일 모델보다 25개 앙상블이 안정적
3. **Calibration 최적화**: Rank-based가 가장 효과적
4. **과적합 주의**: 피처 너무 많으면 오히려 하락

## 최적 설정

```python
# 피처
- Base: 22개
- Engineered: 42+개
- 상호작용, 통계, 시간, 다항식

# 모델
- LightGBM: 25개 (5×5)
- XGBoost: 1-2개
- Ensemble: 가중 평균

# Calibration
- Method: Rank-based Linear
- Formula: 0.248 + 0.504 × rank
```
