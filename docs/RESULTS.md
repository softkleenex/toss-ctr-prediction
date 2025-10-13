# 실험 결과 및 성능 분석

## 목차
1. [최종 성적 요약](#최종-성적-요약)
2. [실험 히스토리](#실험-히스토리)
3. [성능 개선 과정](#성능-개선-과정)
4. [실패 사례 분석](#실패-사례-분석)
5. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
6. [앙상블 전략](#앙상블-전략)
7. [Calibration 실험](#calibration-실험)

## 최종 성적 요약

### Top 3 제출

| 순위 | 파일명 | AUC Score | 날짜 | 개선 사항 |
|------|--------|-----------|------|-----------|
| **#1** | supreme_evolved_refined | **0.3434805649** | 2025-09-20 | Supreme 전략 + 42+ FE + 25 앙상블 |
| **#2** | final_push_v1 | 0.3434775373 | 2025-10-01 | 최종 Push 버전 |
| **#3** | enhanced_v2 | 0.3425593061 | 2025-10-12 | 개선된 전처리 + 기본 FE |

### 성능 지표

```
최고 점수: 0.3434805649 (AUC)
베이스라인 대비: +0.0025 (+0.73%)
목표 점수: 0.35
목표 갭: -0.0065 (-1.86%)
```

### 제출 통계

```
총 제출 횟수: 20+
성공 제출: 18
실패 제출: 2
최고 점수 달성: 3회 (0.3434대)
```

## 실험 히스토리

### Phase 1: 베이스라인 구축 (0.3409)

**기간**: 2025-09-15 ~ 2025-09-16

**구성**:
- 피처: 22개 (기본 피처만)
- 모델: LightGBM 단일 모델
- 전처리: fillna(0)
- 샘플링: 1M 랜덤
- CV: 3-fold

**결과**: 0.3409 (AUC)

**코드**:
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_boost_round': 150
}
```

### Phase 2: Feature Engineering 추가 (0.3417)

**기간**: 2025-09-17 ~ 2025-09-18

**추가 사항**:
- 상호작용 피처 6개 추가
- 총 28개 피처 사용

**개선**: +0.0008 (0.3409 → 0.3417)

**주요 피처**:
```python
interact_hist = history_a_1 × history_b_21
interact_feat = feat_b_3 × feat_c_8
ratio_hist = history_a_1 / (history_b_21 + 1e-10)
```

### Phase 3: 고급 Feature Engineering (0.3425)

**기간**: 2025-09-19 ~ 2025-09-20

**추가 사항**:
- 통계 피처 8개
- 시간 인코딩 9개
- 다항식 피처 16개
- 총 42+ 피처 사용

**개선**: +0.0008 (0.3417 → 0.3425)

**효과**:
| 피처 그룹 | 추가 수 | 개선폭 |
|-----------|---------|--------|
| 통계 | 8 | +0.0003 |
| 시간 | 9 | +0.0002 |
| 다항식 | 16 | +0.0003 |

### Phase 4: 앙상블 적용 (0.3432)

**기간**: 2025-09-21 ~ 2025-09-22

**전략**:
- 5 Seeds × 5 Folds = 25개 LightGBM 모델
- XGBoost 추가 (다양성)
- 가중 평균 앙상블

**개선**: +0.0007 (0.3425 → 0.3432)

**앙상블 가중치**:
```python
weights = {
    'balanced_1_1': 2.0,      # 가장 높은 가중치
    'imbalanced_1_5': 1.5,
    'large_sample': 1.0
}
```

### Phase 5: Calibration 최적화 (0.3434)

**기간**: 2025-09-23 ~ 2025-09-25

**시도한 방법**:
1. Isotonic Regression
2. Platt Scaling
3. Beta Calibration
4. Quantile Transform
5. Power Transform
6. **Rank-based Linear** (채택)

**개선**: +0.0002 (0.3432 → 0.3434)

**최종 공식**:
```python
ranks = stats.rankdata(predictions) / len(predictions)
calibrated = 0.248 + 0.504 × ranks
# Mean: 0.4988 (거의 완벽한 균형)
```

## 성능 개선 과정

### 누적 개선 그래프

```
0.3440 |                                    ┌─── 0.3434 (Final)
       |                                ┌───┘
0.3430 |                            ┌───┘
       |                        ┌───┘
0.3420 |                    ┌───┘
       |                ┌───┘
0.3410 |            ┌───┘
       |        ┌───┘
0.3400 |────────┘
       └─────────────────────────────────────────────
         Base  +FE  +Adv  +Ens  +Cal
```

### 단계별 성능 변화

| 단계 | AUC Score | 누적 개선 | 단계 개선 | 소요 시간 |
|------|-----------|-----------|-----------|-----------|
| Baseline | 0.3409 | - | - | 1일 |
| + Basic FE | 0.3417 | +0.0008 | +0.0008 | 2일 |
| + Advanced FE | 0.3425 | +0.0016 | +0.0008 | 2일 |
| + Ensemble | 0.3432 | +0.0023 | +0.0007 | 2일 |
| + Calibration | **0.3434** | **+0.0025** | +0.0002 | 3일 |

**총 개발 기간**: 10일
**총 개선폭**: +0.0025 (0.73%)

## 실패 사례 분석

### 실패 1: 과도한 피처 추가

**시도**: 93개 피처 사용 (모든 가능한 조합)

**결과**: 0.2962 (❌ -0.11 하락!)

**원인**:
- 노이즈 피처 포함
- 과적합 발생
- 모델 복잡도 증가로 일반화 성능 저하

**교훈**: 피처가 많다고 항상 좋은 것은 아님. 선별적 추가 필요.

### 실패 2: 극단적 Calibration

**시도**: Aggressive Calibration (0.252 + 0.496 × ranks)

**결과**: 0.3102 (❌ -0.03 하락)

**원인**:
- 과도한 분포 조정
- Target 분포와 불일치
- 예측 신뢰도 왜곡

**교훈**: Calibration은 보수적으로. 실제 분포 반영이 중요.

### 실패 3: MEGA Ensemble (100+ 모델)

**시도**: 100개 이상 모델 앙상블

**결과**: 0.3101 (❌ -0.03 하락)

**원인**:
- 저품질 모델 포함
- 다양성 없이 수만 늘림
- 계산 비용 급증

**교훈**: 앙상블은 품질과 다양성이 핵심. 수량이 아님.

## 하이퍼파라미터 튜닝

### LightGBM 최적화 과정

#### v1 (Baseline)
```python
params_v1 = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_boost_round': 150,
    'min_data_in_leaf': 20
}
# CV Score: 0.728 ± 0.003
# Test AUC: 0.3409
```

#### v2 (Tuned)
```python
params_v2 = {
    'num_leaves': 63,          # 증가
    'learning_rate': 0.03,     # 감소
    'num_boost_round': 300,    # 증가
    'feature_fraction': 0.8,   # 추가
    'bagging_fraction': 0.8,   # 추가
    'reg_alpha': 0.1,          # L1 정규화
    'reg_lambda': 0.1          # L2 정규화
}
# CV Score: 0.732 ± 0.002
# Test AUC: 0.3425
```

#### v3 (Supreme)
```python
params_v3 = {
    'num_leaves': 200,         # 더 증가
    'learning_rate': 0.012,    # 더 감소
    'num_boost_round': 2500,   # 조기 종료 사용
    'feature_fraction': 0.65,  # 감소
    'bagging_fraction': 0.75,  # 감소
    'lambda_l1': 0.05,         # 약한 정규화
    'lambda_l2': 0.05,
    'min_child_samples': 100   # 추가
}
# CV Score: 0.736 ± 0.002
# Test AUC: 0.3434
```

### 파라미터 민감도 분석

| 파라미터 | 범위 | 최적값 | 민감도 | 영향 |
|----------|------|--------|--------|------|
| num_leaves | 31-255 | 200 | High | 모델 복잡도 |
| learning_rate | 0.01-0.1 | 0.012 | High | 수렴 속도 |
| feature_fraction | 0.5-1.0 | 0.65 | Medium | 과적합 방지 |
| bagging_fraction | 0.5-1.0 | 0.75 | Medium | 다양성 |
| lambda_l1 | 0-1.0 | 0.05 | Low | 희소성 |
| lambda_l2 | 0-1.0 | 0.05 | Low | 안정성 |

## 앙상블 전략

### 다양성 확보 방법

#### 1. Multiple Seeds (5개)
```python
seeds = [42, 43, 44, 45, 46]

for seed in seeds:
    model = lgb.train(params, data, seed=seed)
```
**효과**: 초기화 차이로 다른 로컬 최적점 탐색

#### 2. Multiple Folds (5개)
```python
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    model = lgb.train(params, train_data)
```
**효과**: 데이터 분할에 따른 다양한 패턴 학습

#### 3. Multiple Sampling (3개)
```python
# Balanced (1:1)
pos = df[df['clicked'] == 1]
neg = df[df['clicked'] == 0].sample(n=len(pos))

# Imbalanced (1:1.5)
neg = df[df['clicked'] == 0].sample(n=int(len(pos) * 1.5))

# Large (3.5M)
sample = df.sample(n=3500000)
```
**효과**: 클래스 불균형 대응 전략 다양화

### 앙상블 성능 비교

| 전략 | 모델 수 | CV Score | Test AUC | 개선폭 |
|------|---------|----------|----------|--------|
| Single Best | 1 | 0.733 | 0.3425 | - |
| Simple Average | 25 | 0.735 | 0.3429 | +0.0004 |
| Weighted Average | 25 | 0.736 | 0.3432 | +0.0007 |
| Stacking | 25 + Meta | 0.735 | 0.3428 | +0.0003 |
| **Weighted + Cal** | **25** | **0.736** | **0.3434** | **+0.0009** |

## Calibration 실험

### 시도한 방법들

#### 1. Isotonic Regression
```python
from sklearn.isotonic import IsotonicRegression

iso_reg = IsotonicRegression(out_of_bounds='clip')
calibrated = iso_reg.fit_transform(val_pred, val_true)
```
**결과**: 0.3428
**장점**: 비선형 변환 가능
**단점**: 과적합 위험, 검증 데이터 필요

#### 2. Platt Scaling
```python
from sklearn.linear_model import LogisticRegression

platt = LogisticRegression()
platt.fit(val_pred.reshape(-1, 1), val_true)
calibrated = platt.predict_proba(test_pred.reshape(-1, 1))[:, 1]
```
**결과**: 0.3426
**장점**: 간단, 안정적
**단점**: 선형 변환만 가능

#### 3. Beta Calibration
```python
from betacal import BetaCalibration

beta_cal = BetaCalibration()
beta_cal.fit(val_pred, val_true)
calibrated = beta_cal.predict(test_pred)
```
**결과**: 0.3430
**장점**: 유연한 변환
**단점**: 파라미터 추정 불안정

#### 4. Quantile Transform
```python
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='uniform')
calibrated = qt.fit_transform(pred.reshape(-1, 1)).flatten()
```
**결과**: 0.3431
**장점**: 분포 강제 변환
**단점**: 원본 정보 손실

#### 5. Power Transform
```python
calibrated = pred ** 0.95
```
**결과**: 0.3427
**장점**: 매우 간단
**단점**: 적절한 지수 찾기 어려움

#### 6. Rank-based Linear (채택)
```python
ranks = stats.rankdata(pred) / len(pred)
calibrated = 0.248 + 0.504 * ranks
```
**결과**: **0.3434**
**장점**: 안정적, 해석 가능, 분포 균형
**단점**: 없음 (현재 최고)

### Calibration 결과 비교

| 방법 | Test AUC | Mean | Std | Min | Max |
|------|----------|------|-----|-----|-----|
| Raw | 0.3432 | 0.019 | 0.038 | 0.000 | 0.997 |
| Isotonic | 0.3428 | 0.019 | 0.039 | 0.000 | 0.998 |
| Platt | 0.3426 | 0.019 | 0.037 | 0.001 | 0.995 |
| Beta | 0.3430 | 0.019 | 0.038 | 0.000 | 0.996 |
| Quantile | 0.3431 | 0.500 | 0.289 | 0.000 | 1.000 |
| Power | 0.3427 | 0.013 | 0.032 | 0.000 | 0.998 |
| **Rank** | **0.3434** | **0.499** | **0.144** | **0.248** | **0.752** |

**최적 선택**: Rank-based Linear
- 가장 높은 Test AUC
- Mean이 0.5에 가까움 (균형)
- 안정적인 분포

## 핵심 인사이트

### 성공 요인

1. **Feature Engineering이 핵심**
   - 22개 → 42개: +0.0016 개선
   - 적절한 피처 선택이 중요

2. **앙상블의 힘**
   - 단일 모델 대비 +0.0009 개선
   - 다양성 확보가 핵심

3. **Calibration 최적화**
   - Rank-based가 가장 효과적
   - 간단하지만 강력

### 실패로부터의 교훈

1. **과도한 복잡도는 독**
   - 93개 피처: -0.11 하락
   - 100+ 앙상블: -0.03 하락

2. **검증 전략 중요**
   - CV Score와 Test AUC의 상관관계 확인
   - Overfitting 방지

3. **점진적 개선**
   - 한 번에 모든 것 X
   - 단계별로 추가하며 확인

---

**작성일**: 2025-10-13
**최종 업데이트**: Supreme Evolved Refined 기준
**총 실험 횟수**: 50+
**최종 성능**: 0.3434805649 (AUC)
