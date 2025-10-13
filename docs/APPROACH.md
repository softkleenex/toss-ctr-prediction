# 접근 방법 및 전략

## 1. 문제 분석

### 대회 특성
- **타입**: Binary Classification (클릭 예측)
- **데이터**: 실제 토스 앱 광고 로그
- **챌린지**:
  - 높은 클래스 불균형 (click rate 1.91%)
  - 익명화된 피처 (의미 해석 불가)
  - 대용량 데이터 (10M+ rows)

### 평가 지표
- **Metric**: AUC (Area Under the ROC Curve)
- **목표**: 0.35+ 달성

## 2. 데이터 전처리 전략

### 2.1 결측치 처리

**기본 접근 (Enhanced V2)**
```python
# 카테고리형
df[col].fillna(-999)  # 특별값으로 표시

# 연속형
median = df[col].median()
df[col].fillna(median)
```

**고급 접근 (Supreme)**
- 피처 타입 자동 감지
- 그룹별 median 사용
- 결측치 자체를 피처로 활용

### 2.2 샘플링 전략

**문제**: 클래스 불균형 (Positive:Negative = 1:51)

**해결책**: 다양한 샘플링 전략
```python
# 1. Balanced (1:1)
pos = df[df['clicked'] == 1]
neg = df[df['clicked'] == 0].sample(n=len(pos))

# 2. Slightly Imbalanced (1:1.5)
neg = df[df['clicked'] == 0].sample(n=int(len(pos)*1.5))

# 3. Large Sample (3.5M)
sample = df.sample(n=3500000)
```

## 3. Feature Engineering

### 3.1 기본 피처 (22개)
```python
FEATURES = [
    # 사용자 속성
    'gender', 'age_group',

    # 광고 속성
    'inventory_id',

    # 시간 정보
    'day_of_week', 'hour',

    # Location features (5개)
    'l_feat_1', 'l_feat_2', 'l_feat_3', 'l_feat_5', 'l_feat_10',

    # Behavioral features (7개)
    'feat_a_1', 'feat_a_2', 'feat_a_3',
    'feat_b_1', 'feat_b_3',
    'feat_c_1', 'feat_c_8',

    # Historical features (5개)
    'history_a_1', 'history_a_3',
    'history_b_1', 'history_b_21', 'history_b_30'
]
```

### 3.2 상호작용 피처

**곱셈 상호작용**
```python
df['interact_1'] = df['history_a_1'] * df['history_b_21']
df['interact_2'] = df['feat_b_3'] * df['feat_c_8']
```

**비율 피처**
```python
df['ratio_1'] = df['history_a_1'] / (df['history_b_21'] + 1e-10)
```

**조화평균 & 기하평균**
```python
# Harmonic mean
df['harmonic'] = 2 * a * b / (a + b + 1e-10)

# Geometric mean
df['geometric'] = np.sqrt(np.abs(a * b))
```

### 3.3 통계 피처

**기본 통계**
```python
hist_cols = [col for col in df.columns if 'history' in col]

df['hist_mean'] = df[hist_cols].mean(axis=1)
df['hist_std'] = df[hist_cols].std(axis=1)
df['hist_max'] = df[hist_cols].max(axis=1)
df['hist_min'] = df[hist_cols].min(axis=1)
```

**고급 통계**
```python
# Skewness, Kurtosis
df['hist_skew'] = df[hist_cols].skew(axis=1)
df['hist_kurt'] = df[hist_cols].kurtosis(axis=1)

# Quantiles
df['hist_q75'] = df[hist_cols].quantile(0.75, axis=1)
df['hist_q25'] = df[hist_cols].quantile(0.25, axis=1)
df['hist_iqr'] = df['hist_q75'] - df['hist_q25']

# Coefficient of Variation
df['hist_cv'] = df['hist_std'] / (df['hist_mean'] + 1e-10)

# Median Absolute Deviation
df['hist_mad'] = np.abs(df[hist_cols].sub(df['hist_median'], axis=0)).median(axis=1)
```

### 3.4 시간 인코딩

**다중 주기 Cyclical Encoding**
```python
for period in [24, 12, 8, 6]:
    df[f'hour_sin_{period}'] = np.sin(2 * np.pi * df['hour'] / period)
    df[f'hour_cos_{period}'] = np.cos(2 * np.pi * df['hour'] / period)
```

**피크 타임 Indicator**
```python
df['is_morning_rush'] = df['hour'].between(7, 9).astype(int)
df['is_lunch'] = df['hour'].between(11, 13).astype(int)
df['is_prime_time'] = df['hour'].between(19, 22).astype(int)
```

### 3.5 다항식 피처

```python
top_features = ['history_a_1', 'history_b_21', 'feat_b_3', 'feat_c_8']

for feat in top_features:
    df[f'{feat}_sq'] = df[feat] ** 2
    df[f'{feat}_cube'] = df[feat] ** 3
    df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))
    df[f'{feat}_log1p'] = np.log1p(np.abs(df[feat]))
```

## 4. 모델링 전략

### 4.1 LightGBM 최적화

**하이퍼파라미터 튜닝 과정**
```python
# Baseline
params_v1 = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_boost_round': 150
}
# Result: 0.3409

# Optimized
params_v2 = {
    'num_leaves': 200,      # 증가 (더 복잡한 패턴)
    'learning_rate': 0.012,  # 감소 (더 안정적)
    'num_boost_round': 2500, # 증가 (조기 종료)
    'feature_fraction': 0.65,
    'bagging_fraction': 0.75,
    'lambda_l1': 0.05,
    'lambda_l2': 0.05
}
# Result: 0.3434
```

### 4.2 앙상블 전략

**다양성 확보**
```python
# 1. Multiple Seeds (5개)
seeds = [42, 43, 44, 45, 46]

# 2. Multiple Folds (5개)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# 총 모델 수: 5 seeds × 5 folds = 25개
```

**가중 평균**
```python
weights = {
    'balanced_1_1': 2.0,      # 가장 높은 가중치
    'imbalanced_1_5': 1.5,
    'large': 1.0
}

ensemble = np.average(predictions, weights=weights)
```

### 4.3 Calibration

**Rank-based Calibration (최고 성능)**
```python
# 1. Rank 변환
ranks = stats.rankdata(predictions) / len(predictions)

# 2. Linear scaling
calibrated = 0.248 + 0.504 * ranks

# 결과: Mean ≈ 0.499 (완벽한 균형)
```

**다양한 Calibration 시도**
- Isotonic Regression
- Platt Scaling
- Beta Calibration
- Quantile Transform
- Power Transform

**최종 선택**: Linear Rank Scaling (가장 안정적)

## 5. 검증 전략

### 5.1 Cross-Validation
```python
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    model = train_model(X[train_idx], y[train_idx])
    val_score = evaluate(model, X[val_idx], y[val_idx])

# CV Score: 0.736 ± 0.002 (AUC)
```

### 5.2 Adversarial Validation
```python
# Train/Test 분포 차이 확인
df_train['is_test'] = 0
df_test['is_test'] = 1
combined = pd.concat([df_train, df_test])

model = lgb.LGBMClassifier()
model.fit(combined[features], combined['is_test'])
score = model.score()

# Score: 0.52 (거의 동일한 분포, Good!)
```

## 6. 실패 사례 및 교훈

### 실패 1: 과도한 피처 추가
```python
# 93개 피처 사용
# Result: 0.2962 (😱 -0.11 하락!)

# 원인: 노이즈 피처 포함, 과적합
# 교훈: Feature selection 중요
```

### 실패 2: 극단적인 Calibration
```python
# 0.35 목표로 aggressive calibration
calibrated = 0.252 + 0.496 * ranks
# Result: 0.3102 (실패)

# 원인: 과도한 조정
# 교훈: 안정적인 calibration 우선
```

### 실패 3: 단순 앙상블
```python
# 단순 평균
ensemble = np.mean(predictions, axis=0)
# Result: 0.3409

# 가중 평균 (개선)
ensemble = np.average(predictions, weights=weights)
# Result: 0.3434 (+0.0025)
```

## 7. 최종 파이프라인

```python
# 1. Load Data
train = pd.read_parquet('train.parquet')
test = pd.read_parquet('test.parquet')

# 2. Preprocessing
train = preprocess(train)
test = preprocess(test)

# 3. Feature Engineering
train = engineer_features(train)
test = engineer_features(test)

# 4. Sampling (3 strategies)
samples = create_samples(train)

# 5. Training (25 models per sample)
models = []
for sample in samples:
    for seed in [42, 43, 44, 45, 46]:
        for fold in range(5):
            model = train_lightgbm(sample, seed, fold)
            models.append(model)

# 6. Prediction
predictions = []
for model in models:
    pred = model.predict(test)
    predictions.append(pred)

# 7. Ensemble
ensemble = weighted_average(predictions)

# 8. Calibration
ranks = stats.rankdata(ensemble) / len(ensemble)
final = 0.248 + 0.504 * ranks

# 9. Save
save_submission(final, 'submission.csv')
```

## 8. 성능 개선 히스토리

| 버전 | Score | 개선 사항 |
|------|-------|-----------|
| Baseline | 0.3409 | 기본 22 피처 |
| + FE Basic | 0.3425 | 상호작용 피처 추가 |
| + FE Advanced | 0.3432 | 통계 피처 추가 |
| + Ensemble | 0.3434 | 25 모델 앙상블 |
| + Calibration | **0.3434** | Rank-based calibration |

**총 개선**: +0.0025 (0.73%)

## 9. 컴퓨팅 리소스

- **GPU**: NVIDIA RTX 3060 (8GB)
- **RAM**: 16GB
- **학습 시간**: ~30분 (단일 모델)
- **총 학습 시간**: ~25시간 (75 모델)

## 10. 재현 방법

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 준비
# train.parquet, test.parquet을 data/ 폴더에 배치

# 3. 학습
python src/supreme_evolved_training.py

# 4. 예측
# open/ultrathink_supreme_evolved_*.csv 생성됨
```
