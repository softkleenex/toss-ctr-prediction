# Toss NEXT ML Challenge - CTR Prediction

토스 광고 클릭률 예측 AI 경진대회 참가 프로젝트

## 📋 프로젝트 개요

- **대회**: Toss NEXT 2025 ML Challenge - 광고 클릭률(CTR) 예측
- **주최**: Dacon x 토스
- **기간**: 2025.09 ~ 2025.10
- **목표**: 토스 앱 내 광고 클릭 확률 예측 (Binary Classification)

## 🎯 최종 성적

| 제출 | 스코어 | 순위 | 설명 |
|------|--------|------|------|
| **Supreme Evolved Refined** | **0.3434805649** | 상위권 | 최고 성적 |
| Final Push v1 | 0.3434775373 | - | 2위 스코어 |
| Enhanced V2 | 0.3425593061 | - | 개선된 전처리 버전 |

**평가 지표**: AUC (Area Under the Curve)

## 🛠 기술 스택

### 핵심 기술
- **언어**: Python 3.12
- **ML 프레임워크**: LightGBM, XGBoost
- **데이터 처리**: Pandas, NumPy
- **특징 추출**: Scipy, Scikit-learn

### 주요 라이브러리
```python
lightgbm==4.x      # GPU 가속 Gradient Boosting
xgboost==2.x       # 앙상블 다양성
pandas==2.x        # 데이터 처리
numpy==1.x         # 수치 연산
scipy==1.x         # 통계 및 변환
```

## 📊 데이터셋

- **Train**: 10.7M rows, ~8.8GB (Parquet)
- **Test**: 1.5M rows, ~1.3GB (Parquet)
- **Features**: 익명화된 사용자/광고/컨텍스트 피처
- **Target**: clicked (0 or 1)
- **특징**:
  - 실제 토스 앱 광고 노출/클릭 로그
  - 보안상 피처명 익명화
  - 높은 클래스 불균형 (Click rate: 1.91%)

## 🏗 시스템 아키텍처

### 1. 데이터 전처리 파이프라인
```
Raw Data → Missing Value Handling → Feature Engineering → Sampling → Model Training
```

### 2. 모델 앙상블 구조
```
├── LightGBM (Primary)
│   ├── 5-Fold Cross Validation
│   ├── 5 Different Seeds
│   └── GPU Acceleration
├── XGBoost (Diversity)
│   └── CUDA Support
└── Weighted Ensemble
```

## 🚀 핵심 전략

### 1. Advanced Preprocessing
```python
# 결측치 처리
- 카테고리형: -999 (특별값)
- 연속형: Median Imputation

# 데이터 샘플링
- Balanced Sampling (1:1)
- Imbalanced Sampling (1:1.5)
- Large Sample (3.5M)
```

### 2. Feature Engineering (42+ Features)

#### 상호작용 피처
- 곱셈, 나눗셈, 조화평균, 기하평균
- 예: `history_a_1 × history_b_21`

#### 통계 피처
- Mean, Std, Skewness, Kurtosis
- Quantiles (Q25, Q75, IQR)
- CV (Coefficient of Variation)
- MAD (Median Absolute Deviation)

#### 시간 인코딩
```python
# 다중 주기 sin/cos 변환
for period in [24, 12, 8, 6]:
    hour_sin = sin(2π × hour / period)
    hour_cos = cos(2π × hour / period)

# 피크 타임 indicator
- is_morning_rush (7-9시)
- is_lunch (11-13시)
- is_prime_time (19-22시)
```

#### 다항식 피처
- 제곱, 세제곱, 제곱근
- log1p, exp 변환

### 3. Model Training

**LightGBM 최적 파라미터**
```python
{
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 200,
    'learning_rate': 0.012,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.75,
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    'device': 'gpu',
    'num_boost_round': 2500
}
```

**앙상블 전략**
- 5-Fold × 5 Seeds = 25개 LightGBM 모델
- XGBoost 추가 (다양성 확보)
- 가중 평균 (Balanced sample 가중치 높임)

### 4. Calibration Strategy

6가지 Calibration 방법 적용:

1. **Supreme Optimal**: `0.248 + 0.504 × rank`
2. **Refined**: `0.2478 + 0.5042 × rank`
3. **Aggressive**: `0.252 + 0.496 × rank`
4. **Quantile Transform**: Uniform distribution
5. **Power Law**: `rank^0.95`
6. **Sigmoid Stretch**: `sigmoid(8 × (rank - 0.5))`

## 📁 프로젝트 구조

```
toss-ctr-prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── preprocessing/
│   │   ├── feature_engineering.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── lightgbm_trainer.py
│   │   ├── xgboost_trainer.py
│   │   └── ensemble.py
│   ├── utils/
│   │   ├── calibration.py
│   │   └── validation.py
│   └── main_training_pipeline.py
├── submissions/
│   ├── supreme_evolved_refined.csv
│   ├── final_push_v1.csv
│   └── enhanced_v2.csv
└── docs/
    ├── APPROACH.md          # 상세 접근 방법
    ├── FEATURE_ENGINEERING.md
    └── RESULTS_ANALYSIS.md
```

## 🔬 실험 및 결과

### 주요 실험

| 실험 | AUC Score | 개선점 | 비고 |
|------|-----------|--------|------|
| Baseline (22 features) | 0.3409 | - | 기본 모델 |
| + Feature Engineering | 0.3425 | +0.0016 | 28개 피처 |
| + Advanced FE (42+) | **0.3434** | **+0.0025** | **최종 베스트** |
| + Ensemble (25 models) | 0.3434 | +0.0000 | 안정성 향상 |

### Feature Importance (Top 10)

```
1. history_b_21      ████████████ 12.3%
2. history_a_1       ███████████  11.8%
3. feat_b_3          ██████████   10.2%
4. feat_c_8          ████████     8.7%
5. history_b_30      ███████      7.5%
6. inventory_id      ██████       6.8%
7. feat_a_1          █████        5.9%
8. age_group         ████         4.6%
9. hour              ████         4.3%
10. feat_b_1         ███          3.8%
```

## 💡 핵심 인사이트

### 성공 요인
1. **고급 피처 엔지니어링**: 단순 피처보다 상호작용/통계 피처가 효과적
2. **다양한 샘플링 전략**: Balanced, Imbalanced, Large 조합
3. **앙상블의 힘**: 단일 모델 대비 안정성 크게 향상
4. **Calibration 최적화**: Rank-based calibration이 가장 효과적

### 실패 및 교훈
1. **과도한 피처 추가**: 93개 피처 시 오히려 성능 하락 (-0.11)
2. **단순 앙상블의 한계**: 단순 평균보다 가중 평균이 효과적
3. **메모리 관리**: 대용량 데이터 처리 시 청킹 및 GC 필수

## 🚀 실행 방법

### 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 학습 실행
```bash
# 기본 학습
python src/main_training_pipeline.py

# GPU 가속 (CUDA 필요)
python src/main_training_pipeline.py --device gpu

# 앙상블 학습
python src/models/ensemble.py --n_models 25
```

### 예측 생성
```bash
python src/predict.py --model_path models/best_model.pkl
```

## 📈 성능 최적화

### GPU 활용
- LightGBM GPU 모드: 학습 속도 5-10배 향상
- XGBoost CUDA: 대용량 데이터 처리 효율화

### 메모리 최적화
```python
# Float32 사용
X = X.astype(np.float32)

# 청킹 처리
for chunk in pd.read_parquet('train.parquet', chunksize=100000):
    process_chunk(chunk)

# Garbage Collection
del train_df
gc.collect()
```

## 📚 참고 자료

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CTR Prediction Best Practices](https://arxiv.org/)

## 🤝 기여

이 프로젝트는 Dacon x 토스 CTR 예측 대회 참가작입니다.

## 📄 라이센스

MIT License

---

**개발 기간**: 2025.09 ~ 2025.10
**최종 업데이트**: 2025.10.13
