# GitHub 업로드 가이드

## 1. Git 초기화

```bash
cd portfolio_clean

# Git 초기화
git init

# .gitignore 확인 (이미 생성됨)
cat .gitignore
```

## 2. 파일 추가

```bash
# 모든 파일 추가
git add .

# 상태 확인
git status
```

## 3. 첫 커밋

```bash
git commit -m "Initial commit: Toss CTR Prediction Competition Project

- Complete ML pipeline for CTR prediction
- LightGBM + XGBoost ensemble
- Advanced feature engineering (42+ features)
- Best score: 0.3434805649 (AUC)
- Comprehensive documentation"
```

## 4. GitHub 레포지토리 생성

1. GitHub 접속: https://github.com
2. 'New repository' 클릭
3. Repository name: `toss-ctr-prediction`
4. Description: `토스 광고 클릭률 예측 AI 경진대회 (Dacon) - Machine Learning Pipeline`
5. Public 선택
6. README, .gitignore, license 체크 해제 (이미 있음)
7. 'Create repository' 클릭

## 5. Remote 추가 및 Push

```bash
# Remote 추가 (your-username을 본인 GitHub ID로 변경)
git remote add origin https://github.com/your-username/toss-ctr-prediction.git

# Branch 이름 확인/변경
git branch -M main

# Push
git push -u origin main
```

## 6. GitHub 페이지 설정 (선택)

### README 꾸미기
- 배지 추가: ![Python](https://img.shields.io/badge/python-3.12-blue)
- 이미지 추가: 결과 그래프, 아키텍처 다이어그램
- GIF 추가: 데모 실행 화면

### Topics 추가
Repository 설정에서 Topics 추가:
- `machine-learning`
- `lightgbm`
- `xgboost`
- `ctr-prediction`
- `feature-engineering`
- `ensemble-learning`
- `kaggle-competition`

## 7. 포트폴리오 강화 팁

### 추가할 내용
1. **결과 시각화**
   - Feature importance 그래프
   - Learning curve
   - Confusion matrix
   - ROC curve

2. **Jupyter Notebook**
   - EDA (탐색적 데이터 분석)
   - 실험 과정
   - 결과 분석

3. **Docker 지원**
   ```dockerfile
   FROM python:3.12
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "src/main_training_pipeline.py"]
   ```

4. **CI/CD**
   - GitHub Actions로 코드 품질 체크
   - 자동 테스트

### README 개선
```markdown
## Highlights
- 🏆 Top 3 Score: 0.3434 (AUC)
- 🚀 42+ Engineered Features
- ⚡ GPU-Accelerated Training
- 📊 25-Model Ensemble
```

## 8. 커밋 메시지 가이드

```bash
# 기능 추가
git commit -m "feat: Add advanced feature engineering module"

# 버그 수정
git commit -m "fix: Correct calibration formula"

# 문서화
git commit -m "docs: Update README with detailed approach"

# 리팩토링
git commit -m "refactor: Optimize memory usage in data loading"
```

## 9. 완성된 README 예시

```markdown
# 토스 광고 클릭률 예측 AI

[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)]()
[![Score](https://img.shields.io/badge/AUC-0.3434-brightgreen)]()

> Dacon x Toss NEXT ML Challenge 참가 프로젝트

[데모] [문서] [결과 분석]

## Quick Start

\`\`\`bash
pip install -r requirements.txt
python src/supreme_evolved_training.py
\`\`\`

## 주요 성과
- 🏆 **최고 점수**: 0.3434 (AUC)
- 📊 **앙상블**: 25개 모델
- ⚡ **GPU 가속**: 5-10배 빠른 학습
- 🔧 **피처**: 42+ 고급 엔지니어링
```

## 10. 완료 체크리스트

- [ ] Git 초기화
- [ ] 모든 파일 커밋
- [ ] GitHub 레포지토리 생성
- [ ] Remote 추가 및 Push
- [ ] README 확인 (렌더링 체크)
- [ ] Topics 추가
- [ ] License 확인
- [ ] .gitignore 확인 (대용량 파일 제외)
- [ ] 링크 동작 확인
- [ ] 스타일 통일성 확인

완료! 🎉
