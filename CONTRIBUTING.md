# 기여 가이드

이 프로젝트에 기여해주셔서 감사합니다! 이 문서는 코드 기여 시 따라야 할 가이드라인을 제공합니다.

## 코드 스타일

### Python 스타일 가이드

- PEP 8 스타일 가이드 준수
- Black formatter 사용 권장
- 최대 줄 길이: 100자
- 들여쓰기: 4 spaces

### 코드 예시

```python
def train_model(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    params: Dict[str, Any]
) -> lgb.Booster:
    """
    LightGBM 모델을 학습합니다.

    Args:
        data: 학습 데이터프레임
        features: 피처 컬럼 리스트
        target: 타겟 컬럼명
        params: LightGBM 하이퍼파라미터

    Returns:
        학습된 LightGBM Booster 모델
    """
    X = data[features]
    y = data[target]

    lgb_train = lgb.Dataset(X, y)
    model = lgb.train(params, lgb_train)

    return model
```

## Git 워크플로우

### 브랜치 전략

- `main`: 안정적인 메인 브랜치
- `develop`: 개발 브랜치
- `feature/기능명`: 새 기능 개발
- `fix/버그명`: 버그 수정
- `experiment/실험명`: 새로운 실험

### 커밋 메시지

Conventional Commits 형식 사용:

```
<타입>: <제목>

<본문>

<푸터>
```

**타입**:
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 수정
- `refactor`: 코드 리팩토링
- `test`: 테스트 추가/수정
- `perf`: 성능 개선
- `chore`: 기타 변경사항

**예시**:
```
feat: Add polynomial features to feature engineering

- history_b_21의 제곱, 세제곱, 제곱근 피처 추가
- feat_b_3와 feat_c_8에 대한 로그 변환 추가
- 성능 +0.0003 개선 확인

Closes #12
```

## 풀 리퀘스트

### PR 체크리스트

- [ ] 코드가 PEP 8 스타일 가이드를 준수하는가?
- [ ] 모든 테스트가 통과하는가?
- [ ] 새로운 기능에 대한 테스트가 추가되었는가?
- [ ] 문서가 업데이트되었는가?
- [ ] 커밋 메시지가 명확한가?

### PR 템플릿

```markdown
## 변경 사항 설명
무엇을 변경했는지 설명해주세요.

## 동기 및 맥락
왜 이 변경이 필요한가요?

## 테스트 방법
어떻게 테스트했나요?

## 체크리스트
- [ ] 코드 스타일 가이드 준수
- [ ] 테스트 추가/통과
- [ ] 문서 업데이트

## 관련 이슈
Closes #이슈번호
```

## 코드 리뷰

### 리뷰 기준

1. **코드 품질**
   - 가독성
   - 유지보수성
   - 재사용성

2. **성능**
   - 메모리 효율성
   - 계산 효율성

3. **테스트**
   - 테스트 커버리지
   - 엣지 케이스 처리

4. **문서화**
   - Docstring 작성
   - README 업데이트

## 이슈 리포트

### 버그 리포트 템플릿

```markdown
## 버그 설명
명확하고 간결한 버그 설명

## 재현 방법
1. '...'로 이동
2. '....'를 클릭
3. '....' 확인
4. 에러 발생

## 예상 동작
무엇이 일어나야 하는가?

## 실제 동작
무엇이 일어났는가?

## 환경
- OS: [예: Windows 10]
- Python: [예: 3.12]
- 패키지 버전: [예: lightgbm==4.0.0]
```

## 연락처

- **이슈**: GitHub Issues
- **이메일**: noreply@example.com

---

**작성일**: 2025-10-13
