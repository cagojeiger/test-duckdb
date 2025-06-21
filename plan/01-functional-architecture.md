# 함수형 아키텍처 설계

## 개요

이 문서는 DuckDB 벡터 검색 벤치마킹 프로젝트를 함수형 프로그래밍 패러다임으로 설계하는 방법을 설명합니다. 순수 함수와 부수 효과를 명확히 분리하여 테스트 가능하고 유지보수가 쉬운 코드를 작성하는 것이 목표입니다.

## 핵심 원칙

### 1. 불변성 (Immutability)
- 모든 데이터 구조는 생성 후 변경 불가
- 상태 변경이 필요한 경우 새로운 객체 생성
- Python의 `@dataclass(frozen=True)`, `namedtuple`, `tuple` 활용

### 2. 순수 함수 (Pure Functions)
- 같은 입력에 대해 항상 같은 출력
- 외부 상태를 읽거나 변경하지 않음
- 부수 효과(side effects) 없음

### 3. 함수 합성 (Function Composition)
```python
# 작은 함수들을 조합하여 복잡한 로직 구성
pipeline = compose(
    validate_config,
    generate_data,
    transform_to_vectors,
    run_benchmark,
    analyze_results
)
```

### 4. 부수 효과 격리 (Effect Isolation)
- DB 작업, 파일 I/O, 네트워크 등은 Effect 타입으로 래핑
- 비즈니스 로직과 I/O 로직 분리
- 테스트 시 Mock 주입 용이

## 레이어 구조

### Pure Layer (순수 함수 레이어)
```
pure/
├── generators/     # 데이터 생성 로직
├── transformers/   # 데이터 변환 로직
├── calculators/    # 계산 및 분석 로직
└── validators/     # 검증 로직
```

### Effect Layer (부수 효과 레이어)
```
effects/
├── db/            # 데이터베이스 작업
├── io/            # 파일 입출력
├── metrics/       # 성능 측정
└── logging/       # 로깅
```

### Pipeline Layer (파이프라인 레이어)
```
pipelines/
├── experiments/   # 실험 실행 파이프라인
├── analysis/      # 분석 파이프라인
└── reporting/     # 리포트 생성 파이프라인
```

## 데이터 플로우

```
Config → Pure Functions → Effects → Results
         ↓                ↓         ↓
         불변 데이터       IO 모나드   불변 결과
```

## 에러 처리 전략

### Either 타입 활용
```python
# 성공/실패를 타입으로 표현
Result[T] = Either[Error, T]

# 체이닝 가능
result = (
    validate_config(config)
    .flat_map(generate_data)
    .flat_map(run_experiment)
    .map(analyze_results)
)
```

### 복구 가능한 에러
- 재시도 로직을 함수로 구현
- 대체 경로(fallback) 제공
- 부분 실패 허용

## 병렬 처리 설계

### 불변성을 활용한 안전한 병렬화
```python
# 48개 실험 조합을 병렬로 실행
experiments = generate_all_experiments()
results = parallel_map(run_single_experiment, experiments)
```

### 리소스 풀 관리
- DB 연결 풀을 Reader 모나드로 전달
- 각 실험은 독립적으로 실행
- 결과는 불변 객체로 수집

## 테스트 전략

### 순수 함수 테스트
- 입력과 출력만 검증
- Property-based testing 활용
- 빠른 실행 속도

### Effect 테스트
- Mock 객체로 I/O 대체
- 격리된 환경에서 실행
- 실제 DB는 통합 테스트에서만 사용

## 모듈 의존성

```
pure → types
effects → types, pure
pipelines → types, pure, effects
main → pipelines
```

순수 함수는 타입만 의존하고, 효과는 순수 함수를 사용할 수 있으며, 파이프라인은 모든 레이어를 조합합니다.
