# DuckDB VSS 벤치마킹 프로젝트 컨텍스트

## 프로젝트 개요

이 프로젝트는 DuckDB의 VSS (Vector Similarity Search) 확장을 사용하여 한국어 텍스트 벡터 검색 성능을 체계적으로 분석하는 벤치마킹 시스템입니다. 함수형 프로그래밍 패러다임을 기반으로 48가지 실험 구성을 통해 벡터 검색 성능을 측정합니다.

## 아키텍처 설계

### 함수형 프로그래밍 기반 설계
- **Pure Layer**: 부작용이 없는 비즈니스 로직 (데이터 생성, 변환, 계산)
- **Effect Layer**: 모나딕 타입으로 래핑된 IO 작업 (데이터베이스, 파일 I/O, 메트릭 수집)
- **Pipeline Layer**: 단순한 함수들을 조합하여 복잡한 워크플로우 구성

### 실험 매트릭스 (48가지 조합)
- **데이터 스케일**: 10K, 100K, 250K 벡터
- **벡터 차원**: 128, 256, 512, 1024
- **검색 타입**: Pure Vector Search, Hybrid (Vector + BM25)
- **필터 조건**: 메타데이터 필터링 유/무

## 구현된 컴포넌트

### Phase 1: 함수형 프로그래밍 기반 구조 (완료)

#### 타입 시스템 (`src/types/`)
- **core.py**: 핵심 데이터 타입 정의 (ExperimentConfig, ExperimentResult, Metrics 등)
- **aliases.py**: 타입 별칭 정의 (Vector, DocumentId 등)
- **monads.py**: 함수형 프로그래밍을 위한 모나드 구현 (IO, Either, Maybe)

#### Pure Functions (`src/pure/`)
- **generators/**: 데이터 생성 함수들
  - `text.py`: 한국어 텍스트 생성 (Faker 사용)
  - `vectors.py`: 벡터 생성 및 변환
- **calculators/**: 계산 함수들
  - `distances.py`: 벡터 거리 계산 (코사인, 유클리드, 맨하탄)
  - `metrics.py`: 성능 메트릭 계산 (Recall@K, MRR, QPS)

#### Effect Management (`src/effects/`)
- **db/**: 데이터베이스 작업 래핑
  - `connection.py`: DuckDB 연결 관리
  - `tables.py`: 테이블 생성 및 관리
  - `search.py`: 벡터 검색 작업

### Phase 2: DuckDB VSS 통합 (완료)

#### DuckDB VSS 확장 통합
- VSS 확장 설치 및 설정
- HNSW 인덱스 생성 및 관리
- 벡터 검색 쿼리 최적화
- 한국어 텍스트 데이터 처리

#### 테스트 및 검증
- DuckDB VSS 설치 검증 스크립트
- 통합 테스트 구현
- 벡터 검색 기능 검증

### Phase 3: 파이프라인 조합 레이어 (완료)

#### 파이프라인 구성 (`src/pipelines/`)
- **composition.py**: 함수 조합 유틸리티 (compose, pipe, parallel_map)
- **experiments.py**: 실험 실행 파이프라인
  - 데이터 생성 → 벡터화 → 인덱싱 → 검색 → 메트릭 계산

#### 실험 워크플로우
- 48가지 실험 구성 매트릭스 생성
- 배치 처리를 통한 메모리 효율적 실행
- 결과 수집 및 분석 파이프라인

### Phase 4A: 실험 워크플로우 핵심 기능 (완료)

#### CLI 실험 실행기 (`src/runners/experiment_runner.py`)
- **배치 처리**: 메모리 사용량 관리를 위한 구성 가능한 배치 실행 (기본값: 4)
- **재시작 기능**: 체크포인트에서 중단된 실험 재개
- **필터링 옵션**: 데이터 스케일, 차원, 검색 타입별 실험 필터링
- **리소스 관리**: 메모리 인식 실행을 위한 리소스 모니터링 통합
- **CLI 인터페이스**: 포괄적인 인수 파싱을 통한 완전한 명령줄 인터페이스

**주요 기능:**
```bash
# 모든 48개 실험 실행
python -m src.runners.experiment_runner --all

# 체크포인트에서 재시작
python -m src.runners.experiment_runner --all --resume

# 실험 필터링
python -m src.runners.experiment_runner --data-scale small --dimensions 128 256
```

#### 체크포인트 관리 시스템 (`src/runners/checkpoint.py`)
- **진행상황 지속성**: pickle과 JSON을 사용하여 실험 결과와 진행상황을 디스크에 저장
- **재시작 지원**: 완료된 실험을 추적하여 중단된 실행 재개 가능
- **메타데이터 추적**: 완료율과 시간 추정을 포함한 체크포인트 메타데이터 유지
- **내보내기 기능**: 분석을 위한 사람이 읽을 수 있는 JSON 형식으로 결과 내보내기
- **오류 복구**: 손상된 체크포인트 파일을 우아하게 처리

**주요 기능:**
- 각 배치 후 자동 결과 저장
- 완료율을 통한 진행상황 추적
- 남은 시간 추정 계산
- 분석을 위한 JSON 내보내기

#### 리소스 모니터링 시스템 (`src/runners/monitoring.py`)
- **메모리 추적**: 실시간 시스템 및 프로세스 메모리 사용량 모니터링
- **백그라운드 모니터링**: 실험 중 별도 스레드에서 모니터링 실행
- **메모리 알림**: 메모리 임계값 위반 추적 및 알림
- **리소스 정리**: 실험 배치 간 정리 메커니즘 제공
- **리소스 요약**: 상세한 리소스 사용량 보고서 생성

**주요 기능:**
- 실시간 메모리 및 CPU 모니터링
- 구성 가능한 메모리 임계값
- 배치 간 자동 정리
- 리소스 사용량 히스토리 추적

## 테스트 커버리지

### 단위 테스트 (총 27개)
- **Pure Functions**: 12개 테스트 (generators, calculators)
- **Effects**: 9개 테스트 (database operations)
- **Pipelines**: 6개 테스트 (composition, experiments)
- **Runners**: 27개 테스트 (checkpoint, monitoring, experiment_runner)

### 벤치마크 테스트
- **pytest-benchmark** 사용
- 성능 회귀 방지
- 메모리 사용량 프로파일링

## 기술 스택

### 핵심 의존성
- **DuckDB**: 메인 데이터베이스 엔진
- **DuckDB VSS**: 벡터 유사도 검색 확장
- **Faker**: 한국어 텍스트 데이터 생성
- **NumPy**: 벡터 연산
- **Pandas**: 데이터 조작
- **psutil**: 시스템 리소스 모니터링

### 개발 도구
- **uv**: Python 패키지 관리
- **pytest**: 테스트 프레임워크
- **pytest-benchmark**: 성능 벤치마킹
- **mypy**: 정적 타입 검사
- **ruff**: 코드 포매팅 및 린팅
- **pre-commit**: Git 훅 관리

## 프로젝트 구조

```
src/
├── types/              # 타입 정의 (frozen dataclasses)
│   ├── core.py        # 핵심 데이터 타입
│   ├── aliases.py     # 타입 별칭
│   └── monads.py      # 함수형 모나드
├── pure/               # 순수 함수 (부작용 없음)
│   ├── generators/    # 데이터 생성
│   │   ├── text.py    # 한국어 텍스트 생성
│   │   └── vectors.py # 벡터 생성
│   └── calculators/   # 계산 함수
│       ├── distances.py # 벡터 거리 계산
│       └── metrics.py   # 성능 메트릭
├── effects/            # 부작용 관리
│   └── db/            # 데이터베이스 IO 작업
│       ├── connection.py # 연결 관리
│       ├── tables.py     # 테이블 관리
│       └── search.py     # 검색 작업
├── pipelines/          # 함수 조합 파이프라인
│   ├── composition.py  # 조합 유틸리티
│   └── experiments.py  # 실험 파이프라인
└── runners/            # 메인 진입점
    ├── experiment_runner.py # CLI 실험 실행기
    ├── checkpoint.py        # 체크포인트 관리
    └── monitoring.py        # 리소스 모니터링

tests/                  # 테스트 코드
├── pure/              # 순수 함수 테스트
├── effects/           # 부작용 테스트
├── pipelines/         # 파이프라인 테스트
└── runners/           # 실행기 테스트

benchmarks/            # 성능 벤치마크
plan/                  # 설계 문서
```

## 함수형 프로그래밍 원칙

### 불변성 (Immutability)
- 모든 데이터 구조에 `@dataclass(frozen=True)` 사용
- 상태 변경 대신 새로운 객체 생성
- 예측 가능하고 안전한 동시성

### 순수 함수 (Pure Functions)
- 비즈니스 로직을 I/O 작업과 분리
- 동일한 입력에 대해 항상 동일한 출력
- 부작용 없음으로 테스트 용이성 확보

### 부작용 처리 (Effect Handling)
- 모든 부작용을 IO 모나드로 래핑
- 예외 대신 Either 타입으로 오류 처리
- 명시적인 부작용 관리

### 함수 조합 (Function Composition)
- `compose`와 `pipe`를 사용한 파이프라인 구축
- 작은 함수들을 조합하여 복잡한 워크플로우 생성
- 재사용 가능하고 테스트 가능한 컴포넌트

## 성능 최적화 고려사항

### DuckDB VSS 최적화
- HNSW 파라미터 튜닝 (ef_construction, ef_search, M)
- 메모리 사용량 신중한 모니터링 (인덱스는 RAM 상주)
- 병렬 처리를 위한 parallel_map 사용
- Reader 모나드로 래핑된 연결 풀링

### 메모리 관리
- 배치 처리를 통한 메모리 사용량 제어
- 실험 간 리소스 정리
- 메모리 임계값 모니터링
- 가비지 컬렉션 최적화

## 현재 상태 및 다음 단계

### 완료된 작업
- ✅ 함수형 프로그래밍 기반 아키텍처
- ✅ DuckDB VSS 통합
- ✅ 파이프라인 조합 레이어
- ✅ Phase 4A 핵심 기능 (CLI 실행기, 체크포인트, 모니터링)
- ✅ 포괄적인 테스트 스위트 (27개 단위 테스트)
- ✅ 타입 안전성 및 코드 품질 보장

### 다음 단계 계획

#### Phase 4B: 고급 기능
- 병렬 실험 실행
- 분산 처리 지원
- 고급 리소스 관리
- 실시간 진행상황 대시보드

#### Phase 5: 분석 및 시각화
- 결과 분석 도구
- 성능 시각화 대시보드
- 통계 분석 및 보고서 생성
- 비교 분석 도구

#### Phase 6: 프로덕션 최적화
- 성능 최적화
- 확장성 개선
- 배포 자동화
- 모니터링 및 알림 시스템

## 실행 방법

### 환경 설정
```bash
# 의존성 설치
uv sync

# DuckDB VSS 확장 설치
python test_duckdb_vss_installation.py

# 테스트 실행
pytest tests/ -v

# 타입 검사
mypy src/ --show-error-codes

# 코드 포매팅
ruff format .
```

### 실험 실행
```bash
# 모든 48개 실험 실행
python -m src.runners.experiment_runner --all

# 특정 조건으로 필터링
python -m src.runners.experiment_runner --data-scale small --dimensions 128 256

# 체크포인트에서 재시작
python -m src.runners.experiment_runner --all --resume

# 도움말
python -m src.runners.experiment_runner --help
```

## 주요 성과

1. **체계적인 아키텍처**: 함수형 프로그래밍 원칙을 따른 확장 가능한 설계
2. **완전한 테스트 커버리지**: 27개 단위 테스트로 모든 핵심 기능 검증
3. **타입 안전성**: mypy를 통한 엄격한 타입 검사
4. **실용적인 도구**: CLI를 통한 사용자 친화적 인터페이스
5. **견고한 오류 처리**: 체크포인트와 복구 메커니즘
6. **성능 모니터링**: 실시간 리소스 추적 및 최적화

이 프로젝트는 DuckDB VSS의 성능 특성을 체계적으로 분석할 수 있는 견고한 기반을 제공하며, 함수형 프로그래밍의 장점을 실제 벤치마킹 시나리오에 적용한 성공적인 사례입니다.
