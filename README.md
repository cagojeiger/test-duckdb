# DuckDB 벡터 검색 벤치마킹 프로젝트

## 프로젝트 개요

이 프로젝트는 **DuckDB VSS(Vector Similarity Search) 확장**을 활용한 텍스트 벡터 검색의 성능 특성을 체계적으로 분석하고 벤치마킹하는 Python 기반 실험 플랫폼입니다.

## 목적

### 🎯 주요 목표
- **성능 벤치마킹**: 다양한 데이터 규모와 벡터 차원에서의 검색 성능 측정
- **최적화 가이드**: HNSW 인덱스 파라미터 튜닝 및 최적 설정 도출
- **확장성 분석**: 메모리 제약과 데이터 크기에 따른 성능 한계 파악
- **하이브리드 검색**: 벡터 검색과 BM25 텍스트 검색의 조합 성능 평가
- **실용적 권장사항**: 프로덕션 환경에서의 DuckDB VSS 활용 가이드라인 제시

### 🔬 실험 범위
- **48가지 실험 조합**: 3개 데이터 크기 × 4개 벡터 차원 × 2개 검색 유형 × 2개 필터 조건
- **5개 성능 영역**: 벌크 업로드, 인덱싱, 검색 속도, 하이브리드 검색, 필터링 검색
- **500+ 성능 데이터 포인트**: 종합적인 성능 특성 분석
- **한국어 텍스트 데이터**: Faker 라이브러리를 활용한 현실적인 텍스트 데이터 생성

## 기술 스택

### 🛠️ 핵심 기술
- **데이터베이스**: DuckDB + VSS 확장 (HNSW 인덱싱)
- **언어**: Python 3.12+
- **병렬 처리**: concurrent.futures (멀티프로세싱)
- **데이터 생성**: Faker (한국어 로케일)
- **성능 측정**: psutil, time, DuckDB EXPLAIN ANALYZE
- **시각화**: matplotlib, seaborn, plotly
- **데이터 처리**: pandas, numpy

### 📊 측정 지표
- **처리량**: 초당 벡터 삽입/검색 수 (vectors/sec, QPS)
- **응답 시간**: 쿼리 실행 시간 (ms)
- **정확도**: Recall@K (K=1,5,10)
- **리소스 사용량**: CPU, 메모리 사용률
- **인덱스 효율성**: 구축 시간, 크기, 압축률



## 실험 설계

### 📈 데이터 규모
- **소규모**: 10,000 벡터 (~40MB)
- **중규모**: 100,000 벡터 (~400MB)
- **대규모**: 250,000 벡터 (~1GB)

### 🔢 벡터 차원
- **128차원**: 경량 임베딩 모델
- **256차원**: 중간 성능 모델
- **512차원**: 고성능 모델
- **1024차원**: 최고 성능 모델

### 🔍 검색 유형
- **순수 벡터 검색**: VSS 확장만 사용
- **하이브리드 검색**: 벡터 + BM25 텍스트 검색

### 🎛️ 필터 조건
- **필터 없음**: 전체 데이터셋 검색
- **필터 적용**: 메타데이터 조건부 검색

## 예상 결과

### 🎯 핵심 발견사항 (가설)
1. **차원의 저주**: 512차원 이후 성능 대비 정확도 향상 미미
2. **메모리 벽**: 100K 벡터 이상에서 급격한 성능 저하
3. **하이브리드 효과**: 검색 품질 20% 향상, 성능 30% 저하
4. **필터링 최적화**: 선택도 10% 이하에서 효과적

### 📊 결과물
- **성능 벤치마크 리포트**: 상세한 분석 및 권장사항
- **최적화 가이드**: HNSW 파라미터 튜닝 가이드
- **병렬 실행 성능**: 멀티프로세싱을 통한 실험 시간 단축
- **시각화 대시보드**: 인터랙티브 성능 분석 도구 (예정)
- **실용 가이드**: 프로덕션 환경 적용 방안

## 🚀 빠른 시작 가이드

### 📋 시스템 요구사항
- **Python**: 3.12 이상
- **메모리**: 최소 8GB RAM (대규모 실험용 16GB+ 권장)
- **CPU**: 멀티코어 (병렬 실행 시 성능 향상)
- **저장공간**: 약 2GB (실험 결과 포함)

### ⚡ 1단계: 환경 설정
```bash
# 1. 저장소 클론
git clone https://github.com/cagojeiger/test-duckdb.git
cd test-duckdb

# 2. 의존성 설치 (uv 권장)
uv sync

# 또는 pip 사용
pip install -r requirements.txt

# 3. DuckDB VSS 확장 설치 확인
uv run python test_duckdb_vss_installation.py
```

**설치 성공 확인:**
```
✅ DuckDB version: 0.9.2
✅ VSS extension loaded successfully
✅ Vector operations working
```

### 🧪 2단계: 시스템 테스트
```bash
# 전체 테스트 실행 (118개 테스트)
uv run python -m pytest tests/ -v

# 핵심 기능만 빠르게 테스트
uv run python -m pytest tests/pure/ -v
```

**테스트 성공 시 출력:**
```
=================== 118 passed in 45.2s ===================
```

### 🔬 3단계: 실험 실행

#### 빠른 테스트 (소규모 데이터)
```bash
# 소규모 데이터로 빠른 테스트 (약 5분)
uv run python -m src.runners.experiment_runner \
  --data-scale small \
  --dimensions 128 256 \
  --parallel

# 특정 검색 타입만 테스트
uv run python -m src.runners.experiment_runner \
  --data-scale small \
  --search-type vector \
  --parallel
```

#### 전체 벤치마크 실행
```bash
# 모든 48개 실험 조합 실행 (약 2-4시간)
uv run python -m src.runners.experiment_runner --all --parallel

# 커스텀 병렬 설정 (메모리 8GB, 워커 6개)
uv run python -m src.runners.experiment_runner \
  --all --parallel \
  --workers 6 \
  --max-memory 8000
```

#### 실험 재시작 및 모니터링
```bash
# 체크포인트에서 재시작
uv run python -m src.runners.experiment_runner --all --resume

# 실험 진행상황 모니터링
uv run python -m src.runners.monitoring --experiment-dir results/
```

### 📊 4단계: 결과 확인 및 분석

#### 실험 결과 위치
```
results/
├── experiment_results_20241221_143022.json    # 원시 실험 데이터
├── performance_summary.json                   # 성능 요약
├── checkpoints/                              # 체크포인트 파일들
└── analysis/                                 # 분석 결과
    ├── performance_analysis.json
    ├── trend_analysis.json
    └── statistical_summary.json
```

#### 결과 분석 실행
```bash
# 성능 분석 실행
uv run python -m src.runners.analysis_runner \
  --input results/experiment_results_*.json \
  --output analysis/

# 시각화 대시보드 실행
uv run python -m src.web.dashboard --port 8080
```

#### 주요 성능 지표 확인
```bash
# 성능 요약 보기
cat results/performance_summary.json | jq '.summary'

# 차원별 성능 비교
cat analysis/performance_analysis.json | jq '.dimension_performance'

# 검색 타입별 정확도 비교
cat analysis/performance_analysis.json | jq '.search_type_performance'
```

### 📈 5단계: 결과 해석

#### 성능 메트릭 이해
- **삽입 성능**: `vectors_per_second` (높을수록 좋음)
- **검색 속도**: `query_time_ms` (낮을수록 좋음)
- **검색 정확도**: `recall_at_k` (높을수록 좋음, 0-1 범위)
- **메모리 사용량**: `memory_usage_mb`

#### 실험 결과 예시
```json
{
  "config": {
    "data_scale": "small",
    "dimension": 256,
    "search_type": "vector"
  },
  "performance": {
    "insert_vectors_per_second": 2847.3,
    "index_build_time_seconds": 12.4,
    "query_time_ms": 3.2,
    "recall_at_10": 0.94
  }
}
```

### 🛠️ 고급 사용법

#### CLI 옵션 전체 목록
```bash
uv run python -m src.runners.experiment_runner --help
```

**주요 옵션:**
- `--data-scale`: `small`, `medium`, `large`
- `--dimensions`: `128`, `256`, `512`, `1024` (공백으로 구분)
- `--search-type`: `vector`, `hybrid`
- `--filter-condition`: `none`, `with_filter` (참고: CLI에서 확인 필요)
- `--parallel`: 병렬 실행 활성화
- `--workers N`: 워커 프로세스 수 (기본값: CPU 코어 수)
- `--max-memory N`: 메모리 임계값 MB (기본값: 6000)
- `--resume`: 체크포인트에서 재시작
- `--output-dir`: 결과 저장 디렉토리

#### 실험 조합 필터링 예시
```bash
# 128차원과 256차원만 테스트
uv run python -m src.runners.experiment_runner \
  --dimensions 128 256 --parallel

# 중규모 데이터 + 하이브리드 검색만
uv run python -m src.runners.experiment_runner \
  --data-scale medium \
  --search-type hybrid \
  --parallel

# 필터 조건 없는 실험만
uv run python -m src.runners.experiment_runner \
  --filter-condition none \
  --parallel
```

### 🔍 문제 해결

#### 일반적인 문제들

**1. VSS 확장 로드 실패**
```bash
# DuckDB 버전 확인
uv run python -c "import duckdb; print(duckdb.__version__)"

# VSS 확장 수동 설치
uv run python -c "
import duckdb
conn = duckdb.connect()
conn.execute('INSTALL vss')
conn.execute('LOAD vss')
print('VSS extension loaded successfully')
"
```

**2. 메모리 부족 오류**
```bash
# 메모리 사용량 모니터링
uv run python -m src.runners.monitoring --memory-only

# 워커 수 줄이기
uv run python -m src.runners.experiment_runner \
  --all --parallel --workers 2 --max-memory 4000
```

**3. 실험 중단 후 재시작**
```bash
# 체크포인트 확인
ls -la checkpoints/

# 특정 체크포인트에서 재시작
uv run python -m src.runners.experiment_runner \
  --resume --checkpoint-dir checkpoints/
```

#### 성능 최적화 팁

**시스템 리소스 최적화:**
- CPU 코어 수에 맞춰 `--workers` 설정
- 사용 가능한 메모리의 70% 정도로 `--max-memory` 설정
- SSD 사용 시 더 빠른 I/O 성능

**실험 시간 단축:**
- 소규모 데이터로 먼저 테스트
- 특정 차원이나 검색 타입만 선택적 실행
- 병렬 실행 활용

### 📚 추가 문서

- **[실험 설계](docs/02-experimental-design.md)**: 48가지 실험 조합 상세 설명
- **[구현 가이드](plan/07-implementation-guide.md)**: 함수형 프로그래밍 구현 방법
- **[워크플로우](plan/06-experiment-workflow.md)**: 실험 실행 단계별 설명
- **[아키텍처](plan/01-functional-architecture.md)**: 시스템 구조 및 설계 원칙

### 🤝 기여하기

이 프로젝트는 DuckDB 벡터 검색 성능에 대한 체계적인 연구를 목표로 합니다.

**기여 방법:**
- 새로운 실험 시나리오 제안
- 성능 최적화 아이디어
- 버그 리포트 및 수정
- 문서 개선

**개발 환경 설정:**
```bash
# 개발용 의존성 설치
uv sync --dev

# 코드 품질 검사
uv run python -m mypy src/
uv run python -m pytest tests/ --cov=src/

# 커밋 전 검사
pre-commit run --all-files
```

### 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

**💡 도움이 필요하신가요?**
- 이슈 생성: [GitHub Issues](https://github.com/cagojeiger/test-duckdb/issues)
- 실험 결과 공유 및 토론 환영!
