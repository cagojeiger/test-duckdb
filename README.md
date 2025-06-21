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

## 프로젝트 구조

```
src/
├── data_generation/          # Faker 기반 데이터 생성
│   ├── __init__.py
│   ├── text_embeddings.py    # 텍스트 임베딩 데이터 생성
│   ├── image_features.py     # 이미지 특징 벡터 생성
│   ├── recommendation_vectors.py  # 추천 시스템 벡터 생성
│   └── base_generator.py     # 공통 생성 로직
├── benchmarks/               # 성능 측정 및 분석
│   ├── __init__.py
│   ├── performance_metrics.py  # 성능 지표 수집
│   ├── query_executor.py     # 쿼리 실행 엔진
│   ├── result_analyzer.py    # 결과 분석 및 시각화
│   └── benchmark_runner.py   # 벤치마크 실행 관리
├── scenarios/                # 벡터 검색 시나리오
│   ├── __init__.py
│   ├── text_similarity.py    # 텍스트 유사도 검색
│   ├── image_matching.py     # 이미지 매칭
│   ├── recommendation.py     # 추천 시스템
│   └── base_scenario.py      # 시나리오 기본 클래스
├── utils/                    # DuckDB 관리 유틸리티
│   ├── __init__.py
│   ├── duckdb_manager.py     # DuckDB 연결 및 관리
│   ├── vector_operations.py  # 벡터 연산 헬퍼
│   └── config.py            # 설정 관리
├── tests/                    # 테스트 코드
│   ├── __init__.py
│   ├── test_data_generation.py
│   ├── test_benchmarks.py
│   └── test_scenarios.py
└── main.py                   # 메인 실행 스크립트
```

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
- **시각화 대시보드**: 인터랙티브 성능 분석 도구
- **실용 가이드**: 프로덕션 환경 적용 방안

## 시작하기

### 📋 요구사항
- Python 3.12+
- DuckDB + VSS 확장
- 최소 8GB RAM (대규모 실험용 16GB+ 권장)

### 🚀 설치 및 실행
```bash
# 저장소 클론
git clone https://github.com/cagojeiger/test-duckdb.git
cd test-duckdb

# 의존성 설치
pip install -r requirements.txt

# DuckDB VSS 확장 설치
python -c "import duckdb; duckdb.install_extension('vss')"

# 실험 실행
python main.py --experiment text_similarity --scale small
```

## 기여하기

이 프로젝트는 DuckDB 벡터 검색 성능에 대한 체계적인 연구를 목표로 합니다. 실험 설계 개선, 새로운 시나리오 추가, 성능 최적화 등 다양한 기여를 환영합니다.

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
