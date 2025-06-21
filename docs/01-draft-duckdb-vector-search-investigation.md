# DuckDB 벡터 검색 기능 조사 보고서

## 개요

이 문서는 DuckDB의 VSS(Vector Similarity Search) 확장을 통한 벡터 검색 기능에 대한 포괄적인 조사 결과를 정리합니다. DuckDB VSS 확장은 HNSW(Hierarchical Navigable Small Worlds) 인덱싱을 통해 벡터 유사도 검색을 가속화하는 실험적 확장입니다.

## DuckDB VSS 확장 소개

### 주요 특징
- **HNSW 인덱스 지원**: 벡터 유사도 검색 가속화를 위한 계층적 탐색 가능한 소규모 세계 인덱싱
- **고정 크기 ARRAY 타입**: DuckDB의 ARRAY 데이터 타입을 사용한 벡터 저장
- **다중 거리 메트릭**: L2(유클리드), 코사인, 내적 거리 지원
- **실험적 상태**: 현재 실험적 기능으로 일부 제한사항 존재

### 설치 및 로드
```sql
INSTALL vss;
LOAD vss;
```

## 기술 스펙 및 제한사항

### 데이터 타입 제약
- **지원 타입**: FLOAT(32비트) 벡터만 지원
- **벡터 형식**: `FLOAT[n]` 형태의 고정 크기 배열
- **예시**: `[1.0, 2.0, 3.0]::FLOAT[3]`

### 메모리 제약사항
- **RAM 의존성**: 인덱스가 완전히 RAM에 로드되어야 함
- **버퍼 관리 없음**: DuckDB의 버퍼 관리 시스템을 사용하지 않음
- **메모리 요구량**: 벡터 차원과 데이터 크기에 비례하여 메모리 사용량 증가
- **실질적 제한**: 사용 가능한 시스템 메모리에 의해 최대 벡터 차원과 데이터 크기 제한

### 벡터 차원 제한
- **명시적 최대값**: 공식 문서에서 명시적인 최대 차원 제한 없음
- **실질적 제한**: 메모리 제약에 의해 제한됨
- **권장 차원**: 64-1024 차원 범위에서 실용적 사용
- **테스트된 차원**: 문서 예제에서 3차원 벡터 사용

### 지속성 및 안정성
- **실험적 지속성**: `SET hnsw_enable_experimental_persistence=true`로 활성화
- **데이터 손실 위험**: 실험적 기능으로 데이터 손실 가능성 존재
- **권장사항**: 프로덕션 환경에서 신중한 사용 필요

### 업데이트 및 삭제 제한
- **삭제 처리**: 삭제된 항목은 마킹만 되고 즉시 제거되지 않음
- **압축 필요**: 성능 유지를 위해 주기적 압축 작업 필요
- **벌크 로딩**: 인덱스 생성 전 대량 데이터 로딩이 더 효율적

## HNSW 인덱스 파라미터

### 인덱스 생성 구문
```sql
CREATE INDEX my_hnsw_index ON my_vector_table USING HNSW(vec);
```

### 주요 파라미터 및 기본값

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `ef_construction` | 128 | 인덱스 구축 시 후보 정점 수 |
| `ef_search` | 64 | 검색 시 후보 정점 수 |
| `M` | 16 | 각 정점의 최대 이웃 수 |
| `M0` | 32 (2*M) | 0레벨 기본 연결성 |

### 런타임 설정
```sql
-- 검색 파라미터 오버라이드
SET hnsw_ef_search = 100;

-- 실험적 지속성 활성화
SET hnsw_enable_experimental_persistence = true;
```

### 파라미터 튜닝 가이드
- **ef_construction**: 높을수록 인덱스 품질 향상, 구축 시간 증가
- **ef_search**: 높을수록 검색 정확도 향상, 검색 시간 증가
- **M**: 높을수록 연결성 증가, 메모리 사용량 증가
- **M0**: 기본적으로 M의 2배 설정 권장

## 거리 메트릭 및 함수

### 지원 거리 메트릭

| 메트릭 | 함수 | 설명 | 사용 사례 |
|--------|------|------|----------|
| `l2sq` | `array_distance(vec1, vec2)` | 유클리드 거리 (L2-norm 제곱) | 이미지 특징, 일반적 유사도 |
| `cosine` | `array_cosine_distance(vec1, vec2)` | 코사인 유사도 거리 | 텍스트 임베딩, 문서 유사도 |
| `ip` | `array_negative_inner_product(vec1, vec2)` | 음의 내적 | 추천 시스템, 협업 필터링 |

### 거리 함수 사용 예제
```sql
-- L2 거리 계산
SELECT array_distance([1.0, 2.0, 3.0]::FLOAT[3], [2.0, 3.0, 4.0]::FLOAT[3]);

-- 코사인 거리 계산
SELECT array_cosine_distance([1.0, 0.0, 0.0]::FLOAT[3], [0.0, 1.0, 0.0]::FLOAT[3]);

-- 내적 계산
SELECT array_negative_inner_product([1.0, 2.0, 3.0]::FLOAT[3], [4.0, 5.0, 6.0]::FLOAT[3]);
```

## 쿼리 패턴 및 사용법

### 1. ORDER BY + LIMIT 패턴
가장 일반적인 k-최근접 이웃 검색 패턴:

```sql
-- 상위 3개 유사 벡터 검색
SELECT * 
FROM my_vector_table 
ORDER BY array_distance(vec, [1.0, 2.0, 3.0]::FLOAT[3]) 
LIMIT 3;
```

### 2. min_by 집계 함수 사용
```sql
-- 가장 유사한 3개 벡터를 구조체로 반환
SELECT min_by(my_vector_table, array_distance(vec, [1.0, 2.0, 3.0]::FLOAT[3]), 3 ORDER BY vec) AS result 
FROM my_vector_table;
```

### 3. 인덱스 사용 확인
```sql
-- 쿼리 실행 계획에서 HNSW_INDEX_SCAN 확인
EXPLAIN 
SELECT * 
FROM my_vector_table 
ORDER BY array_distance(vec, [1.0, 2.0, 3.0]::FLOAT[3]) 
LIMIT 3;
```

## 벡터 조인 연산

### vss_join 함수
```sql
-- 두 테이블 간 벡터 조인 (현재 브루트포스 방식)
SELECT vss_join(left_table, right_table, left_col, right_col, k, metric := 'l2sq');
```

### vss_match 함수
```sql
-- 벡터 매칭 함수
SELECT vss_match(right_table, left_col, right_col, k, metric := 'l2sq');
```

**주의사항**: 현재 벡터 조인 함수들은 HNSW 인덱스를 사용하지 않고 브루트포스 방식으로 동작합니다.

## 실제 사용 예제

### 테이블 생성 및 데이터 삽입
```sql
-- 벡터 테이블 생성
CREATE TABLE my_vector_table (
    id INTEGER,
    vec FLOAT[3],
    description TEXT
);

-- 샘플 데이터 삽입
INSERT INTO my_vector_table VALUES
    (1, [1.0, 2.0, 3.0], '첫 번째 벡터'),
    (2, [2.0, 3.0, 4.0], '두 번째 벡터'),
    (3, [1.5, 2.5, 3.5], '세 번째 벡터');

-- HNSW 인덱스 생성
CREATE INDEX my_hnsw_index ON my_vector_table USING HNSW(vec);
```

### 유사도 검색 쿼리
```sql
-- 쿼리 벡터와 가장 유사한 상위 2개 결과
SELECT id, description, array_distance(vec, [1.1, 2.1, 3.1]::FLOAT[3]) as distance
FROM my_vector_table 
ORDER BY distance 
LIMIT 2;
```

## 벡터 검색 시나리오별 구현 방안

### 1. 텍스트 임베딩 유사도 검색
- **사용 사례**: 문서 유사도, 의미 검색, 질의응답
- **거리 메트릭**: 코사인 유사도 (`array_cosine_distance`)
- **권장 차원**: 128, 256, 512, 768, 1024
- **데이터 생성**: Faker로 텍스트 생성 후 임베딩 시뮬레이션

```sql
-- 텍스트 임베딩 테이블 예제
CREATE TABLE document_embeddings (
    doc_id INTEGER,
    title TEXT,
    content TEXT,
    embedding FLOAT[512]
);

CREATE INDEX doc_embedding_idx ON document_embeddings USING HNSW(embedding);
```

### 2. 이미지 특징 벡터 매칭
- **사용 사례**: 이미지 유사도 검색, 중복 이미지 탐지
- **거리 메트릭**: L2 거리 (`array_distance`)
- **권장 차원**: 256, 512, 1024, 2048
- **데이터 생성**: 이미지 특징 벡터 시뮬레이션

```sql
-- 이미지 특징 테이블 예제
CREATE TABLE image_features (
    image_id INTEGER,
    filename TEXT,
    features FLOAT[1024]
);

CREATE INDEX img_features_idx ON image_features USING HNSW(features);
```

### 3. 추천 시스템 벡터
- **사용 사례**: 사용자-아이템 협업 필터링, 개인화 추천
- **거리 메트릭**: 내적 (`array_negative_inner_product`)
- **권장 차원**: 64, 128, 256
- **데이터 생성**: 사용자 선호도 및 아이템 특성 벡터

```sql
-- 사용자 임베딩 테이블 예제
CREATE TABLE user_embeddings (
    user_id INTEGER,
    preferences FLOAT[128]
);

CREATE TABLE item_embeddings (
    item_id INTEGER,
    features FLOAT[128]
);

CREATE INDEX user_pref_idx ON user_embeddings USING HNSW(preferences);
CREATE INDEX item_feat_idx ON item_embeddings USING HNSW(features);
```

## Faker를 활용한 데이터 생성 전략

### 한국어 로케일 설정
```python
from faker import Faker
fake = Faker('ko_KR')
```

### 텍스트 시나리오 데이터 생성
```python
# 문서 데이터 생성
documents = []
for i in range(10000):
    doc = {
        'id': i,
        'title': fake.sentence(nb_words=6),
        'content': fake.text(max_nb_chars=500),
        'category': fake.word(),
        'author': fake.name()
    }
    documents.append(doc)
```

### 사용자 프로필 데이터 생성
```python
# 사용자 프로필 생성
users = []
for i in range(5000):
    user = {
        'id': i,
        'name': fake.name(),
        'age': fake.random_int(min=18, max=80),
        'interests': [fake.word() for _ in range(fake.random_int(min=3, max=8))],
        'location': fake.city()
    }
    users.append(user)
```

### 제품 데이터 생성
```python
# 제품 카탈로그 생성
products = []
for i in range(20000):
    product = {
        'id': i,
        'name': fake.catch_phrase(),
        'description': fake.text(max_nb_chars=200),
        'category': fake.word(),
        'price': fake.random_int(min=1000, max=1000000),
        'brand': fake.company()
    }
    products.append(product)
```

## 성능 벤치마킹 프레임워크

### 측정 지표
1. **쿼리 실행 시간**: 평균, 중앙값, P95, P99 응답 시간
2. **메모리 사용량**: 인덱스 크기, 최대 메모리 소비량
3. **정확도**: Recall@k (k=1, 5, 10, 20)
4. **처리량**: 초당 쿼리 수 (QPS)
5. **인덱스 구축 시간**: 인덱스 생성 소요 시간
6. **저장 공간**: 지속적 인덱스 크기

### 테스트 데이터 규모
- **소규모**: 1,000개 벡터
- **중간 규모**: 10,000개 벡터
- **대규모**: 100,000개 벡터
- **초대규모**: 1,000,000개 벡터

### 벡터 차원별 테스트
- **저차원**: 64, 128 차원
- **중간 차원**: 256, 512 차원
- **고차원**: 1024 차원

### 거리 메트릭별 성능 비교
각 거리 메트릭(L2, 코사인, 내적)별로 다음 항목 측정:
- 계산 복잡도
- 메모리 사용 패턴
- 인덱스 효율성
- 검색 정확도

## 벤치마킹 실험 준비사항

### DuckDB VSS 확장 설정
```sql
-- VSS 확장 설치 및 로드
INSTALL vss;
LOAD vss;

-- 텍스트 벡터 테이블 생성 예시
CREATE TABLE text_vectors (
    id INTEGER PRIMARY KEY,
    text TEXT,
    embedding FLOAT[512],
    category VARCHAR,
    created_at TIMESTAMP
);

-- HNSW 인덱스 생성
CREATE INDEX text_hnsw_idx ON text_vectors USING HNSW(embedding) 
WITH (metric = 'cosine', ef_construction = 128, M = 16);
```

### 성능 측정 기본 쿼리
```sql
-- 벡터 유사도 검색 (기본)
SELECT id, text, array_cosine_distance(embedding, ?::FLOAT[512]) as distance
FROM text_vectors 
ORDER BY distance 
LIMIT 10;

-- 필터링 조건부 검색
SELECT id, text, array_cosine_distance(embedding, ?::FLOAT[512]) as distance
FROM text_vectors 
WHERE category = '뉴스'
ORDER BY distance 
LIMIT 10;
```

## 성능 최적화 권장사항

### 1. 인덱스 파라미터 튜닝
- **대용량 데이터**: `ef_construction` 증가 (256-512)
- **고정밀도 검색**: `ef_search` 증가 (128-256)
- **메모리 제약**: `M` 값 조정 (8-32 범위)

### 2. 쿼리 최적화
- 벌크 삽입 후 인덱스 생성
- 배치 쿼리 처리로 오버헤드 감소
- 적절한 LIMIT 값 설정

### 3. 메모리 관리
- 시스템 메모리의 70-80% 이내로 인덱스 크기 제한
- 주기적인 압축 작업 수행
- 불필요한 인덱스 제거

## 제한사항 및 주의사항

### 1. 기술적 제한사항
- FLOAT 32비트만 지원 (DOUBLE 미지원)
- 실험적 기능으로 안정성 제한
- 벡터 조인 함수의 HNSW 미지원
- 동적 업데이트 성능 제한

### 2. 운영상 고려사항
- 프로덕션 환경 사용 시 신중한 검토 필요
- 백업 및 복구 전략 수립
- 모니터링 및 알림 시스템 구축
- 성능 저하 시 대응 방안 준비

### 3. 확장성 제한
- 단일 노드 처리 (분산 처리 미지원)
- 메모리 기반 인덱스로 확장성 제한
- 실시간 업데이트 성능 이슈

## 실험 실행 계획

### Phase 1: 기본 성능 측정 (1-2주)
- 소규모 데이터 (10K 벡터) 전체 48개 조합 테스트
- 기본 HNSW 파라미터 최적화
- 벡터 차원별 성능 특성 파악

### Phase 2: 확장성 테스트 (2-3주)
- 중규모/대규모 데이터 성능 측정
- 메모리 사용량 모니터링
- 병목 지점 식별 및 최적화

### Phase 3: 고급 기능 테스트 (1-2주)
- 하이브리드 검색 (벡터 + BM25) 구현 및 성능 측정
- 필터링 검색 최적화
- 실시간 업데이트 성능 테스트

## 결론

DuckDB VSS 확장은 벡터 검색을 위한 강력한 도구이지만, 실험적 상태로 인한 제한사항들이 존재합니다. HNSW 인덱싱을 통한 효율적인 유사도 검색이 가능하며, 텍스트 임베딩, 이미지 특징, 추천 시스템 등 다양한 시나리오에 적용할 수 있습니다.

주요 강점:
- 빠른 벡터 유사도 검색 (HNSW 인덱스)
- 다양한 거리 메트릭 지원
- SQL 기반 쿼리 인터페이스
- 메모리 효율적인 처리

주요 제약사항:
- 실험적 기능으로 안정성 제한
- FLOAT 32비트만 지원
- 메모리 기반 인덱스 제한
- 동적 업데이트 성능 이슈

이러한 특성을 고려하여 적절한 사용 사례를 선택하고, 충분한 테스트를 통해 안정성을 확보한 후 활용하는 것이 권장됩니다.

## 참고 자료

- [DuckDB VSS Extension 공식 문서](https://duckdb.org/docs/stable/core_extensions/vss.html)
- [DuckDB Array Functions 문서](https://duckdb.org/docs/stable/sql/functions/array.html)
- [HNSW 알고리즘 논문](https://arxiv.org/abs/1603.09320)
- [Faker 라이브러리 문서](https://faker.readthedocs.io/)
