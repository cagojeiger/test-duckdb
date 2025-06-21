from datetime import datetime
from src.types.core import Category, Dimension
from src.pure.generators.text import (
    generate_korean_text,
    generate_title,
    create_text_content,
    generate_timestamps,
)
from src.pure.generators.vectors import (
    generate_vector,
    generate_query_vectors,
    normalize_vector,
)


class TestTextGeneration:
    """한국어 텍스트 생성 테스트"""

    def test_generate_korean_text_deterministic(self) -> None:
        """동일한 시드로 동일한 텍스트 생성"""
        seed = 42
        category = Category.NEWS

        text1 = generate_korean_text(seed, category)
        text2 = generate_korean_text(seed, category)

        assert text1 == text2
        assert len(text1) >= 100
        assert len(text1) <= 500

    def test_generate_korean_text_categories(self) -> None:
        """모든 카테고리에서 텍스트 생성"""
        seed = 123

        for category in Category:
            text = generate_korean_text(seed, category)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_generate_title_deterministic(self) -> None:
        """제목 생성 결정성 테스트"""
        seed = 42
        category = Category.REVIEW

        title1 = generate_title(seed, category)
        title2 = generate_title(seed, category)

        assert title1 == title2
        assert len(title1) > 0

    def test_create_text_content(self) -> None:
        """완전한 텍스트 콘텐츠 생성"""
        seed = 42
        category = Category.DOCUMENT
        timestamp = datetime.now()

        content = create_text_content(seed, category, timestamp)

        assert content.category == category
        assert content.created_at == timestamp
        assert len(content.text) > 0
        assert len(content.title) > 0

    def test_generate_timestamps(self) -> None:
        """타임스탬프 생성 테스트"""
        seed = 42
        count = 10
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        timestamps = generate_timestamps(seed, count, start_date, end_date)

        assert len(timestamps) == count
        assert all(start_date <= ts <= end_date for ts in timestamps)
        assert timestamps == sorted(timestamps)  # 정렬된 상태


class TestVectorGeneration:
    """벡터 생성 테스트"""

    def test_normalize_vector(self) -> None:
        """벡터 정규화 테스트"""
        values = [3.0, 4.0]  # L2 norm = 5.0
        normalized = normalize_vector(values)

        assert abs(normalized[0] - 0.6) < 1e-10
        assert abs(normalized[1] - 0.8) < 1e-10

        norm = sum(x * x for x in normalized) ** 0.5
        assert abs(norm - 1.0) < 1e-10

    def test_generate_vector_deterministic(self) -> None:
        """벡터 생성 결정성 테스트"""
        seed = 42
        dimension = Dimension(128)

        vector1 = generate_vector(seed, dimension)
        vector2 = generate_vector(seed, dimension)

        assert vector1.dimension == vector2.dimension
        assert vector1.data == vector2.data
        assert len(vector1.data) == dimension

    def test_generate_vector_normalized(self) -> None:
        """생성된 벡터가 정규화되었는지 확인"""
        seed = 42
        dimension = Dimension(256)

        vector = generate_vector(seed, dimension)

        norm = sum(x * x for x in vector.data) ** 0.5
        assert abs(norm - 1.0) < 1e-10

    def test_generate_query_vectors(self) -> None:
        """쿼리 벡터들 생성 테스트"""
        seed = 42
        dimension = Dimension(128)
        count = 5

        vectors = generate_query_vectors(seed, dimension, count)

        assert len(vectors) == count
        assert all(v.dimension == dimension for v in vectors)
        assert all(len(v.data) == dimension for v in vectors)

        for i in range(count):
            for j in range(i + 1, count):
                assert vectors[i].data != vectors[j].data
