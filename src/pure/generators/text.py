import random
from datetime import datetime, timedelta
from typing import List, Tuple
from ...types.core import Category, TextContent


def generate_korean_text(
    seed: int, category: Category, length_range: Tuple[int, int] = (100, 500)
) -> str:
    """시드 기반 한국어 텍스트 생성"""
    rng = random.Random(seed)

    templates = {
        Category.NEWS: [
            "오늘 {}에서 {}에 관한 중요한 발표가 있었습니다. {}는 {}에 대해 설명하며, {}의 중요성을 강조했습니다.",
            "{}에서 {}가 {}를 발표했습니다. 이는 {}에 큰 영향을 미칠 것으로 예상됩니다.",
            "최근 {}에서 {}와 관련된 {}가 화제가 되고 있습니다. 전문가들은 {}라고 분석했습니다.",
        ],
        Category.REVIEW: [
            "{}를 사용해본 결과 {}한 경험을 했습니다. 특히 {}가 인상적이었고, {}는 아쉬웠습니다.",
            "{}에 대한 리뷰를 작성합니다. {}의 장점은 {}이고, 단점은 {}입니다.",
            "{}를 구매하고 {}개월 사용한 후기입니다. {}는 만족스럽지만 {}는 개선이 필요합니다.",
        ],
        Category.DOCUMENT: [
            "{}에 관한 문서입니다. {}의 목적은 {}이며, {}를 통해 {}를 달성하고자 합니다.",
            "본 문서는 {}에 대한 {}를 다룹니다. {}의 절차는 {}와 같으며, {}를 준수해야 합니다.",
            "{}에 대한 가이드라인입니다. {}를 위해서는 {}가 필요하며, {}를 고려해야 합니다.",
        ],
        Category.SOCIAL: [
            "오늘 {}에서 {}를 했습니다! {}가 정말 {}해서 {}한 하루였어요.",
            "{}에 대해 생각해보니 {}네요. {}할 때 {}하면 좋을 것 같아요.",
            "{}와 함께 {}를 다녀왔어요. {}가 너무 {}해서 또 가고 싶어요!",
        ],
    }

    word_pools = {
        Category.NEWS: {
            "subjects": ["정부", "기업", "연구소", "대학교", "협회"],
            "topics": ["기술", "정책", "경제", "환경", "교육"],
            "actions": ["발표", "연구", "개발", "시행", "계획"],
            "objects": ["시스템", "프로그램", "정책", "기술", "서비스"],
            "adjectives": ["혁신적인", "효과적인", "중요한", "새로운", "획기적인"],
        },
        Category.REVIEW: {
            "products": ["스마트폰", "노트북", "이어폰", "카메라", "태블릿"],
            "experiences": ["만족스러운", "실망스러운", "놀라운", "평범한", "특별한"],
            "features": ["디자인", "성능", "배터리", "화질", "음질"],
            "issues": ["발열", "무게", "가격", "호환성", "내구성"],
            "periods": ["1", "2", "3", "6", "12"],
        },
        Category.DOCUMENT: {
            "topics": ["보안", "개발", "운영", "관리", "품질"],
            "purposes": ["가이드", "매뉴얼", "정책", "절차", "기준"],
            "methods": ["분석", "검토", "승인", "실행", "모니터링"],
            "requirements": ["규정", "표준", "원칙", "지침", "요구사항"],
            "considerations": ["보안성", "효율성", "안정성", "확장성", "유지보수성"],
        },
        Category.SOCIAL: {
            "places": ["카페", "공원", "영화관", "식당", "쇼핑몰"],
            "activities": ["산책", "영화감상", "쇼핑", "식사", "만남"],
            "emotions": ["행복", "즐거운", "편안한", "신나는", "따뜻한"],
            "descriptions": ["아름다운", "맛있는", "재미있는", "편리한", "친절한"],
            "feelings": ["기분좋은", "만족스러운", "즐거운", "행복한", "감동적인"],
        },
    }

    template = rng.choice(templates[category])
    words = word_pools[category]

    word_lists = list(words.values())
    selected_words = []
    for word_list in word_lists:
        selected_words.append(rng.choice(word_list))

    base_text = template.format(*selected_words[: template.count("{}")])
    target_length = rng.randint(*length_range)

    while len(base_text) < target_length:
        additional_template = rng.choice(templates[category])
        additional_words = []
        for word_list in word_lists:
            additional_words.append(rng.choice(word_list))
        additional_text = additional_template.format(
            *additional_words[: additional_template.count("{}")]
        )
        base_text += " " + additional_text

    return base_text[:target_length]


def generate_title(seed: int, category: Category) -> str:
    """카테고리별 제목 생성"""
    rng = random.Random(seed)

    title_templates = {
        Category.NEWS: [
            "{} {} 발표",
            "{} 관련 {} 소식",
            "{} {} 계획 공개",
            "{} {} 정책 시행",
        ],
        Category.REVIEW: [
            "{} {} 리뷰",
            "{} 사용 {} 후기",
            "{} {} 체험기",
            "{} {} 평가",
        ],
        Category.DOCUMENT: [
            "{} {} 가이드",
            "{} {} 매뉴얼",
            "{} {} 정책서",
            "{} {} 절차서",
        ],
        Category.SOCIAL: ["{} {} 일상", "{} {} 후기", "{} {} 경험", "{} {} 이야기"],
    }

    title_words = {
        Category.NEWS: ["정부", "기업", "기술", "정책", "혁신", "개발"],
        Category.REVIEW: ["제품", "서비스", "앱", "기기", "솔루션", "플랫폼"],
        Category.DOCUMENT: ["시스템", "프로세스", "보안", "운영", "관리", "품질"],
        Category.SOCIAL: ["일상", "여행", "맛집", "문화", "취미", "생활"],
    }

    template = rng.choice(title_templates[category])
    words = title_words[category]
    selected_words = [rng.choice(words) for _ in range(template.count("{}"))]

    return template.format(*selected_words)


def create_text_content(
    seed: int, category: Category, timestamp: datetime
) -> TextContent:
    """완전한 텍스트 콘텐츠 생성"""
    return TextContent(
        text=generate_korean_text(seed, category),
        title=generate_title(seed * 2, category),
        category=category,
        created_at=timestamp,
    )


def generate_timestamps(
    seed: int, count: int, start_date: datetime, end_date: datetime
) -> List[datetime]:
    """시드 기반 타임스탬프 생성"""
    rng = random.Random(seed)
    timestamps = []

    time_diff = end_date - start_date
    total_seconds = int(time_diff.total_seconds())

    for i in range(count):
        random_seconds = rng.randint(0, total_seconds)
        timestamp = start_date + timedelta(seconds=random_seconds)
        timestamps.append(timestamp)

    return sorted(timestamps)
