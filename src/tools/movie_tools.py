"""
movie_tools.py

영화 검색 및 추천 Tool 함수들
강의 코드 참조:
- multiple-tools-with-template/tool_definitions.py: Tool 함수 구조
- react_tool_agent (1).py: tool_get_time, tool_calculator 패턴
"""

from typing import Dict, Any, List


def search_movies(query: str, year: int = None, genre: str = None) -> Dict[str, Any]:
    """
    영화 검색 Tool (현재는 모의 데이터)

    TODO: 실제 구현 시
    - data/movies/ 폴더의 CSV/JSON 읽기
    - 또는 TMDB API 호출

    참고: multiple-tools-with-template/tool_definitions.py의 get_weather 패턴
    """
    # 모의 영화 데이터
    mock_movies = {
        "인터스텔라": {
            "title": "인터스텔라",
            "year": 2014,
            "director": "크리스토퍼 놀란",
            "genre": "SF",
            "rating": 8.6,
            "description": "우주를 여행하며 인류의 미래를 찾는 이야기"
        },
        "인셉션": {
            "title": "인셉션",
            "year": 2010,
            "director": "크리스토퍼 놀란",
            "genre": "SF",
            "rating": 8.8,
            "description": "꿈 속에서 생각을 심는 특수 요원의 이야기"
        },
        "매트릭스": {
            "title": "매트릭스",
            "year": 1999,
            "director": "워쇼스키 자매",
            "genre": "SF",
            "rating": 8.7,
            "description": "가상현실 세계에서 벗어나려는 해커의 이야기"
        },
        "다크나이트": {
            "title": "다크나이트",
            "year": 2008,
            "director": "크리스토퍼 놀란",
            "genre": "액션",
            "rating": 9.0,
            "description": "배트맨과 조커의 대결"
        }
    }

    # 간단한 검색 로직
    query_lower = query.lower()
    results = []

    for title, info in mock_movies.items():
        # 제목 매칭
        if query_lower in title.lower():
            # 연도 필터
            if year and info["year"] != year:
                continue
            # 장르 필터
            if genre and info["genre"].lower() != genre.lower():
                continue
            results.append(info)

    # 결과가 없으면 유사한 것 반환 (간단한 장르 매칭)
    if not results and genre:
        for title, info in mock_movies.items():
            if info["genre"].lower() == genre.lower():
                results.append(info)

    return {
        "query": query,
        "filters": {"year": year, "genre": genre},
        "count": len(results),
        "movies": results[:5]  # 최대 5개
    }


def recommend_movies(preferences: str, count: int = 5) -> Dict[str, Any]:
    """
    영화 추천 Tool (현재는 모의 데이터)

    TODO: 실제 구현 시
    - 사용자 선호도 임베딩
    - 벡터 유사도 기반 추천
    - 또는 협업 필터링

    참고: react_tool_agent (1).py의 tool 패턴
    """
    # 간단한 키워드 기반 추천
    preferences_lower = preferences.lower()

    recommendations = []

    # SF 키워드
    if any(kw in preferences_lower for kw in ["sf", "공상과학", "우주", "미래"]):
        recommendations.extend([
            {
                "title": "인터스텔라",
                "year": 2014,
                "genre": "SF",
                "rating": 8.6,
                "reason": "우주 여행과 미래 과학 요소"
            },
            {
                "title": "인셉션",
                "year": 2010,
                "genre": "SF",
                "rating": 8.8,
                "reason": "독창적인 SF 설정"
            },
            {
                "title": "매트릭스",
                "year": 1999,
                "genre": "SF",
                "rating": 8.7,
                "reason": "고전 SF 명작"
            }
        ])

    # 액션 키워드
    if any(kw in preferences_lower for kw in ["액션", "전투", "싸움", "히어로"]):
        recommendations.extend([
            {
                "title": "다크나이트",
                "year": 2008,
                "genre": "액션",
                "rating": 9.0,
                "reason": "최고의 히어로 액션"
            },
            {
                "title": "매트릭스",
                "year": 1999,
                "genre": "SF",
                "rating": 8.7,
                "reason": "혁신적인 액션 신"
            }
        ])

    # 기본 추천 (키워드 없을 때)
    if not recommendations:
        recommendations = [
            {
                "title": "인터스텔라",
                "year": 2014,
                "genre": "SF",
                "rating": 8.6,
                "reason": "평점이 높은 영화"
            },
            {
                "title": "다크나이트",
                "year": 2008,
                "genre": "액션",
                "rating": 9.0,
                "reason": "가장 인기있는 영화"
            }
        ]

    # 중복 제거
    seen_titles = set()
    unique_recs = []
    for rec in recommendations:
        if rec["title"] not in seen_titles:
            seen_titles.add(rec["title"])
            unique_recs.append(rec)

    return {
        "preferences": preferences,
        "count": min(count, len(unique_recs)),
        "recommendations": unique_recs[:count]
    }


# Tool 레지스트리
MOVIE_TOOLS = {
    "search_movies": search_movies,
    "recommend_movies": recommend_movies,
}
