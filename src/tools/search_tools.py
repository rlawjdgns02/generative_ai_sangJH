"""
search_tools.py

RAG 검색 Tool 함수
구글 검색 Tool 함수

"""

from typing import Dict, Any
from ..rag.retriever import MovieRetriever
import re
import os
from googleapiclient.discovery import build


# 전역 Retriever 인스턴스
_retriever = None

def initialize_rag_database(document_directory: str = "data", file_extension: str = ".pdf", force: bool = False):
    """
    RAG 데이터베이스 초기화 헬퍼 함수 (덮어쓰기 방지)

    code/rag/build_index.py의 build_index 패턴 적용
    - data 폴더의 PDF 파일들을 ChromaDB에 색인
    - 이미 색인된 경우 스킵 (중복 방지)

    Args:
        document_directory: 문서 디렉토리 (기본: data/)
        file_extension: 파일 확장자 (기본: .pdf)
        force: True면 기존 데이터가 있어도 재색인 (기본: False)

    Returns:
        초기화 결과
    """
    try:
        retriever = get_retriever()

        # 덮어쓰기 방지: 이미 색인된 경우 스킵
        current_count = retriever.vectorstore.count()
        if current_count > 0 and not force:
            return {
                "ok": False,
                "message": f"이미 {current_count}개 청크가 색인되어 있습니다.",
                "count": current_count,
                "skipped": True,
                "hint": "재색인하려면 data/vector_db/ 폴더를 삭제하거나 force=True로 실행하세요."
            }

        # 색인 실행
        retriever.initialize_from_documents(document_directory, file_extension)

        return {
            "ok": True,
            "message": f"{retriever.vectorstore.count()}개 청크로 초기화 완료",
            "count": retriever.vectorstore.count()
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }

def get_retriever() -> MovieRetriever:
    """
    Retriever 싱글톤 패턴

    """
    global _retriever
    if _retriever is None:
        _retriever = MovieRetriever(persist_directory="data/vector_db")
    return _retriever


# 장르 키워드 매핑 (필요에 따라 확장)
GENRE_KEYWORDS = {
    "액션": ["액션", "action"],
    "모험": ["모험", "adventure"],
    "애니메이션": ["애니메이션", "animation"],
    "코미디": ["코미디", "comedy"],
    "범죄": ["범죄", "crime"],
    "드라마": ["드라마", "drama"],
    "다큐멘터리": ["다큐", "다큐멘터리", "documentary"],
    "가족": ["가족", "family"],
    "판타지": ["판타지", "fantasy"],
    "역사": ["역사", "history"],
    "공포": ["공포", "호러", "horror"],
    "음악": ["음악", "music"],
    "미스터리": ["미스터리", "mystery"],
    "로맨스": ["로맨스", "romance"],
    "SF": ["sf", "sci-fi", "science fiction", "공상과학"],
    "TV 영화": ["tv 영화", "tv movie"],
    "스릴러": ["스릴러", "thriller"],
    "전쟁": ["전쟁", "war"],
    "서부": ["서부", "western"],
}

# 영화 장르, 정보 파싱 함수
def _parse_movie_fields(text: str) -> Dict[str, Any]:
    """텍스트 블록에서 title/year/genres/vote/popularity/poster_path 등을 추출."""
    meta: Dict[str, Any] = {}
    lines = text.splitlines()
    blob = "\n".join(lines)

    # title
    m = re.search(r"title:\s*(.+)", blob, re.IGNORECASE)
    if m:
        meta["title"] = m.group(1).strip()

    # release year
    m = re.search(r"release_date:\s*([0-9]{4})", blob)
    if m:
        try:
            meta["year"] = int(m.group(1))
        except ValueError:
            pass

    # vote_average / popularity
    m = re.search(r"vote_average:\s*([\d\.]+)", blob)
    if m:
        try:
            meta["vote_average"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"popularity:\s*([\d\.]+)", blob)
    if m:
        try:
            meta["popularity"] = float(m.group(1))
        except ValueError:
            pass

    # poster_path
    m = re.search(r"poster_path:\s*(\S+)", blob)
    if m:
        meta["poster_path"] = m.group(1).strip()

    # genre_ids: 값이 숫자/문자 혼합일 수 있어 split
    m = re.search(r"genre_ids:\s*([^\n]+)", blob)
    if m:
        raw = m.group(1).strip()
        # 콤마/공백 기준 분리
        parts = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
        if parts:
            meta["genre_names"] = parts

    return meta


# 장르에 맞는 영화 추천 함수
def recommend_by_genre(query: str, top_k: int = 3, exclude_titles: str = "") -> Dict[str, Any]:
    q_lower = query.lower()
    target_genre = None
    for g, kws in GENRE_KEYWORDS.items():
        if any(kw in q_lower for kw in kws):
            target_genre = g
            break
    if not target_genre:
        target_genre = query.strip()

    # 제외할 영화 제목 파싱 (쉼표 또는 줄바꿈으로 구분)
    exclude_set = set()
    if exclude_titles:
        parts = [t.strip().lower() for t in exclude_titles.replace('\n', ',').split(',') if t.strip()]
        exclude_set = set(parts)

    retriever = get_retriever()
    internal_k = max(top_k * 20, 100)
    result = retriever.retrieve_with_context(query, internal_k)
    contexts = result.get("contexts", [])

    # 장르 정통성 반영 함수
    def genre_strength(genres, target):
        """목표 장르가 1순위이면 가중치 2, 포함만 되면 1, 없으면 0"""
        if not genres:
            return 0
        g = [str(x).lower() for x in genres]
        t = target.lower()
        if g and g[0] == t:
            return 2
        if t in g:
            return 1
        return 0

    filtered = []
    seen_keys = set()
    for ctx in contexts:
        md = ctx.get("metadata", {}) or {}
        text_raw = ctx.get("text") or ""

        # 텍스트에서 누락된 메타 채우기
        parsed = _parse_movie_fields(text_raw)
        for k, v in parsed.items():
            md.setdefault(k, v)
        ctx["metadata"] = md  # 업데이트된 메타 보존

        # 제외 필터링: 제목이 exclude_set에 있으면 스킵
        title = md.get("title", "")
        if title and any(excl in title.lower() for excl in exclude_set):
            continue

        genres = md.get("genre_names") or md.get("genres") or []
        text_lower = text_raw.lower()

        strength = genre_strength(genres, target_genre)
        if strength == 0 and any(kw in text_lower for kw in GENRE_KEYWORDS.get(target_genre, [target_genre.lower()])):
            strength = 1

        if strength > 0:
            md["_genre_strength"] = strength
            # title/year로 중복 제거
            key = (md.get("title"), md.get("year"))
            if key not in seen_keys:
                seen_keys.add(key)
                filtered.append(ctx)

    def sort_key(c):
        md = c.get("metadata", {}) or {}
        return (md.get("_genre_strength", 0), md.get("vote_average", 0.0), md.get("popularity", 0.0))

    filtered.sort(key=sort_key, reverse=True)

    # 장르 필터 후 모자라면 나머지로 채우기
    if len(filtered) < top_k:
        remaining = [c for c in contexts if c not in filtered]
        remaining.sort(key=sort_key, reverse=True)
        filtered.extend(remaining[: top_k - len(filtered)])

    filtered = filtered[:top_k]

    return {
        "query": query,
        "genre": target_genre,
        "count": len(filtered),
        "recommendations": [
            {
                "title": c.get("metadata", {}).get("title"),
                "year": c.get("metadata", {}).get("year"),
                "genres": c.get("metadata", {}).get("genre_names"),
                "vote_average": c.get("metadata", {}).get("vote_average"),
                "popularity": c.get("metadata", {}).get("popularity"),
                "poster_path": c.get("metadata", {}).get("poster_path"),
                "overview": c.get("text", ""),
                "source": c.get("metadata", {}).get("source"),
                "chunk_id": c.get("metadata", {}).get("chunk_id"),
            }
            for c in filtered
        ],
        "sources": [f"{c.get('metadata', {}).get('source')}:{c.get('metadata', {}).get('chunk_id')}" for c in filtered],
    }

# RAG 데이터 검색 함수 (제목 요청 시 정보 반환용)
def search_rag(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    RAG 검색 Tool 함수 (과제 방식)

    Args:
        query: 검색 질문
        top_k: 반환할 컨텍스트 개수

    Returns:
        {
            "query": 질문,
            "contexts": 컨텍스트 리스트,
            "context_text": LLM용 텍스트,
            "sources": 출처 리스트,
            "count": 결과 개수,
            "multiple_candidates": 동명 영화가 여러 개인 경우 True,
            "top_candidate": 가장 인기 있는 영화 (투표 수 기준)
        }
    """
    try:
        retriever = get_retriever()

        # 청킹 -> 색인 -> 임베딩되어있는 문서가 없으면 안내
        if retriever.vectorstore.count() == 0:
            return {
                "query": query,
                "contexts": [],
                "context_text": (
                    "벡터 데이터베이스에 문서가 없습니다.\n"
                    "먼저 다음 명령어로 데이터베이스를 초기화하세요:\n"
                    "python -m src.rag.loader"
                ),
                "sources": [],
                "count": 0,
                "warning": "Database is empty"
            }

        # 검색 실행 (더 많은 결과 가져와서 필터링)
        internal_k = max(top_k * 3, 10)
        result = retriever.retrieve_with_context(query, internal_k)

        contexts = result.get("contexts", [])

        # 메타데이터 파싱 및 보강
        for ctx in contexts:
            md = ctx.get("metadata", {}) or {}
            text_raw = ctx.get("text") or ""

            # 텍스트에서 누락된 메타 채우기
            parsed = _parse_movie_fields(text_raw)
            for k, v in parsed.items():
                md.setdefault(k, v)
            ctx["metadata"] = md

        # 동명/유사 제목 영화 체크
        # 유사도 판단: 제목의 처음 8글자가 비슷하면 같은 그룹으로 간주
        def normalize_title(title):
            """제목 정규화: 소문자, 특수문자 제거, 공백 제거"""
            import re
            normalized = re.sub(r'[^a-z0-9가-힣]', '', title.lower())
            return normalized[:12]  # 처음 12글자로 비교

        title_groups = {}
        for ctx in contexts:
            title = ctx.get("metadata", {}).get("title", "").strip()
            if title:
                normalized = normalize_title(title)
                if normalized not in title_groups:
                    title_groups[normalized] = []
                title_groups[normalized].append(ctx)

        multiple_candidates = False
        top_candidate = None

        # 동명/유사 영화가 2개 이상이면
        for normalized_title, candidates in title_groups.items():
            if len(candidates) >= 2:
                multiple_candidates = True
                # 투표 수(vote_count)로 정렬 (높은 순)
                candidates.sort(
                    key=lambda c: c.get("metadata", {}).get("vote_count", 0),
                    reverse=True
                )
                # 가장 인기 있는 영화를 top_candidate로 설정
                if not top_candidate or candidates[0].get("metadata", {}).get("vote_count", 0) > top_candidate.get("metadata", {}).get("vote_count", 0):
                    top_candidate = candidates[0]

        # 투표 수 기준으로 전체 정렬 (인기도 우선)
        contexts.sort(
            key=lambda c: (
                c.get("metadata", {}).get("vote_count", 0),  # 투표 수 (높은 순)
                -c.get("distance", 1.0)  # 거리 (낮은 순, 음수로 역정렬)
            ),
            reverse=True
        )

        # top_k 개수로 제한
        contexts = contexts[:top_k]

        # 출처 정보 추가
        sources = [f"{c.get('metadata', {}).get('source')}:{c.get('metadata', {}).get('chunk_id')}" for c in contexts]

        # context_text 재생성
        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            md = ctx.get("metadata", {}) or {}
            context_text += f"[{i}] TITLE={md.get('title', '')} YEAR={md.get('year', '')} GENRES={md.get('genre_names', '')} SOURCE={md.get('source', '')} | CHUNK={md.get('chunk_id', '')}\n"
            context_text += ctx.get("text", "") + "\n\n"

        return {
            "query": query,
            "contexts": contexts,
            "context_text": context_text,
            "sources": sources,
            "count": len(contexts),
            "multiple_candidates": multiple_candidates,
            "top_candidate": {
                "title": top_candidate.get("metadata", {}).get("title"),
                "year": top_candidate.get("metadata", {}).get("year"),
                "vote_count": top_candidate.get("metadata", {}).get("vote_count"),
                "overview": top_candidate.get("text", "")[:200] + "..."
            } if top_candidate else None
        }

    except Exception as e:
        return {
            "query": query,
            "contexts": [],
            "context_text": f"RAG 검색 중 오류 발생: {str(e)}",
            "sources": [],
            "count": 0,
            "error": str(e)
        }


def search_ott_availability(movie_title: str) -> Dict[str, Any]:
    """
    Google Custom Search API를 사용하여 JustWatch 링크를 찾아 OTT 시청 정보를 제공

    Args:
        movie_title: 영화 제목

    Returns:
        {
            "movie_title": 영화 제목,
            "justwatch_link": JustWatch 링크,
            "ott_info": OTT 안내 메시지,
            "found": 링크 발견 여부
        }
    """
    try:
        # 환경 변수에서 API 키 가져오기
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        if not api_key or not search_engine_id:
            return {
                "movie_title": movie_title,
                "justwatch_link": None,
                "ott_info": "Google Search API 설정이 필요합니다.",
                "found": False,
                "error": "Missing API credentials"
            }

        # Google Custom Search API 클라이언트 생성
        service = build("customsearch", "v1", developerKey=api_key)

        # 검색 쿼리 구성: JustWatch 사이트 우선 검색
        search_query = f"{movie_title} justwatch"

        # 검색 실행
        result = service.cse().list(
            q=search_query,
            cx=search_engine_id,
            num=5,  # 상위 5개 결과
            lr="lang_ko"  # 한국어 결과 우선
        ).execute()

        # 검색 결과에서 JustWatch 링크 찾기
        items = result.get("items", [])
        justwatch_link = None

        for item in items:
            link = item.get("link", "")
            title = item.get("title", "")

            # JustWatch.com 한국 사이트 링크 찾기
            if "justwatch.com/kr" in link.lower() and "영화" in title:
                # 영화 제목이 타이틀에 포함되어 있는지 확인 (정확도 향상)
                if any(word in title for word in movie_title.split()):
                    justwatch_link = link
                    break

        # 결과 포맷팅
        if justwatch_link:
            ott_info = (
                f"'{movie_title}' 영화의 OTT 시청 가능 여부는 아래 링크에서 확인하실 수 있습니다.\n"
                f"링크: {justwatch_link}\n\n"
                f"JustWatch에서 넷플릭스, 왓챠, 디즈니+, 티빙, 웨이브 등 다양한 플랫폼의 시청 정보를 제공합니다."
            )
            return {
                "movie_title": movie_title,
                "justwatch_link": justwatch_link,
                "ott_info": ott_info,
                "found": True
            }
        else:
            ott_info = f"'{movie_title}' 영화의 OTT 시청 정보를 찾을 수 없습니다. JustWatch에서 직접 검색해보시기 바랍니다."
            return {
                "movie_title": movie_title,
                "justwatch_link": None,
                "ott_info": ott_info,
                "found": False
            }

    except Exception as e:
        return {
            "movie_title": movie_title,
            "justwatch_link": None,
            "ott_info": f"OTT 검색 중 오류 발생: {str(e)}",
            "found": False,
            "error": str(e)
        }


# Tool 레지스트리
SEARCH_TOOLS = {
    "search_rag": search_rag,
    "recommend_by_genre": recommend_by_genre,
    "search_ott_availability": search_ott_availability,
}
