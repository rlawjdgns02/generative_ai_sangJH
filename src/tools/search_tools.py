"""
search_tools.py

RAG 검색 Tool 함수 (과제 방식 적용)
참고:
- code/rag/query.py의 search_documents 패턴
- code/utils.py의 build_prompt
"""

from typing import Dict, Any
from ..rag.retriever import MovieRetriever
import re


# 전역 Retriever 인스턴스 (과제 방식: 싱글톤 패턴)
_retriever = None

def initialize_rag_database(document_directory: str = "data", file_extension: str = ".pdf", force: bool = False):
    """
    RAG 데이터베이스 초기화 헬퍼 함수 (과제 방식 + 덮어쓰기 방지)

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

    code/rag/query.py와 동일하게 한 번만 초기화하여 재사용
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

def search_rag(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    RAG 검색 Tool 함수 (과제 방식)

    code/rag/query.py의 search_documents 패턴 적용
    - ChromaDB에서 관련 문서 검색
    - OpenAI embedding으로 유사도 계산
    - 컨텍스트 포맷팅하여 반환

    Args:
        query: 검색 질문
        top_k: 반환할 컨텍스트 개수

    Returns:
        {
            "query": 질문,
            "contexts": 컨텍스트 리스트,
            "context_text": LLM용 텍스트,
            "sources": 출처 리스트,
            "count": 결과 개수
        }
    """
    try:
        retriever = get_retriever()

        # 문서가 없으면 안내 (과제 방식)
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

        # 검색 실행 (code/rag/query.py의 search_documents 패턴)
        result = retriever.retrieve_with_context(query, top_k)

        # 출처 정보 추가 (과제 코드 방식: SOURCE:CHUNK 형태)
        sources = retriever.get_sources(query, top_k)
        result["sources"] = sources

        return result

    except Exception as e:
        return {
            "query": query,
            "contexts": [],
            "context_text": f"RAG 검색 중 오류 발생: {str(e)}",
            "sources": [],
            "count": 0,
            "error": str(e)
        }


# Tool 레지스트리
SEARCH_TOOLS = {
    "search_rag": search_rag,
    "recommend_by_genre": recommend_by_genre,
}
