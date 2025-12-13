"""
search_tools.py

RAG 검색 Tool 함수 (과제 방식 적용)
참고:
- code/rag/query.py의 search_documents 패턴
- code/utils.py의 build_prompt
"""

from typing import Dict, Any
from ..rag.retriever import MovieRetriever


# 전역 Retriever 인스턴스 (과제 방식: 싱글톤 패턴)
_retriever = None


def get_retriever() -> MovieRetriever:
    """
    Retriever 싱글톤 패턴

    code/rag/query.py와 동일하게 한 번만 초기화하여 재사용
    """
    global _retriever
    if _retriever is None:
        _retriever = MovieRetriever(persist_directory="data/vector_db")
    return _retriever


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


# Tool 레지스트리
SEARCH_TOOLS = {
    "search_rag": search_rag,
}


# 테스트용
if __name__ == "__main__":
    print("="*60)
    print("RAG Search Tool Test")
    print("="*60)

    # 데이터베이스 초기화 (처음 한 번만)
    print("\n1. Initializing database...")
    init_result = initialize_rag_database("data", ".pdf")
    print(f"Result: {init_result}")

    # 검색 테스트
    if init_result.get("ok"):
        print("\n2. Testing search...")
        search_result = search_rag("Tell me about Christopher Nolan movies", top_k=3)

        print(f"\nQuery: {search_result['query']}")
        print(f"Found: {search_result['count']} contexts")
        print(f"Sources: {search_result['sources']}")
        print(f"\nContext text:\n{search_result['context_text'][:500]}...")
    else:
        print("\n⚠️  Database initialization failed. Cannot test search.")
