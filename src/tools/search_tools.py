"""
search_tools.py

RAG 검색 Tool 함수
강의 코드 참조:
- utils.py의 build_prompt (lines 72-98)
- react_tool_agent (1).py의 tool 함수 패턴
"""

from typing import Dict, Any
from ..rag.retriever import MovieRetriever


# 전역 Retriever 인스턴스 (한 번만 초기화)
_retriever = None


def get_retriever() -> MovieRetriever:
    """
    Retriever 싱글톤 패턴 (과제 방식)

    매번 새로 만들지 않고 재사용
    """
    global _retriever
    if _retriever is None:
        _retriever = MovieRetriever(persist_directory="data/vector_db")
    return _retriever


def search_rag(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    RAG 검색 Tool 함수

    참고:
    - utils.py의 build_prompt 패턴
    - movie_tools.py의 Tool 함수 구조

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

        # 문서가 없으면 안내 메시지
        if retriever.vectorstore.count() == 0:
            return {
                "query": query,
                "contexts": [],
                "context_text": (
                    "No documents found in the vector database.\n"
                    "Please initialize the database first by running:\n"
                    "python -m src.rag.retriever"
                ),
                "sources": [],
                "count": 0,
                "warning": "Database is empty"
            }

        # 검색 실행
        result = retriever.retrieve_with_context(query, top_k)

        # 출처 정보 추가
        sources = retriever.get_sources(query, top_k)
        result["sources"] = sources

        return result

    except Exception as e:
        return {
            "query": query,
            "contexts": [],
            "context_text": f"Error during RAG search: {str(e)}",
            "sources": [],
            "count": 0,
            "error": str(e)
        }


def initialize_rag_database(document_directory: str = "data/pdfs", file_extension: str = ".pdf"):
    """
    RAG 데이터베이스 초기화 헬퍼 함수

    Args:
        document_directory: 문서 디렉토리
        file_extension: 파일 확장자

    Returns:
        초기화 결과
    """
    try:
        retriever = get_retriever()
        retriever.initialize_from_documents(document_directory, file_extension)

        return {
            "ok": True,
            "message": f"Initialized with {retriever.vectorstore.count()} documents",
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
    init_result = initialize_rag_database("data/pdfs", ".pdf")
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
