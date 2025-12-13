"""
retriever.py

RAG 검색 로직
강의 코드 참조:
- utils.py의 build_prompt (lines 72-98)
"""

import os
from typing import List, Dict, Any
from .vectorstore import MovieVectorStore
from .loader import load_documents_from_directory


class MovieRetriever:
    """
    영화 정보 검색기

    vectorstore를 사용하여 질문에 맞는 문서 조각 검색
    """

    def __init__(self, persist_directory: str = "data/vector_db"):
        """
        Retriever 초기화 (과제 방식 - 항상 OpenAI embedding 사용)

        Args:
            persist_directory: ChromaDB 저장 경로
        """
        self.vectorstore = MovieVectorStore(persist_directory=persist_directory)

    def initialize_from_documents(self, document_directory: str, file_extension: str = ".txt"):
        """
        문서 디렉토리에서 벡터 저장소 초기화

        Args:
            document_directory: 문서 디렉토리 경로
            file_extension: 파일 확장자 (.txt, .pdf 등)
        """
        print(f"📂 Loading documents from {document_directory}...")

        # 문서 로드 및 청킹
        chunks = load_documents_from_directory(document_directory, file_extension)

        if not chunks:
            print(f"⚠️  No documents found in {document_directory}")
            return

        # 벡터 저장소에 추가
        self.vectorstore.add_documents(chunks)

        print(f"✅ Initialization complete! Total documents: {self.vectorstore.count()}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        질문에 맞는 문서 조각 검색 (과제 방식 - OpenAI embedding)

        Args:
            query: 검색 질문
            top_k: 반환할 결과 개수

        Returns:
            검색 결과 리스트
        """
        return self.vectorstore.search_with_openai_embedding(query, top_k)

    def retrieve_with_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        검색 결과를 LLM이 사용하기 쉬운 형식으로 반환

        참고: utils.py의 build_prompt (lines 72-98)

        Args:
            query: 검색 질문
            top_k: 반환할 결과 개수

        Returns:
            {
                "query": 원래 질문,
                "contexts": 컨텍스트 리스트,
                "context_text": LLM용 텍스트
            }
        """
        # 검색
        results = self.retrieve(query, top_k)

        # 컨텍스트 정리
        contexts = []
        context_lines = []

        for idx, result in enumerate(results, start=1):
            # 메타데이터 추출
            source = result['metadata'].get('source', 'unknown')
            chunk_id = result['metadata'].get('chunk_id', '?')
            text = result['text']

            # 컨텍스트 저장
            contexts.append({
                "source": source,
                "chunk_id": chunk_id,
                "text": text,
                "distance": result.get('distance', 0.0)
            })

            # LLM용 텍스트 생성
            source_name = os.path.basename(source) if source != 'unknown' else 'unknown'
            context_lines.append(
                f"[{idx}] SOURCE={source_name} | CHUNK={chunk_id}\n{text}"
            )

        context_text = "\n\n".join(context_lines)

        return {
            "query": query,
            "contexts": contexts,
            "context_text": context_text,
            "count": len(contexts)
        }

    def get_context_for_llm(self, query: str, top_k: int = 3) -> str:
        """
        LLM 프롬프트에 바로 사용할 수 있는 컨텍스트 문자열 반환

        Args:
            query: 검색 질문
            top_k: 반환할 결과 개수

        Returns:
            포맷된 컨텍스트 문자열
        """
        result = self.retrieve_with_context(query, top_k)
        return result["context_text"]

    def get_sources(self, query: str, top_k: int = 3) -> List[str]:
        """
        출처 정보만 반환

        Args:
            query: 검색 질문
            top_k: 반환할 결과 개수

        Returns:
            출처 리스트 (예: ["movie.pdf:0", "movie.pdf:1"])
        """
        result = self.retrieve_with_context(query, top_k)

        sources = []
        for ctx in result["contexts"]:
            source_name = os.path.basename(ctx["source"]) if ctx["source"] != 'unknown' else 'unknown'
            sources.append(f"{source_name}:{ctx['chunk_id']}")

        return sources