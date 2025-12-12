"""
vectorstore.py

ChromaDB 기반 벡터 저장소 (과제 방식)
참고:
- 과제 코드의 build_index 함수
- utils.py의 embed_texts (lines 63-65)
"""

import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class MovieVectorStore:
    """
    ChromaDB 기반 영화 정보 벡터 저장소 (과제 방식)

    과제 방식:
    - OpenAI로 직접 embedding 생성
    - ChromaDB에 embedding과 함께 저장
    """

    def __init__(self, persist_directory: str = "data/vector_db", collection_name: str = "movies"):
        """
        벡터 저장소 초기화

        Args:
            persist_directory: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

        # ChromaDB 클라이언트 생성 (자동 persist)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 컬렉션 생성 또는 로드
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"✅ ChromaDB initialized at {persist_directory}")
        print(f"📊 Collection '{collection_name}' has {self.collection.count()} documents")

    def add_documents(self, chunks: List[Any]) -> None:
        """
        OpenAI embedding을 사용하여 문서 추가 (과제 방식)

        Args:
            chunks: loader.py의 Chunk 리스트
        """
        if not chunks:
            print("⚠️  No chunks to add")
            return

        # 텍스트 추출
        texts = [chunk.text for chunk in chunks]

        # OpenAI로 embedding 생성 (과제 코드 방식)
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        response = self.openai_client.embeddings.create(model=self.embed_model, input=texts)
        embeddings = [item.embedding for item in response.data]

        # ChromaDB에 추가
        ids = [chunk.id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"✅ Added {len(chunks)} chunks with OpenAI embeddings")
        print(f"📊 Total documents: {self.collection.count()}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        유사도 기반 검색

        Args:
            query: 검색 질문
            top_k: 반환할 결과 개수

        Returns:
            검색 결과 리스트 (text, metadata, distance 포함)
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # 결과 정리
        formatted_results = []

        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0.0
                })

        return formatted_results

    def search_with_openai_embedding(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        OpenAI embedding을 사용한 검색 (과제 방식)

        Args:
            query: 검색 질문
            top_k: 반환할 결과 개수

        Returns:
            검색 결과 리스트
        """
        # 질문 임베딩 (과제 코드 방식)
        response = self.openai_client.embeddings.create(model=self.embed_model, input=[query])
        query_embedding = response.data[0].embedding

        # ChromaDB 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # 결과 정리
        formatted_results = []

        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0.0
                })

        return formatted_results

    def clear(self) -> None:
        """컬렉션 초기화"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Movie information chunks"}
        )
        print(f"🗑️  Collection '{self.collection_name}' cleared")

    def count(self) -> int:
        """저장된 문서 개수"""
        return self.collection.count()


# 테스트용
if __name__ == "__main__":
    # 벡터 저장소 생성
    store = MovieVectorStore()

    # 테스트 문서 추가
    from ..rag.loader import Chunk

    test_chunks = [
        Chunk(
            id="test_1",
            text="Interstellar is a 2014 science fiction film directed by Christopher Nolan.",
            metadata={"source": "test.txt", "chunk_id": 0}
        ),
        Chunk(
            id="test_2",
            text="The movie explores themes of space travel, time dilation, and human survival.",
            metadata={"source": "test.txt", "chunk_id": 1}
        )
    ]

    # 문서 추가
    store.add_documents(test_chunks)

    # 검색 테스트
    results = store.search("Tell me about Interstellar", top_k=2)

    print("\n🔍 Search results:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Distance: {result['distance']:.4f}")
        print(f"Text: {result['text'][:100]}...")
        print(f"Metadata: {result['metadata']}")
