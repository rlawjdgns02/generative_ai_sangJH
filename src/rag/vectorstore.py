"""
vectorstore.py

ChromaDB 기반 벡터 저장소

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
    ChromaDB 기반 영화 정보 벡터 저장소

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

        count = self.collection.count()
        print(f"[INIT] ChromaDB 초기화: {count}개 문서")

    def add_documents(self, chunks: List[Any], batch_size: int = 128) -> None:
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        total = len(texts)
        total_batches = (total + batch_size - 1) // batch_size
        print(f"[INIT] 총 {total}개 문서를 {batch_size}개씩 {total_batches}개 배치로 처리 중...")

        for batch_idx, start in enumerate(range(0, total, batch_size), 1):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            batch_ids = ids[start:end]
            batch_metas = metadatas[start:end]

            print(f"[INIT] 배치 {batch_idx}/{total_batches} 임베딩 생성 중... ({start+1}-{end}/{total})")
            resp = self.openai_client.embeddings.create(
                model=self.embed_model,
                input=batch_texts,
            )
            batch_embeddings = [item.embedding for item in resp.data]

            print(f"[INIT] 배치 {batch_idx}/{total_batches} DB 저장 중...")
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
            )
            print(f"[INIT] 배치 {batch_idx}/{total_batches} 완료 ✓")

        count = self.collection.count()
        print(f"[INIT] ✅ 문서 추가 완료: {count}개")


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