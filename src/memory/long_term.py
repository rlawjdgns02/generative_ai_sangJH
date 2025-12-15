"""
long_term.py

Long Term Memory - ChromaDB Persistent를 사용한 대화 메모리 저장
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LongTermMemory:
    """
    ChromaDB Persistent를 사용한 장기 메모리 저장소
    
    대화 내용을 임베딩하여 벡터 검색 가능한 형태로 저장
    """

    def __init__(
        self,
        persist_directory: str = "data/memory_db",
        collection_name: str = "conversation_memories"
    ):
        """
        장기 메모리 초기화

        Args:
            persist_directory: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

        # ChromaDB 클라이언트 생성 (Persistent)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 컬렉션 생성 또는 로드
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "description": "Long-term conversation memories"}
        )

        memory_count = self.collection.count()
        print(f"[INIT] Long-term memory: {memory_count}개 기존 메모리")

    def save_memory(
        self,
        user_query: str,
        assistant_response: str,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> str:
        """
        대화 메모리를 장기 저장소에 저장

        Args:
            user_query: 사용자 질문
            assistant_response: 어시스턴트 응답
            context: 추가 컨텍스트 (도구 사용, 검색 결과 등)
            importance: 메모리 중요도 (0.0 ~ 1.0)

        Returns:
            저장된 메모리의 ID
        """
        # 메모리 텍스트 구성
        memory_text = f"User: {user_query}\nAssistant: {assistant_response}"
        if context:
            context_str = json.dumps(context, ensure_ascii=False)
            memory_text += f"\nContext: {context_str}"

        # 임베딩 생성
        response = self.openai_client.embeddings.create(
            model=self.embed_model,
            input=[memory_text]
        )
        embedding = response.data[0].embedding

        # 메타데이터 구성
        timestamp = datetime.now().isoformat()
        # 안전한 ID 생성 (타임스탬프와 해시 조합)
        hash_value = hashlib.md5(memory_text.encode()).hexdigest()[:8]
        timestamp_safe = timestamp.replace(":", "-").replace(".", "-")
        memory_id = f"memory_{timestamp_safe}_{hash_value}"

        metadata = {
            "user_query": user_query,
            "assistant_response": assistant_response,
            "timestamp": timestamp,
            "importance": importance,
            "context": json.dumps(context, ensure_ascii=False) if context else ""
        }

        # ChromaDB에 저장
        self.collection.add(
            ids=[memory_id],
            documents=[memory_text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        print(f"[MEMORY] 저장 완료: {memory_id[:20]}...")
        return memory_id

    def search_memories(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        관련 메모리 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 메모리 개수
            min_importance: 최소 중요도 필터

        Returns:
            검색된 메모리 리스트
        """
        # 쿼리 임베딩 생성
        response = self.openai_client.embeddings.create(
            model=self.embed_model,
            input=[query]
        )
        query_embedding = response.data[0].embedding

        # ChromaDB 검색
        # 중요도 필터링은 검색 후에 적용 (ChromaDB where 절 제한)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2 if min_importance > 0 else top_k  # 필터링을 위해 더 많이 가져오기
        )

        # 결과 정리 및 중요도 필터링
        formatted_results = []

        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                importance = float(metadata.get("importance", 0.0))
                
                # 중요도 필터링
                if importance < min_importance:
                    continue
                
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "user_query": metadata.get("user_query", ""),
                    "assistant_response": metadata.get("assistant_response", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "importance": importance,
                    "context": json.loads(metadata.get("context", "{}")) if metadata.get("context") else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0.0
                })
                
                # top_k 개수만큼만 반환
                if len(formatted_results) >= top_k:
                    break

        print(f"[MEMORY] 검색 완료: {len(formatted_results)}개 발견")
        return formatted_results

    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        최근 메모리 조회

        Args:
            limit: 반환할 메모리 개수

        Returns:
            최근 메모리 리스트
        """
        # ChromaDB에서 모든 메모리 가져오기 (타임스탬프로 정렬)
        all_results = self.collection.get()
        
        if not all_results['ids']:
            return []

        # 메타데이터와 함께 정리
        memories = []
        for i, memory_id in enumerate(all_results['ids']):
            metadata = all_results['metadatas'][i] if all_results['metadatas'] else {}
            memories.append({
                "id": memory_id,
                "text": all_results['documents'][i],
                "user_query": metadata.get("user_query", ""),
                "assistant_response": metadata.get("assistant_response", ""),
                "timestamp": metadata.get("timestamp", ""),
                "importance": float(metadata.get("importance", 0.0)),
                "context": json.loads(metadata.get("context", "{}")) if metadata.get("context") else {}
            })

        # 타임스탬프로 정렬 (최신순)
        memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return memories[:limit]

    def count(self) -> int:
        """저장된 메모리 개수"""
        return self.collection.count()

    def clear(self) -> None:
        """모든 메모리 삭제"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "Long-term conversation memories"}
        )
        print(f"🗑️  All memories cleared")


# 전역 인스턴스 (싱글톤 패턴)
_long_term_memory_instance: Optional[LongTermMemory] = None


def get_long_term_memory() -> LongTermMemory:
    """장기 메모리 인스턴스 가져오기 (싱글톤)"""
    global _long_term_memory_instance
    if _long_term_memory_instance is None:
        _long_term_memory_instance = LongTermMemory()
    return _long_term_memory_instance

