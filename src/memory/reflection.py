"""
reflection.py

Reflection - 자동 메모리 저장 로직
대화 내용을 분석하여 중요한 정보를 장기 메모리에 자동 저장
"""

import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from .long_term import get_long_term_memory

load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def should_save_memory(state: Dict[str, Any]) -> bool:
    """
    메모리를 저장해야 하는지 판단
    
    Args:
        state: AgentState
        
    Returns:
        저장 여부
    """
    # 최종 답변이 있는 경우만 저장
    if not state.get("final_answer"):
        return False
    
    # 사용자 질문이 있는지 확인
    if not state.get("user_query"):
        return False
    
    return True


def calculate_importance(user_query: str, assistant_response: str, context: Dict[str, Any]) -> float:
    """
    메모리 중요도 계산
    
    Args:
        user_query: 사용자 질문
        assistant_response: 어시스턴트 응답
        context: 추가 컨텍스트
        
    Returns:
        중요도 (0.0 ~ 1.0)
    """
    # 간단한 휴리스틱: 도구 사용 여부, 응답 길이, 특정 키워드 등
    importance = 0.3  # 기본 중요도
    
    # 도구를 사용한 경우 중요도 증가
    if context.get("tool_used"):
        importance += 0.3
    
    # RAG 검색을 사용한 경우 중요도 증가
    if context.get("rag_used"):
        importance += 0.2
    
    # 응답이 긴 경우 (상세한 정보 제공)
    if len(assistant_response) > 200:
        importance += 0.1
    
    # 사용자 선호도나 개인 정보가 포함된 경우
    preference_keywords = ["좋아", "선호", "싫어", "관심", "원해", "원하는", "기억"]
    if any(keyword in user_query for keyword in preference_keywords):
        importance += 0.2
    
    # 최대 1.0으로 제한
    return min(importance, 1.0)


def reflect_and_save(state: Dict[str, Any]) -> Optional[str]:
    """
    Reflection 노드 - 대화 내용을 분석하고 장기 메모리에 저장
    
    Args:
        state: AgentState
        
    Returns:
        저장된 메모리 ID (저장하지 않은 경우 None)
    """
    if not should_save_memory(state):
        return None
    
    user_query = state.get("user_query", "")
    assistant_response = state.get("final_answer", "")
    
    if not user_query or not assistant_response:
        return None
    
    # 컨텍스트 추출
    context = {
        "tool_used": state.get("tool_result") is not None,
        "rag_used": len(state.get("retrieved_contexts", [])) > 0,
        "retrieved_contexts_count": len(state.get("retrieved_contexts", [])),
    }
    
    # 중요도 계산
    importance = calculate_importance(user_query, assistant_response, context)
    
    # 중요도가 낮으면 저장하지 않음 (선택적 저장)
    if importance < 0.3:
        return None
    
    # 장기 메모리에 저장
    try:
        long_term_memory = get_long_term_memory()
        memory_id = long_term_memory.save_memory(
            user_query=user_query,
            assistant_response=assistant_response,
            context=context,
            importance=importance
        )
        return memory_id
    except Exception as e:
        print(f"⚠️  Failed to save memory: {e}")
        return None


def get_relevant_memories(query: str, top_k: int = 3) -> list:
    """
    현재 질문과 관련된 과거 메모리 검색
    
    Args:
        query: 현재 사용자 질문
        top_k: 반환할 메모리 개수
        
    Returns:
        관련 메모리 리스트
    """
    try:
        long_term_memory = get_long_term_memory()
        memories = long_term_memory.search_memories(query, top_k=top_k)
        return memories
    except Exception as e:
        print(f"⚠️  Failed to retrieve memories: {e}")
        return []


def format_memories_for_context(memories: list) -> str:
    """
    메모리를 LLM 컨텍스트에 추가할 수 있는 형식으로 변환
    
    Args:
        memories: 메모리 리스트
        
    Returns:
        포맷된 메모리 문자열
    """
    if not memories:
        return ""
    
    formatted = "\n\n[과거 대화 기록]\n"
    for i, memory in enumerate(memories, 1):
        formatted += f"\n{i}. 사용자: {memory.get('user_query', '')}\n"
        formatted += f"   어시스턴트: {memory.get('assistant_response', '')[:200]}...\n"
    
    return formatted


