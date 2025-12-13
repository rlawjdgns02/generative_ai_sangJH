"""
short_term.py

Short Term Memory - LangGraph State를 사용한 단기 메모리
"""

from typing import Dict, Any, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..schemas import AgentState


class ShortTermMemory:
    """
    LangGraph State를 사용한 단기 메모리 관리
    
    MemorySaver를 통해 대화 세션 내에서 상태를 유지
    """

    def __init__(self, enable: bool = True):
        """
        단기 메모리 초기화

        Args:
            enable: 메모리 활성화 여부
        """
        self.enable = enable
        self.checkpointer: BaseCheckpointSaver = MemorySaver() if enable else None
        print(f"[ShortTermMemory] 초기화 완료 - 활성화: {enable}")

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Checkpointer 인스턴스 반환"""
        return self.checkpointer

    def get_state_summary(self, state: AgentState) -> Dict[str, Any]:
        """
        현재 상태 요약 정보 추출

        Args:
            state: AgentState

        Returns:
            상태 요약 딕셔너리
        """
        messages = state.get("messages", [])
        summary = {
            "message_count": len(messages),
            "has_tool_result": state.get("tool_result") is not None,
            "has_final_answer": state.get("final_answer") is not None,
            "retrieved_contexts_count": len(state.get("retrieved_contexts", [])),
            "user_query": state.get("user_query", "")
        }
        print(f"[ShortTermMemory] 상태 요약: {summary}")
        return summary

    def extract_conversation_turn(self, state: AgentState) -> Dict[str, Any]:
        """
        현재 대화 턴 추출 (사용자 질문 + 어시스턴트 응답)

        Args:
            state: AgentState

        Returns:
            대화 턴 정보
        """
        messages = state.get("messages", [])
        user_query = state.get("user_query", "")
        final_answer = state.get("final_answer", "")
        
        # messages에서 최근 assistant 메시지 찾기
        assistant_message = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                assistant_message = msg.get("content", "")
                break
        
        if not final_answer and assistant_message:
            final_answer = assistant_message
        
        turn_info = {
            "user_query": user_query,
            "assistant_response": final_answer,
            "message_count": len(messages),
            "has_tool_usage": state.get("tool_result") is not None,
            "has_rag_context": len(state.get("retrieved_contexts", [])) > 0
        }
        print(f"[ShortTermMemory] 대화 턴 추출: 사용자 질문={user_query[:50]}..., 응답 길이={len(final_answer) if final_answer else 0}")
        return turn_info


