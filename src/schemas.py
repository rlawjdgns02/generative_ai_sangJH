"""
schemas.py

LangGraph State 정의 및 Pydantic 모델
강의 코드 참조:
- final_ai_project/app/schemas.py: ChatRequest, ChatResponse
- final_ai_project/app/agent.py: AgentState with chat_history
- human_in_the_loop/app/agent.py: AgentState with TypedDict
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ==========================================
# 1. LangGraph State 정의
# ==========================================
class AgentState(TypedDict):
    """
    LangGraph 에이전트의 상태

    참고:
    - final_ai_project/app/agent.py의 AgentState 패턴 활용
    - messages 키는 add 연산자로 누적
    """
    # 대화 메시지 (OpenAI 포맷)
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # 현재 사용자 질의
    user_query: str

    # Tool 실행 결과 (JSON 문자열)
    tool_result: Optional[str]

    # 검색된 컨텍스트 (RAG)
    retrieved_contexts: List[Dict[str, Any]]

    # 최종 답변
    final_answer: Optional[str]

    # 메모리 관련 필드
    relevant_memories: Annotated[List[Dict[str, Any]], operator.add]  # 관련 장기 메모리
    saved_memory_id: Optional[str]  # 저장된 메모리 ID


# ==========================================
# 2. Tool 입력/출력 스키마 (Pydantic)
# ==========================================

class MovieSearchInput(BaseModel):
    """
    영화 검색 Tool 입력 스키마

    참고: multiple-tools-with-template/tool_definitions.py의 Pydantic 패턴
    """
    query: str = Field(..., description="영화 검색 쿼리 (제목, 장르, 키워드 등)")
    year: Optional[int] = Field(None, description="개봉 연도 필터 (선택)")
    genre: Optional[str] = Field(None, description="장르 필터 (선택)")


class MovieRecommendInput(BaseModel):
    """영화 추천 Tool 입력 스키마"""
    preferences: str = Field(..., description="사용자 선호도 설명 (장르, 분위기 등)")
    count: int = Field(default=5, description="추천할 영화 개수", ge=1, le=20)


class RAGSearchInput(BaseModel):
    """RAG 검색 Tool 입력 스키마"""
    query: str = Field(..., description="검색할 질문 또는 키워드")
    top_k: int = Field(default=3, description="반환할 컨텍스트 개수", ge=1, le=10)


# ==========================================
# 3. UI/API용 Request/Response 스키마
# ==========================================

class ChatRequest(BaseModel):
    """
    채팅 요청 스키마

    참고: final_ai_project/app/schemas.py
    """
    user_message: str
    history: Optional[List[List[str]]] = []

    class Config:
        json_schema_extra = {
            "example": {
                "user_message": "액션 영화 추천해줘",
                "history": []
            }
        }


class ChatResponse(BaseModel):
    """채팅 응답 스키마"""
    answer: str
    sources: Optional[List[str]] = None  # RAG 출처 정보
