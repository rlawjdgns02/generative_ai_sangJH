"""
agent.py

메인 LangGraph 에이전트 그래프 정의
강의 코드 참조:
- example.py: StateGraph 구성, conditional_edges
- final_ai_project/app/agent.py: AIAgent 클래스 패턴
- human_in_the_loop/app/agent.py: checkpointer, interrupt 지원
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, List

from ..schemas import AgentState
from .nodes import llm_node, tool_node, route_after_llm


class MovieChatAgent:
    """
    영화 추천 채팅 에이전트

    강의 코드 패턴 통합:
    - final_ai_project/app/agent.py의 AIAgent 클래스 구조
    - example.py의 그래프 구성 방식
    - human_in_the_loop/app/agent.py의 checkpointer 활용
    """

    def __init__(self, enable_memory: bool = True):
        """
        에이전트 초기화

        Args:
            enable_memory: 대화 메모리 활성화 여부 (checkpointer 사용)
        """
        self.checkpointer = MemorySaver() if enable_memory else None
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        LangGraph 구성

        참고:
        - example.py의 그래프 구성 (lines 99-114)
        - final_ai_project/app/agent.py의 workflow 구성 (lines 30-35)
        """
        # StateGraph 생성
        builder = StateGraph(AgentState)

        # 노드 추가
        builder.add_node("llm", llm_node)
        builder.add_node("tool", tool_node)

        # 엔트리 포인트 설정
        builder.set_entry_point("llm")

        # Conditional Edge: LLM → Tool or END
        builder.add_conditional_edges(
            "llm",
            route_after_llm,
            {
                "tool": "tool",
                "END": END
            }
        )

        # Tool → LLM (ReAct loop)
        builder.add_edge("tool", "llm")

        # 컴파일
        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, input_data: Dict[str, Any], config: Dict[str, Any] = None):
        """
        그래프 실행

        참고:
        - final_ai_project/app/agent.py의 invoke (line 91)
        - human_in_the_loop/app/agent.py의 invoke (line 70)
        """
        return self.graph.invoke(input_data, config=config)

    def stream(self, input_data: Dict[str, Any], config: Dict[str, Any] = None):
        """
        스트리밍 실행

        참고: examples/2_stream.py
        """
        return self.graph.stream(input_data, config=config)

    def get_response(self, user_message: str, history: List[List[str]] = None) -> str:
        """
        Gradio UI를 위한 인터페이스

        참고: final_ai_project/app/agent.py의 get_response (lines 74-96)
        """
        if history is None:
            history = []

        # 시스템 메시지
        conversation = [
            {
                "role": "system",
                "content": (
                    "당신은 영화 추천 AI 어시스턴트입니다.\n"
                    "- 사용자의 영화 관련 질문에 친절하게 답변합니다.\n"
                    "- 필요시 search_movies, recommend_movies, search_rag 도구를 활용합니다.\n"
                    "- 구체적이고 유용한 정보를 제공합니다."
                )
            }
        ]

        # 대화 히스토리 추가
        for item in history or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                user_msg, bot_msg = item[:2]
            elif isinstance(item, dict):
                if item.get("role") == "user":
                    user_msg, bot_msg = item.get("content"), None
                elif item.get("role") == "assistant":
                    user_msg, bot_msg = None, item.get("content")
                else:
                    continue
            else:
                continue

            if user_msg:
                conversation.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                conversation.append({"role": "assistant", "content": str(bot_msg)})


        # 현재 질문 추가
        conversation.append({"role": "user", "content": str(user_message)})

        # 그래프 실행 입력
        inputs = {
            "messages": conversation,
            "user_query": user_message,
            "tool_result": None,
            "retrieved_contexts": [],
            "final_answer": None
        }

        # checkpointer(MemorySaver)를 사용할 때는 thread_id 등 configurable 키가 필요함
        # Gradio ChatInterface에서는 세션 단위 스레드로 간단히 고정 ID를 사용
        config = {
            "configurable": {
                "thread_id": "gradio-chat-session"
            }
        }

        result_state = self.graph.invoke(inputs, config=config)

        # 최종 답변 추출
        if result_state.get("final_answer"):
            return result_state["final_answer"]

        # messages에서 마지막 assistant 메시지 추출
        messages = result_state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "죄송합니다. 답변을 생성할 수 없습니다.")

        return "죄송합니다. 답변을 생성할 수 없습니다."


# ==========================================
# 직접 실행 테스트
# ==========================================
if __name__ == "__main__":
    print("=== MovieChatAgent 테스트 ===\n")

    agent = MovieChatAgent(enable_memory=False)

    # 테스트 1: 간단한 질문
    print("Q1: 안녕하세요!")
    response1 = agent.get_response("안녕하세요!", [])
    print(f"A1: {response1}\n")

    # 테스트 2: 영화 검색
    print("Q2: 인터스텔라에 대해 알려줘")
    response2 = agent.get_response("인터스텔라에 대해 알려줘", [])
    print(f"A2: {response2}\n")

    # 테스트 3: 영화 추천
    print("Q3: SF 영화 추천해줘")
    response3 = agent.get_response("SF 영화 추천해줘", [])
    print(f"A3: {response3}\n")

    print("=== 테스트 완료 ===")
