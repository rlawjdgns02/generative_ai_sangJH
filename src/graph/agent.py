"""
agent.py

메인 LangGraph 에이전트 그래프 정의
강의 코드 참조:
- example.py: StateGraph 구성, conditional_edges
- final_ai_project/app/agent.py: AIAgent 클래스 패턴
- human_in_the_loop/app/agent.py: checkpointer, interrupt 지원
"""

from langgraph.graph import StateGraph, END
from typing import Dict, Any, List

from ..schemas import AgentState
from ..memory.short_term import ShortTermMemory
from .nodes import llm_node, tool_node, route_after_llm, reflection_node


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
        # Short Term Memory 초기화
        self.short_term_memory = ShortTermMemory(enable=enable_memory)
        self.checkpointer = self.short_term_memory.get_checkpointer()
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
        builder.add_node("reflection", reflection_node)  # Reflection 노드 추가

        # 엔트리 포인트 설정
        builder.set_entry_point("llm")

        # Conditional Edge: LLM → Tool or Reflection or END
        builder.add_conditional_edges(
            "llm",
            route_after_llm,
            {
                "tool": "tool",
                "reflection": "reflection",
                "END": END
            }
        )

        # Tool → LLM (ReAct loop)
        builder.add_edge("tool", "llm")
        
        # Reflection → END (메모리 저장 후 종료)
        builder.add_edge("reflection", END)

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
                    "당신은 영화 정보/RAG 어시스턴트입니다.\n"
                    "- 영화 정보, 추천, 제목 찾기 등 관련 모든 질문은 search_rag 도구를 먼저 호출하여 유사한 정보를 우선적으로 탐색합니다.\n"
                    "- 도구 결과가 비어 있으면 '관련 정보를 찾지 못했다'고 솔직히 답합니다.\n"
                    "- 관련 없는 질문이나 'adfadfadsf' 같이 아무 의미없는 질문이 들어오면 자신의 역할을 말하며 다시 질문을 유도합니다.\n"
                    "- 답변 형식: 간결한 한국어, 필요 시 bullet 3~5개 이내로 핵심만 요약. \n"
                    " • 첫 줄: gradio ui에서 포스터가 표시될 수 있도록 포스터 url.\n"
                    " • 두번째 줄: 영화명(원제/국문) + 연도 + 장르 + 평점(있을 때)\n"
                    " • 이후: 핵심 줄거리/특징\n"
                    "- **절대 추측으로 지어내지 말고**, 도구 결과에 기반해 답하십시오."
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
            "final_answer": None,
            "relevant_memories": [],  # 메모리 필드 초기화
            "saved_memory_id": None
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
            answer = result_state["final_answer"]
            print(f"[get_response] final answer preview: \n {answer}")
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
