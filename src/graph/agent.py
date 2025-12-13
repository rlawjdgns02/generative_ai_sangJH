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
        print(f"[MovieChatAgent] 메모리 시스템 초기화 중...")
        self.short_term_memory = ShortTermMemory(enable=enable_memory)
        self.checkpointer = self.short_term_memory.get_checkpointer()
        print(f"[MovieChatAgent] Short Term Memory: {'활성화' if enable_memory else '비활성화'}")
        self.graph = self._build_graph()
        print(f"[MovieChatAgent] 그래프 빌드 완료")

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
                    "- 영화 정보/제목/줄거리/배우/감독/평점 질문은 반드시 search_rag로 근거를 찾은 뒤 답합니다.\n"
                    "- 장르/추천 요청(예: 공포 영화 추천)은 recommend_by_genre를 호출해 장르 필터 + 평점/인기순으로 추천합니다.\n"
                    "  • 사용자가 '다른 영화 추천' 또는 '제외하고'라고 하면, 이전 대화에서 추천한 영화 제목을 exclude_titles 파라미터에 전달하세요.\n"
                    "  • 예: recommend_by_genre(query='SF', exclude_titles='2001: A Space Odyssey, Finch')\n"
                    "- 도구 결과가 비어 있으면 '관련 정보를 찾지 못했습니다'라고 솔직히 답합니다.\n"
                    "- 의미 없는 입력(adfadf 등)이면 역할을 말하고 다시 질문을 유도합니다.\n"
                    "- 답변 형식: 간결한 한국어, bullet 3~5개 이내.\n"
                    "- 🖼️ 포스터 URL(있을 때)\n"
                    "- 🎬 작품 제목\n"
                    "- 📅 개봉일\n"
                    "- 🎭 장르 / 키워드\n"
                    "- ⭐ 평점\n"
                    "- 📖 줄거리\n"
                    "- 추측 금지, 반드시 도구 결과에 기반해 답하십시오."
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
