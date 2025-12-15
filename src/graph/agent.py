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

    def __init__(self, enable_memory: bool = True, enable_interrupt: bool = False):
        """
        에이전트 초기화

        Args:
            enable_memory: 대화 메모리 활성화 여부 (checkpointer 사용)
        """
        # Short Term Memory 초기화
        self.short_term_memory = ShortTermMemory(enable=enable_memory)
        self.checkpointer = self.short_term_memory.get_checkpointer()
        # human_in_the_loop 스타일 interrupt 사용 여부는 옵션으로 제어
        self.enable_interrupt = enable_interrupt
        interrupt_before = ["tool"] if enable_interrupt else None
        self.graph = self._build_graph(interrupt_before=interrupt_before)
        print(f"[INIT] MovieChatAgent 초기화 완료")

    def _build_graph(self, interrupt_before=None):
        """
        LangGraph 구성

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

        # 컴파일 (필요 시 interrupt_before 설정)
        compile_kwargs = {"checkpointer": self.checkpointer}
        if interrupt_before:
            compile_kwargs["interrupt_before"] = interrupt_before
        return builder.compile(**compile_kwargs)

    def invoke(self, input_data: Dict[str, Any], config: Dict[str, Any] = None):
        """
        그래프 실행

        """
        return self.graph.invoke(input_data, config=config)

    def stream(self, input_data: Dict[str, Any], config: Dict[str, Any] = None):
        """
        스트리밍 실행

        """
        return self.graph.stream(input_data, config=config)

    # ==========================================
    # Human-in-the-loop / Interrupt 지원용 메서드
    # ==========================================
    def run_with_interrupt(self, input_data: Dict[str, Any], config: Dict[str, Any] = None):
        """
        interrupt_before 설정을 활용한 실행 헬퍼

        - 내부적으로 stream을 사용하여 이벤트를 순회합니다.
        - 중간에 인터럽트가 발생하면 그 시점의 이벤트와 함께 반환합니다.
        - 기존 get_response에서는 사용하지 않으므로 기존 동작에는 영향을 주지 않습니다.
        """
        last_event = None
        for event in self.graph.stream(input_data, config=config):
            last_event = event
            # LangGraph의 human-in-the-loop 예제에서는 interrupt 이벤트를
            # 별도의 키로 구분합니다. 여기서는 안전하게 그대로 전달만 합니다.
            if isinstance(event, dict) and event.get("interrupted"):
                return {"status": "interrupt", "event": event}

        return {"status": "completed", "event": last_event}

    def continue_after_interrupt(self, updated_input: Dict[str, Any], config: Dict[str, Any] = None):
        """
        인터럽트 이후 재실행 헬퍼

        - checkpointer + 동일 thread_id를 활용해 이전 상태에서 이어서 실행합니다.
        - UI/서버 레이어에서 updated_input을 만들어 전달하는 패턴을 위한 메서드입니다.
        - 현재 Gradio/FastAPI 경로에서는 사용하지 않으므로 기존 동작에는 영향을 주지 않습니다.
        """
        return self.graph.invoke(updated_input, config=config)

    def get_response(self, user_message: str, history: List[List[str]] = None) -> str:
        """
        Gradio UI를 위한 인터페이스

        """
        if history is None:
            history = []

        # 시스템 메시지
        conversation = [
            {
                "role": "system",
                "content": (
                    "당신은 영화 정보/RAG 어시스턴트입니다.\n"
                    "\n"
                    "## 중요: OTT 정보 조회 정책\n"
                    "**반환하는 모든 영화에 대해 반드시 search_ott_availability를 호출해야 합니다.**\n"
                    "- 단일 영화든 여러 영화든 상관없이 각 영화마다 OTT 정보를 조회하세요.\n"
                    "- 여러 영화의 OTT를 조회할 때는 한 번에 모든 search_ott_availability를 동시에 호출할 수 있습니다.\n"
                    "- OTT 정보를 조회했으나 찾지 못한 경우:\n"
                    "  • '시청 가능한 OTT 정보를 찾을 수 없습니다' 또는 유사한 메시지를 표시\n"
                    "  • 영화 정보는 정상적으로 제공하되, OTT 부분만 'OTT 시청 정보를 찾을 수 없습니다. JustWatch에서 직접 검색해보시기 바랍니다.'으로 표시\n"
                    "- OTT 조회를 시도하지 않고 답변을 생성하는 것은 금지\n"
                    "\n"
                    "## ReAct 패턴 (Reason + Act)\n"
                    "영화 정보 질문이 들어오면:\n"
                    "1. **먼저** search_rag를 호출하여 영화 정보를 검색합니다.\n"
                    "2. search_rag 결과를 받으면:\n"
                    "   • **multiple_candidates=True**이면 동명 영화가 여러 개입니다.\n"
                    "   • 이 경우:\n"
                    "     a) 먼저 top_candidate에 대해 search_ott_availability를 호출\n"
                    "     b) top_candidate 정보와 OTT 정보를 함께 보여주며 확인 요청:\n"
                    "        \"'{제목} ({연도})' 영화가 맞으신가요? (투표 수: {vote_count})\"\n"
                    "        + OTT 정보도 함께 표시\n"
                    "   • 사용자가 '네', '맞아요', 'yes' 등으로 확인하면 최종 답변 제공\n"
                    "   • **multiple_candidates=False**이거나 단일 영화인 경우:\n"
                    "     contexts에 있는 모든 영화에 대해 search_ott_availability를 동시에 호출한 후 답변\n"
                    "3. 모든 OTT 정보를 받은 후에만 최종 답변을 생성합니다.\n"
                    "\n"
                    "장르 추천 요청이 들어오면:\n"
                    "1. recommend_by_genre를 호출합니다.\n"
                    "2. **필수**: recommendations에 있는 **모든 영화**에 대해 search_ott_availability를 호출합니다.\n"
                    "   • 예: 3편 추천받으면 3개의 search_ott_availability를 동시에 호출\n"
                    "3. 모든 OTT 정보를 받은 후 추천 결과와 OTT 링크를 함께 제공합니다.\n"
                    "   • '다른 영화 추천'이나 '제외하고' 요청 시 exclude_titles 파라미터를 사용하세요.\n"
                    "   • 예: recommend_by_genre(query='SF', exclude_titles='2001: A Space Odyssey, Finch')\n"
                    "\n"
                    "## 답변 형식 (모든 영화에 적용)\n"
                    "- 🖼️ 포스터 URL (있을 때)\n"
                    "- 🎬 작품 제목\n"
                    "- 📅 개봉일\n"
                    "- 🎭 장르 / 키워드\n"
                    "- ⭐ 평점 (vote_count도 함께 표시)\n"
                    "- 📖 줄거리\n"
                    "- 📺 시청가능 OTT (search_ott_availability 결과의 'ott_info' 필드 - 반드시 포함)\n"
                    "\n"
                    "## 주의사항\n"
                    "- 도구 결과가 비어 있으면 솔직히 '정보를 찾지 못했습니다'라고 답변\n"
                    "- 의미 없는 입력은 역할을 설명하고 재질문 유도\n"
                    "- 추측 금지, 반드시 도구 결과에 기반\n"
                    "- **중요**: OTT 정보를 조회하지 않고 답변하는 것은 금지"
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
            return result_state["final_answer"]

        # messages에서 마지막 assistant 메시지 추출
        messages = result_state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "죄송합니다. 답변을 생성할 수 없습니다.")

        return "죄송합니다. 답변을 생성할 수 없습니다."