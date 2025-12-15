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

        """
        return self.graph.invoke(input_data, config=config)

    def stream(self, input_data: Dict[str, Any], config: Dict[str, Any] = None):
        """
        스트리밍 실행

        """
        return self.graph.stream(input_data, config=config)

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
                    "## 중요: 한 번에 하나의 도구만 호출하세요!\n"
                    "- 여러 도구를 동시에 호출하지 마세요. 순차적으로 하나씩 호출합니다.\n"
                    "\n"
                    "## ReAct 패턴 (Reason + Act)\n"
                    "영화 정보 질문이 들어오면:\n"
                    "1. **먼저** search_rag를 호출하여 영화 정보를 검색합니다.\n"
                    "2. search_rag 결과를 받으면:\n"
                    "   • **multiple_candidates=True**이면 동명 영화가 여러 개입니다.\n"
                    "   • 이 경우 top_candidate 정보를 사용자에게 보여주고 확인을 요청하세요:\n"
                    "     \"'{제목} ({연도})' 영화가 맞으신가요? (투표 수: {vote_count})\"\n"
                    "   • 사용자가 '네', '맞아요', 'yes' 등으로 확인하면 다음 단계 진행\n"
                    "3. **그 다음** search_ott_availability를 호출하여 OTT 시청 링크를 가져옵니다.\n"
                    "4. 두 결과를 모두 받은 후 최종 답변을 생성합니다.\n"
                    "\n"
                    "장르 추천 요청이 들어오면:\n"
                    "1. recommend_by_genre를 호출합니다.\n"
                    "2. 결과의 첫 번째 영화에 대해 search_ott_availability를 호출합니다.\n"
                    "3. 추천 결과와 OTT 링크를 함께 제공합니다.\n"
                    "   • '다른 영화 추천'이나 '제외하고' 요청 시 exclude_titles 파라미터를 사용하세요.\n"
                    "   • 예: recommend_by_genre(query='SF', exclude_titles='2001: A Space Odyssey, Finch')\n"
                    "\n"
                    "## 답변 형식\n"
                    "- 🖼️ 포스터 URL (있을 때)\n"
                    "- 🎬 작품 제목\n"
                    "- 📅 개봉일\n"
                    "- 🎭 장르 / 키워드\n"
                    "- ⭐ 평점 (vote_count도 함께 표시)\n"
                    "- 📖 줄거리\n"
                    "- 📺 시청가능 OTT 확인 링크 (JustWatch) - search_ott_availability 결과의 'ott_info'를 그대로 사용\n"
                    "\n"
                    "## 주의사항\n"
                    "- 도구 결과가 비어 있으면 솔직히 '정보를 찾지 못했습니다'라고 답변\n"
                    "- 의미 없는 입력은 역할을 설명하고 재질문 유도\n"
                    "- 추측 금지, 반드시 도구 결과에 기반"
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