"""
gradio_ui.py

Gradio Blocks UI that exposes both the chat agent and helper widgets
for the underlying tools so they are easy to trigger from the browser.
"""

from typing import Any, Dict, Tuple

import gradio as gr

from ..tools.search_tools import search_rag, initialize_rag_database


def _parse_year(year_value: Any) -> Tuple[bool, Any, str]:
    """Utility to safely convert year input to int."""
    if year_value in ("", None):
        return True, None, ""
    try:
        return True, int(year_value), ""
    except (TypeError, ValueError):
        return False, None, "연도는 숫자로 입력해주세요."


def create_ui(agent):
    """
    FastAPI(app.py)에서 호출할 UI 생성 함수

    Args:
        agent: MovieChatAgent 인스턴스 (src.graph.agent.MovieChatAgent)

    Returns:
        gr.Blocks: 채팅/툴 제어가 모두 포함된 Gradio Blocks UI
    """

    # =========================
    # 1) ChatInterface handlers
    # =========================
    def chat_function(message, history):
        """
        Gradio ChatInterface가 호출하는 함수

        Args:
            message: 사용자 입력
            history: 대화 히스토리 [[user, ai], [user, ai], ...]

        Returns:
            AI 응답
        """
        return agent.get_response(message, history)

    # =========================
    # 2) Tool helper handlers
    # =========================
    # def handle_movie_search(query, year, genre) -> Dict[str, Any]:
    #     if not query:
    #         return {"ok": False, "error": "검색어를 입력해주세요."}

    #     ok, parsed_year, err = _parse_year(year)
    #     if not ok:
    #         return {"ok": False, "error": err}

    #     return search_movies(query=query.strip(), year=parsed_year, genre=(genre or None))
    
    # =========================
    # 3) Compose Blocks layout
    # =========================
    with gr.Blocks(title="Movie Chat Agent") as demo:
        gr.Markdown(
            "## 🎬 Movie Chat Agent\n"
            "LangGraph 기반 에이전트가 영화 검색/추천 Tool과 RAG 검색을 통해 답변합니다.\n"
            "아래 탭에서 바로 Tool을 실행하거나 채팅으로 자연어 질의를 보낼 수 있습니다."
        )

        with gr.Tab("Chat"):
            # 일부 Gradio 버전에서는 submit_btn / retry_btn / clear_btn 인자를 지원하지 않으므로
            # 호환성을 위해 필수 인자만 사용한다.
            gr.ChatInterface(
                fn=chat_function,
                title="영화 Q&A",
                description="Tool을 자동으로 호출하는 ReAct 기반 챗봇입니다.",
                examples=[
                    "안녕하세요!",
                    "인터스텔라에 대해 알려줘",
                    "SF 영화 추천해줘",
                    "크리스토퍼 놀란 영화에 대해 알려줘",
                ],
                chatbot=gr.Chatbot(height=600),  # 채팅창 높이 조정
            )

        gr.Markdown(
            "💡 FastAPI `/chat` 엔드포인트에서도 동일한 모델을 사용할 수 있으며, `/ui` 경로에 이 Gradio UI가 마운트되어 있습니다."
        )

    return demo
