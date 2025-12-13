"""
gradio_ui.py

Gradio Blocks UI that exposes both the chat agent and helper widgets
for the underlying tools so they are easy to trigger from the browser.
"""

from typing import Any, Dict, Tuple

import gradio as gr

from ..tools.movie_tools import search_movies, recommend_movies
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
    def handle_movie_search(query, year, genre) -> Dict[str, Any]:
        if not query:
            return {"ok": False, "error": "검색어를 입력해주세요."}

        ok, parsed_year, err = _parse_year(year)
        if not ok:
            return {"ok": False, "error": err}

        return search_movies(query=query.strip(), year=parsed_year, genre=(genre or None))

    def handle_movie_recommend(preferences, count) -> Dict[str, Any]:
        if not preferences:
            return {"ok": False, "error": "선호도를 입력해주세요."}
        count_int = max(1, min(int(count or 5), 20))
        return recommend_movies(preferences=preferences.strip(), count=count_int)

    def handle_rag_search(query, top_k):
        if not query:
            empty = {
                "query": "",
                "contexts": [],
                "count": 0,
                "error": "질문을 입력해주세요."
            }
            return empty, ""
        top_k_int = max(1, min(int(top_k or 3), 10))
        result = search_rag(query=query.strip(), top_k=top_k_int)
        text = result.get("context_text", "")
        display = {k: v for k, v in result.items() if k != "context_text"}
        return display, text

    def handle_rag_initialize(force):
        return initialize_rag_database(document_directory="data", file_extension=".pdf", force=bool(force))

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
            )

        with gr.Tab("Tool 사용하기"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔍 영화 검색 Tool")
                    movie_query = gr.Textbox(label="검색어", placeholder="예) 인터스텔라")
                    movie_year = gr.Number(label="개봉 연도 (선택)", precision=0)
                    movie_genre = gr.Textbox(label="장르 (선택)", placeholder="예) SF")
                    movie_search_btn = gr.Button("검색 실행")
                    movie_result = gr.JSON(label="검색 결과")
                    movie_search_btn.click(
                        handle_movie_search,
                        inputs=[movie_query, movie_year, movie_genre],
                        outputs=movie_result,
                    )

                with gr.Column():
                    gr.Markdown("### 🎯 영화 추천 Tool")
                    pref_box = gr.Textbox(
                        label="선호도 설명",
                        placeholder="예) 우주 배경 SF 영화를 3편 추천해줘",
                        lines=3,
                    )
                    rec_count = gr.Slider(label="추천 개수", minimum=1, maximum=10, value=5, step=1)
                    rec_btn = gr.Button("추천 받기")
                    rec_result = gr.JSON(label="추천 결과")
                    rec_btn.click(handle_movie_recommend, inputs=[pref_box, rec_count], outputs=rec_result)

        with gr.Tab("RAG 도구"):
            gr.Markdown("### 📚 영화 메타데이터 RAG 검색")
            rag_query = gr.Textbox(
                label="질문", placeholder="예) 영화 '인터스텔라'의 핵심 주제는?", lines=2
            )
            rag_topk = gr.Slider(label="Top K", minimum=1, maximum=8, value=3, step=1)
            rag_btn = gr.Button("RAG 검색")
            rag_result = gr.JSON(label="검색 메타 정보")
            rag_context = gr.Textbox(label="LLM 컨텍스트", lines=10)
            rag_btn.click(
                handle_rag_search,
                inputs=[rag_query, rag_topk],
                outputs=[rag_result, rag_context],
            )

            gr.Markdown("### 🧱 RAG 데이터베이스 초기화")
            force_checkbox = gr.Checkbox(label="기존 데이터를 덮어쓰고 재색인", value=False)
            init_btn = gr.Button("PDF 재색인 실행")
            init_result = gr.JSON(label="초기화 결과")
            init_btn.click(handle_rag_initialize, inputs=[force_checkbox], outputs=init_result)

        gr.Markdown(
            "💡 FastAPI `/chat` 엔드포인트에서도 동일한 모델을 사용할 수 있으며, `/ui` 경로에 이 Gradio UI가 마운트되어 있습니다."
        )

    return demo
