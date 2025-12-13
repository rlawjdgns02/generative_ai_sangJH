"""
app.py

FastAPI + Gradio 통합 Movie Chat Application
"""

import os

import gradio as gr
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.graph.agent import MovieChatAgent
from src.schemas import ChatRequest, ChatResponse
from src.ui import create_ui

# 환경 변수 로드
load_dotenv()

# 1. FastAPI 앱 인스턴스 생성 (Gradio 마운트 전까지 유지)
fastapi_app = FastAPI(
    title="Movie Chat Agent Server",
    description="영화 검색, 추천, RAG 기반 영화 정보 검색을 지원하는 LangGraph 에이전트",
    version="1.0.0",
)

# 2. AI 에이전트 초기화
agent = MovieChatAgent(enable_memory=True)


# 3. 기본 경로 (안내 페이지)
@fastapi_app.get("/")
def root():
    return {
        "message": "Movie Chat Agent 서버가 정상 작동 중입니다!",
        "chat_ui": "/ui",
        "api_docs": "/docs",
        "rest_api": "/chat",
    }


# 4. API 엔드포인트 (선택사항 - 간단하게 구현)
@fastapi_app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    REST API 엔드포인트 (선택사항)

    대부분의 사용자는 /ui를 통해 접속하므로
    이 엔드포인트는 최소한으로 구현
    """
    try:
        ai_answer = agent.get_response(request.user_message, request.history)
        return ChatResponse(answer=ai_answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==========================================
# [핵심] Gradio UI 마운트
# ==========================================
gradio_interface = create_ui(agent)
app = gr.mount_gradio_app(fastapi_app, gradio_interface, path="/ui")


if __name__ == "__main__":
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set!")
        exit(1)

    # 서버 실행
    print("\n" + "=" * 60)
    print("Movie Chat Agent Server Starting...")
    print("=" * 60)
    print("Chat UI: http://127.0.0.1:8000/ui")
    print("API Docs: http://127.0.0.1:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
