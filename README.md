# generative_ai_sangJH
agent project for generative ai

movie-chat/
├── main.py                      # 메인 실행 파일
├── requirements.txt             # 의존성 관리
├── .env                        # 환경변수 (API 키 등)
├── .env.example               # 환경변수 예시
│
├── src/                       # 소스 코드 루트
│   ├── __init__.py
│   │
│   ├── graph/                 # LangGraph 에이전트 로직
│   │   ├── __init__.py
│   │   ├── agent.py          # 메인 에이전트 그래프 정의
│   │   └── nodes.py          # 그래프의 각 노드 함수들
│   │
│   ├── tools/                 # Tool 함수들
│   │   ├── __init__.py
│   │   ├── movie_tools.py    # 영화 검색, 추천 tool
│   │   └── search_tools.py   # 일반 검색 tool (선택)
│   │
│   ├── rag/                   # RAG 관련
│   │   ├── __init__.py
│   │   ├── loader.py         # 문서 로더 (PDF, 영화 데이터)
│   │   ├── vectorstore.py    # FAISS/Chroma 벡터 저장소
│   │   └── retriever.py      # 검색 로직
│   │
│   ├── memory/                # 대화 메모리 관리
│   │   ├── __init__.py
│   │   └── chat_memory.py    # 대화 히스토리 저장
│   │
│   ├── ui/                    # 사용자 인터페이스
│   │   ├── __init__.py
│   │   ├── gradio_app.py     # Gradio UI
│   │   └── fastapi_app.py    # FastAPI 서버 (선택)
│   │
│   ├── schemas.py             # Pydantic 스키마, TypedDict
│   ├── config.py              # 설정 파일
│   └── utils.py               # 유틸리티 함수
│
├── data/                      # 데이터 파일
│   ├── movies/               # 영화 데이터 (CSV, JSON)
│   ├── pdfs/                 # PDF 문서
│   └── vector_db/            # 벡터 DB 저장소
│
└── tests/                     # 테스트 코드
    ├── test_tools.py
    ├── test_agent.py
    └── test_rag.py