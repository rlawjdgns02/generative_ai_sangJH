"""
nodes.py

LangGraph의 각 노드 함수들
강의 코드 참조:
- example.py: llm_node, tool_node 패턴
- react_tool_agent (1).py: ReAct 패턴의 tool dispatch
- final_ai_project/app/agent.py: call_model 노드
"""

import os
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from ..schemas import AgentState

load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


# ==========================================
# 1. LLM Node
# ==========================================
def llm_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM 호출 노드 - Tool calling 지원

    참고:
    - example.py의 llm_node 패턴
    - react_tool_agent (1).py의 ReActToolAgent._chat
    """
    messages = state["messages"]

    # Tool 정의 (나중에 tools/ 폴더에서 가져올 예정)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "recommend_movies",
                "description": "사용자 선호도 기반 영화를 추천합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "preferences": {
                            "type": "string",
                            "description": "사용자 선호도 (장르, 분위기 등)"
                        },
                        "count": {
                            "type": "integer",
                            "description": "추천할 영화 개수",
                            "default": 5
                        }
                    },
                    "required": ["preferences"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_rag",
                "description": "영화 관련 문서에서 정보를 검색합니다 (RAG).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색할 질문"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "반환할 컨텍스트 개수",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # OpenAI API 호출
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message

    # Tool call이 있는 경우
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        return {
            "messages": [msg.model_dump()],
            "tool_result": json.dumps(tool_call.model_dump())
        }

    # 최종 답변인 경우
    return {
        "messages": [msg.model_dump()],
        "tool_result": None,
        "final_answer": msg.content
    }


# ==========================================
# 2. Tool Execution Node
# ==========================================
def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Tool 실행 노드

    참고:
    - example.py의 tool_node 패턴
    - react_tool_agent (1).py의 dispatch_tool
    """
    tool_result_json = state["tool_result"]
    if not tool_result_json:
        print("[tool_node] no tool_result, skipping")
        return {"messages": [], "tool_result": None}

    tool_call = json.loads(tool_result_json)
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    print(f"[tool_node] executing tool: {name} args={args}")

    result = execute_tool(name, args)
    print(f"[tool_node] result: {result}")

    observation = {
        "role": "tool",
        "content": json.dumps(result, ensure_ascii=False),
        "tool_call_id": tool_call["id"]
    }
    return {"messages": [observation], "tool_result": None}


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool 디스패처 - 실제 tool 함수 호출

    참고:
    - react_tool_agent (1).py의 dispatch_tool (lines 101-117)
    - multiple-tools-with-template/tool_registry.py의 call 메서드
    """
    # tools/ 폴더에서 실제 함수 임포트
    from ..tools.movie_tools import MOVIE_TOOLS
    from ..tools.search_tools import SEARCH_TOOLS

    # Tool 레지스트리 (모든 tool 통합)
    TOOL_REGISTRY = {
        **MOVIE_TOOLS,
        **SEARCH_TOOLS,
    }

    # Tool 실행
    if name not in TOOL_REGISTRY:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    try:
        tool_func = TOOL_REGISTRY[name]
        result = tool_func(**args)
        return {"ok": True, "tool": name, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e), "tool": name}


# ==========================================
# 3. Routing Function
# ==========================================
def route_after_llm(state: AgentState) -> str:
    """
    LLM 이후 라우팅 결정

    참고: example.py의 route 함수
    """
    if state.get("tool_result") is not None:
        return "tool"
    return "END"
