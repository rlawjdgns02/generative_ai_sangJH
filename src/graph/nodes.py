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
from ..memory.reflection import get_relevant_memories, format_memories_for_context

load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


# ==========================================
# 1. LLM Node
# ==========================================
def llm_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM 호출 노드 - Tool calling 지원 + 메모리 통합

    참고:
    - example.py의 llm_node 패턴
    - react_tool_agent (1).py의 ReActToolAgent._chat
    """
    messages = state["messages"].copy()
    
    # 관련 장기 메모리 검색 및 컨텍스트에 추가
    user_query = state.get("user_query", "")
    relevant_memories = []
    print(f"[llm_node] 메모리 검색 시작 - 사용자 질문: {user_query[:50]}...")
    if user_query:
        relevant_memories = get_relevant_memories(user_query, top_k=3)
        if relevant_memories:
            print(f"[llm_node] {len(relevant_memories)}개 관련 메모리 발견, 컨텍스트에 추가")
            memory_context = format_memories_for_context(relevant_memories)
            # 시스템 메시지에 메모리 컨텍스트 추가
            system_message_found = False
            for i, msg in enumerate(messages):
                if isinstance(msg, dict) and msg.get("role") == "system":
                    messages[i] = msg.copy()
                    messages[i]["content"] = msg.get("content", "") + memory_context
                    system_message_found = True
                    print(f"[llm_node] 시스템 메시지에 메모리 컨텍스트 추가됨")
                    break
            
            if not system_message_found:
                # 시스템 메시지가 없으면 추가
                messages.insert(0, {
                    "role": "system",
                    "content": (
                        "당신은 영화 추천 AI 어시스턴트입니다.\n\n"
                        "지원하는 기능:\n"
                        "1. 영화 제목으로 정보 검색 (search_rag)\n"
                        "2. 특정 영화와 비슷한 영화 추천\n"
                        "3. 장르별 영화 추천 (recommend_by_genre)\n\n"
                        "중요: 사용자가 줄거리만 말하면 '영화 제목을 알려주시면 더 정확히 도와드릴 수 있습니다'라고 안내하세요."
                    ) + memory_context
                })
                print(f"[llm_node] 새로운 시스템 메시지 생성 (메모리 포함)")
        else:
            print(f"[llm_node] 관련 메모리 없음")
    else:
        print(f"[llm_node] 사용자 질문 없음, 메모리 검색 스킵")

    # Tool 정의 (나중에 tools/ 폴더에서 가져올 예정)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "recommend_by_genre",
                "description": "질문에서 장르를 추출해 해당 장르 영화 중 평점/인기순으로 상위 N편을 추천합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "사용자 요청/장르 힌트"},
                        "top_k": {"type": "integer", "description": "추천 개수", "default": 3},
                        "exclude_titles": {
                            "type": "string",
                            "description": "제외할 영화 제목들 (쉼표로 구분). 예: '2001: A Space Odyssey, Finch'",
                            "default": ""
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_rag",
                "description": "영화 제목으로 영화 정보를 검색합니다. 제목이 명확할 때만 사용하세요.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "영화 제목. 예: '인터스텔라', '기생충', '어벤져스'"
                        },
                        "top_k": {"type": "integer", "description": "검색 결과 개수", "default": 3}
                    },
                    "required": ["query"]
                }
            }
        },
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
        print(f"[llm_node] Tool call 감지: {tool_call.function.name}")
        return {
            "messages": [msg.model_dump()],
            "tool_result": json.dumps(tool_call.model_dump()),
            "relevant_memories": relevant_memories  # 관련 메모리 저장
        }

    # 최종 답변인 경우
    print(f"[llm_node] 최종 답변 생성 완료 (길이: {len(msg.content) if msg.content else 0}자)")
    return {
        "messages": [msg.model_dump()],
        "tool_result": None,
        "final_answer": msg.content,
        "relevant_memories": relevant_memories  # 관련 메모리 저장
    }


# ==========================================
# 4. Reflection Node (자동 메모리 저장)
# ==========================================
def reflection_node(state: AgentState) -> Dict[str, Any]:
    """
    Reflection 노드 - 대화 내용을 분석하고 장기 메모리에 자동 저장
    
    참고: 메모리 시스템의 reflection 모듈 사용
    """
    print(f"[reflection_node] Reflection 노드 실행 시작")
    print(f"[reflection_node] State 확인:")
    print(f"  - user_query: {state.get('user_query', '')[:50]}...")
    print(f"  - final_answer 존재: {bool(state.get('final_answer'))}")
    print(f"  - tool_result 존재: {bool(state.get('tool_result'))}")
    print(f"  - retrieved_contexts 개수: {len(state.get('retrieved_contexts', []))}")
    
    from ..memory.reflection import reflect_and_save
    
    saved_memory_id = reflect_and_save(state)
    
    if saved_memory_id:
        print(f"[reflection_node] ✅ 메모리 저장 완료: {saved_memory_id}")
    else:
        print(f"[reflection_node] ⏭️  메모리 저장 스킵됨")
    
    return {
        "saved_memory_id": saved_memory_id
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
    from ..tools.search_tools import SEARCH_TOOLS

    # Tool 레지스트리 (모든 tool 통합)
    TOOL_REGISTRY = {
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
    tool_result = state.get("tool_result")
    final_answer = state.get("final_answer")
    
    print(f"[route_after_llm] 라우팅 결정:")
    print(f"  - tool_result 존재: {tool_result is not None}")
    print(f"  - final_answer 존재: {final_answer is not None}")
    
    # Tool call이 있으면 tool 노드로
    if tool_result is not None:
        print(f"[route_after_llm] → 'tool' 노드로 라우팅")
        return "tool"
    # 최종 답변이 있으면 reflection 노드로 이동 (메모리 저장)
    if final_answer:
        print(f"[route_after_llm] → 'reflection' 노드로 라우팅 (메모리 저장)")
        return "reflection"
    print(f"[route_after_llm] → 'END'로 라우팅")
    return "END"
