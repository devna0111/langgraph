from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
import re

# 도구 정의
@tool
def get_weather(city: str) -> str:
    """특정 도시의 날씨를 조회합니다."""
    weather_data = {
        "서울": "맑음, 15도",
        "부산": "흐림, 18도",
        "제주": "비, 20도"
    }
    return weather_data.get(city, f"{city}의 날씨 정보 없음")

@tool
def calculate(expression: str) -> str:
    """수식을 계산합니다. 예: 2+3, 15*23"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"계산 오류: {str(e)}"

@tool
def get_current_time() -> str:
    """현재 시간을 조회합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 도구 딕셔너리
tools_map = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_current_time": get_current_time
}

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_executed: bool

# LLM
llm = ChatOllama(
    model="devna0111-7b-q4",  # 파인튜닝 모델 사용!
    base_url="http://localhost:11434",
    temperature=0
)

# 도구 설명 생성
def get_tools_description():
    descriptions = []
    for name, tool_func in tools_map.items():
        desc = tool_func.description
        descriptions.append(f"- {name}: {desc}")
    return "\n".join(descriptions)

# 에이전트 노드
def agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    tool_executed = state.get("tool_executed", False)
    
    # 도구 실행 후라면 최종 답변 모드
    if tool_executed:
        system_prompt = """이전에 도구를 실행한 결과가 있습니다.
이 결과를 바탕으로 사용자에게 자연스러운 답변을 해주세요.
도구 형식(TOOL:/INPUT:)은 사용하지 마세요."""
    else:
        # 도구 사용 가능 모드
        system_prompt = f"""당신은 도구를 사용할 수 있는 AI 어시스턴트입니다.

사용 가능한 도구:
{get_tools_description()}

**중요**: 도구가 필요하면 **반드시** 아래 형식으로만 응답하세요:
TOOL: 도구이름
INPUT: 입력값

예시:
질문: "서울 날씨 알려줘"
답변:
TOOL: get_weather
INPUT: 서울

질문: "15 곱하기 23은?"
답변:
TOOL: calculate
INPUT: 15*23

질문: "지금 몇 시야?"
답변:
TOOL: get_current_time
INPUT: 

도구가 필요없는 일반 대화는 그냥 답변하세요."""

    full_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(full_messages)
    
    return {"messages": [response], "tool_executed": False}

# 도구 실행 노드
def tool_execution_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    content = last_message.content
    
    # TOOL: 패턴 파싱
    tool_match = re.search(r'TOOL:\s*(\w+)', content, re.IGNORECASE)
    input_match = re.search(r'INPUT:\s*(.+?)(?:\n|$)', content, re.IGNORECASE | re.DOTALL)
    
    if tool_match:
        tool_name = tool_match.group(1).strip()
        tool_input = input_match.group(1).strip() if input_match else ""
        
        print(f"🔧 도구 실행: {tool_name}('{tool_input}')")
        
        if tool_name in tools_map:
            try:
                result = tools_map[tool_name].invoke(tool_input)
                print(f"✅ 결과: {result}")
                
                # 도구 결과를 사용자 메시지로 추가
                result_msg = HumanMessage(
                    content=f"[도구 실행 결과]\n도구: {tool_name}\n입력: {tool_input}\n결과: {result}\n\n위 결과를 바탕으로 사용자에게 답변하세요."
                )
                return {"messages": [result_msg], "tool_executed": True}
            except Exception as e:
                error_msg = HumanMessage(
                    content=f"도구 실행 오류: {str(e)}\n사용자에게 오류를 알려주세요."
                )
                return {"messages": [error_msg], "tool_executed": True}
        else:
            error_msg = HumanMessage(
                content=f"'{tool_name}' 도구를 찾을 수 없습니다. 사용자에게 알려주세요."
            )
            return {"messages": [error_msg], "tool_executed": True}
    
    return {"tool_executed": False}

# 라우팅 함수
def should_use_tool(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    
    # 도구 실행 후라면 종료
    if state.get("tool_executed", False):
        return "end"
    
    # TOOL: 패턴이 있으면 도구 실행
    if "TOOL:" in last_message.content or "tool:" in last_message.content:
        return "tools"
    
    return "end"

# 그래프 생성
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_execution_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_use_tool,
    {
        "tools": "tools",
        "end": END
    }
)

# 도구 실행 후 다시 agent로 (최종 답변 생성)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# 테스트
if __name__ == "__main__":
    print("=== 프롬프트 엔지니어링 기반 도구 사용 ===\n")
    
    test_cases = [
        "서울 날씨 알려줘",
        "15 곱하기 23은?",
        "지금 몇 시야?",
        "안녕하세요"  # 도구 없이 일반 대화
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"[테스트 {i}] {query}")
        print("-" * 60)
        
        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "tool_executed": False
        })
        
        print(f"최종 답변: {result['messages'][-1].content}\n")