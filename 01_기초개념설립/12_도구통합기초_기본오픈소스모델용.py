from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import operator

# 도구 정의
@tool
def get_weather(city: str) -> str:
    """특정 도시의 날씨를 조회합니다."""
    # 실제로는 API 호출하지만, 여기서는 더미 데이터
    weather_data = {
        "서울": "맑음, 15도",
        "부산": "흐림, 18도",
        "제주": "비, 20도"
    }
    return weather_data.get(city, "날씨 정보 없음")

@tool
def calculate(expression: str) -> str:
    """수식을 계산합니다. 예: '2+3*4'"""
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"

# 도구 리스트
tools = [get_weather, calculate]

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 메시지 누적

# LLM 설정 (도구 바인딩)
llm = ChatOllama(
    # model="devna0111-7b-q4", 
    # 현재 위 모델은 한국어 파인튜닝 모델을 양자화한 것 처럼 메모리 제한을 두고 활용하는 것으로
    # 파인튜닝 시 Function Calling 도구를 학습시키지 않아 Catastrophic Forgetting (파국적 망각) 상태임
    model="qwen2.5:3b", 
    # 기본 오픈소스 모델은 지원
    base_url="http://localhost:11434",
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# 에이전트 노드
def agent_node(state: AgentState) -> AgentState:
    """LLM이 도구 사용 여부를 결정"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 도구 실행 노드
tool_node = ToolNode(tools)

# 조건부 라우팅: 도구 사용 여부
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    # LLM이 도구를 호출했는지 확인
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# 그래프 생성
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")  # 도구 실행 후 다시 에이전트로

app = workflow.compile()
try:
    img = "img/workflow_graph_TOOLS.png"
    png_data = app.get_graph().draw_mermaid_png()
    with open(img, "wb") as f:
        f.write(png_data)
    print(f"✅ 그래프 이미지 저장됨: {img}")
except Exception as e:
    print(f"⚠️  PNG 저장 실패: {e}")
    print("Graphviz가 설치되지 않았을 수 있습니다.")
# 테스트
if __name__ == "__main__":
    print("=== 도구 통합 테스트 ===\n")
    
    # 테스트 1: 날씨 조회
    print("[테스트 1] 날씨 조회")
    result1 = app.invoke({
        "messages": [HumanMessage(content="서울 날씨 알려줘")]
    })
    print(f"질문: 서울 날씨 알려줘")
    print(f"답변: {result1['messages'][-1].content}\n")
    
    # 테스트 2: 계산
    print("[테스트 2] 계산")
    result2 = app.invoke({
        "messages": [HumanMessage(content="15 곱하기 23은?")]
    })
    print(f"질문: 15 곱하기 23은?")
    print(f"답변: {result2['messages'][-1].content}\n")
    
    # 테스트 3: 도구 없이 일반 대화
    print("[테스트 3] 일반 대화")
    result3 = app.invoke({
        "messages": [HumanMessage(content="안녕하세요")]
    })
    print(f"질문: 안녕하세요")
    print(f"답변: {result3['messages'][-1].content}")