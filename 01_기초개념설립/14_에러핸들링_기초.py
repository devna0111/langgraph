from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
import re

# 도구 정의 (에러 발생 가능)
@tool
def get_weather(city: str) -> str:
    """특정 도시의 날씨를 조회합니다."""
    weather_data = {
        "서울": "맑음, 15도",
        "부산": "흐림, 18도",
        "제주": "비, 20도"
    }
    
    # 에러 케이스 1: 도시 정보 없음
    if city not in weather_data:
        raise ValueError(f"'{city}'의 날씨 정보가 없습니다")
    
    return weather_data[city]

@tool
def calculate(expression: str) -> str:
    """수식을 계산합니다."""
    # 에러 케이스 2: 잘못된 수식
    if not expression or expression.strip() == "":
        raise ValueError("수식이 비어있습니다")
    
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        raise ValueError(f"잘못된 수식입니다: {expression}")

@tool
def divide(a: float, b: float) -> str:
    """두 수를 나눕니다."""
    # 에러 케이스 3: 0으로 나누기
    if b == 0:
        raise ZeroDivisionError("0으로 나눌 수 없습니다")
    
    return str(a / b)

tools_map = {
    "get_weather": get_weather,
    "calculate": calculate,
    "divide": divide
}

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_executed: bool
    error_count: int  # 에러 카운트 추가
    max_retries: int  # 최대 재시도 횟수

# LLM
llm = ChatOllama(
    model="devna0111-7b-q4",
    base_url="http://localhost:11434",
    temperature=0
)

def get_tools_description():
    descriptions = []
    for name, tool_func in tools_map.items():
        descriptions.append(f"- {name}: {tool_func.description}")
    return "\n".join(descriptions)

# 에이전트 노드
def agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    tool_executed = state.get("tool_executed", False)
    error_count = state.get("error_count", 0)
    
    if tool_executed:
        system_prompt = """도구 실행 결과를 바탕으로 답변하세요.
                            에러가 발생했다면 사용자에게 친절하게 설명하세요."""
    else:
        system_prompt = f"""당신은 도구를 사용할 수 있는 AI입니다.

                            사용 가능한 도구:
                            {get_tools_description()}

                            도구가 필요하면 다음 형식으로 응답하세요:
                            TOOL: 도구이름
                            INPUT: 입력값

                            예시:
                            TOOL: get_weather
                            INPUT: 서울"""

    full_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(full_messages)
    
    return {"messages": [response], "tool_executed": False}

# 도구 실행 노드 (에러 핸들링 포함)
def tool_execution_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    content = last_message.content
    error_count = state.get("error_count", 0)
    
    tool_match = re.search(r'TOOL:\s*(\w+)', content, re.IGNORECASE)
    input_match = re.search(r'INPUT:\s*(.+?)(?:\n|$)', content, re.IGNORECASE | re.DOTALL)
    
    if tool_match:
        tool_name = tool_match.group(1).strip()
        tool_input = input_match.group(1).strip() if input_match else ""
        
        print(f"🔧 도구 실행 시도: {tool_name}('{tool_input}')")
        
        if tool_name not in tools_map:
            error_msg = HumanMessage(
                content=f"[에러] '{tool_name}' 도구를 찾을 수 없습니다.\n사용자에게 알려주세요."
            )
            return {
                "messages": [error_msg],
                "tool_executed": True,
                "error_count": error_count + 1
            }
        
        # 도구 실행 (try-except로 에러 처리)
        try:
            result = tools_map[tool_name].invoke(tool_input)
            print(f"✅ 성공: {result}")
            
            result_msg = HumanMessage(
                content=f"[도구 실행 성공]\n도구: {tool_name}\n결과: {result}\n\n위 결과로 답변하세요."
            )
            return {
                "messages": [result_msg],
                "tool_executed": True,
                "error_count": 0  # 성공 시 에러 카운트 리셋
            }
            
        except ValueError as e:
            print(f"❌ ValueError: {str(e)}")
            error_msg = HumanMessage(
                content=f"[도구 실행 실패]\n도구: {tool_name}\n에러: {str(e)}\n\n사용자에게 친절하게 설명하세요."
            )
            return {
                "messages": [error_msg],
                "tool_executed": True,
                "error_count": error_count + 1
            }
            
        except ZeroDivisionError as e:
            print(f"❌ ZeroDivisionError: {str(e)}")
            error_msg = HumanMessage(
                content=f"[도구 실행 실패]\n도구: {tool_name}\n에러: {str(e)}\n\n사용자에게 설명하세요."
            )
            return {
                "messages": [error_msg],
                "tool_executed": True,
                "error_count": error_count + 1
            }
            
        except Exception as e:
            print(f"❌ 예상치 못한 에러: {str(e)}")
            error_msg = HumanMessage(
                content=f"[시스템 에러]\n예상치 못한 오류가 발생했습니다: {str(e)}\n\n사용자에게 사과하고 다시 시도하도록 안내하세요."
            )
            return {
                "messages": [error_msg],
                "tool_executed": True,
                "error_count": error_count + 1
            }
    
    return {"tool_executed": False}

# 라우팅 함수
def should_use_tool(state: AgentState) -> Literal["tools", "end", "error_limit"]:
    last_message = state["messages"][-1]
    error_count = state.get("error_count", 0)
    max_retries = state.get("max_retries", 3)
    
    # 에러가 너무 많으면 종료
    if error_count >= max_retries:
        return "error_limit"
    
    if state.get("tool_executed", False):
        return "end"
    
    if "TOOL:" in last_message.content or "tool:" in last_message.content:
        return "tools"
    
    return "end"

# 에러 리밋 노드
def error_limit_node(state: AgentState) -> AgentState:
    error_msg = AIMessage(
        content="죄송합니다. 요청을 처리하는 중 문제가 반복적으로 발생했습니다. 다른 방식으로 도와드릴까요?"
    )
    return {"messages": [error_msg]}

# 그래프 생성
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_execution_node)
workflow.add_node("error_limit", error_limit_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_use_tool,
    {
        "tools": "tools",
        "end": END,
        "error_limit": "error_limit"
    }
)

workflow.add_edge("tools", "agent") 
# 처음에 agent 노드에서 도구 필요여부 파악
# should_use_tool 함수로 라우팅
# 이후 결과에 따라 노드를 따라 추가됨

workflow.add_edge("error_limit", END)
app = workflow.compile()

try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("img/workflow_graph.png_에러핸들링", "wb") as f:
        f.write(png_data)
    print("✅ 그래프 이미지 저장됨: img/workflow_graph.png_에러핸들링")
except Exception as e:
    print(f"⚠️  PNG 저장 실패: {e}")
    print("Graphviz가 설치되지 않았을 수 있습니다.")

# 테스트
if __name__ == "__main__":
    print("=== 에러 핸들링 테스트 ===\n")
    
    test_cases = [
        ("정상 케이스", "서울 날씨 알려줘"),
        ("에러 케이스 1", "뉴욕 날씨 알려줘"),  # 데이터 없음
        ("에러 케이스 2", "10을 0으로 나눠줘"),  # ZeroDivisionError
    ]
    
    for title, query in test_cases:
        print(f"\n{'='*60}")
        print(f"[{title}] {query}")
        print('='*60)
        
        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "tool_executed": False,
            "error_count": 0,
            "max_retries": 3
        })
        
        print(f"\n최종 답변: {result['messages'][-1].content}")
        print(f"에러 카운트: {result.get('error_count', 0)}")

'''
[왜 정규표현식으로 답변을 받나]
정규표현식 사용 (현재 방식)

소형 모델 (7B 이하)
빠른 프로토타이핑
간단한 도구

JSON 파싱 사용

대형 모델 (13B 이상)
복잡한 도구 (여러 파라미터)
프로덕션 환경
'''