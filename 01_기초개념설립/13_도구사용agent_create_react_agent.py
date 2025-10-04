from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

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
    """수식을 계산합니다. 예: '2+3' 또는 '15*23'"""
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

# 도구 리스트
tools = [get_weather, calculate, get_current_time]

# LLM 설정 (Function Calling 지원 모델 필요!)
llm = ChatOllama(
    model="qwen2.5:3b",  # 도구 지원 모델
    base_url="http://localhost:11434",
    temperature=0
)

# Agent 생성 (단 한 줄!)
agent = create_react_agent(llm, tools)

# 테스트
if __name__ == "__main__":
    print("=== create_react_agent 테스트 ===\n")
    
    test_cases = [
        "서울 날씨 알려줘",
        "15 곱하기 23은?",
        "지금 몇 시야?",
        "서울 날씨와 현재 시간 둘 다 알려줘",
        "안녕하세요"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"[테스트 {i}] {query}")
        print('='*60)
        
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # 전체 메시지 흐름 출력 (디버깅용)
        print("\n[실행 과정]")
        for j, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
            print(f"  {j+1}. {msg_type}: {content_preview}...")
        
        # 최종 답변만 출력
        print(f"\n[최종 답변]")
        print(result["messages"][-1].content)