def create_tool_agent():
    """도구 사용 Agent 생성"""
    print("도구 사용 Agent 생성 시작")
    
    try:
        from langgraph.graph import StateGraph, END
        from langchain_community.llms import Ollama
        from typing import TypedDict
        import requests
        import math
        import re
        
        # 필요한 라이브러리 설치 확인
        try:
            from bs4 import BeautifulSoup
            print("✅ BeautifulSoup 사용 가능")
        except ImportError:
            print("⚠️  BeautifulSoup이 설치되지 않음. pip install beautifulsoup4 필요")
            print("현재는 기본 검색으로 대체됩니다.")
        
        try:
            import requests
            print("✅ requests 사용 가능")
        except ImportError:
            print("⚠️  requests가 설치되지 않음. pip install requests 필요")
        
        # 1. State 정의
        class ToolAgentState(TypedDict):
            messages: list
            tool_call_needed: bool
            tool_name: str
            tool_result: str
            
        # 2. LLM 설정
        llm = Ollama(
            model="devna0111-7b-q4",
            base_url="http://localhost:11434",
            temperature=0.3,
        )
        
        # 3. 도구 함수들 정의 (계산기 + 검색만)
        def calculator_tool(expression: str):
            """계산기 도구"""
            try:
                # 안전한 수학 계산만 허용
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "계산식에 허용되지 않은 문자가 있습니다"
                
                result = eval(expression)
                return f"계산 결과: {result}"
            except:
                return "계산 오류가 발생했습니다"
        
        def search_tool(query: str, max_results=5):
            """DuckDuckGo 실제 검색"""
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = ddgs.text(query)
                return results
        
        # 4. 도구 판단 노드 (계산 + 검색만)
        def analyze_tool_need_node(state: ToolAgentState):
            """사용자 입력에서 필요한 도구 판단"""
            messages = state['messages']
            if not messages:
                return state
            
            user_message = messages[-1].lower()
            
            # 계산 필요 판단
            if any(op in user_message for op in ['+', '-', '*', '/', '계산', '더하기', '곱하기', '나누기', '빼기']):
                return {
                    "messages": state['messages'],
                    "tool_call_needed": True,
                    "tool_name": "calculator",
                    "tool_result": ""
                }
            
            # 검색 필요 판단 (날씨도 검색으로 처리)
            elif any(word in user_message for word in ['검색', 'search', '찾아', '알려줘', '?', '날씨', '정보']):
                return {
                    "messages": state['messages'],
                    "tool_call_needed": True,
                    "tool_name": "search",
                    "tool_result": ""
                }
            
            # 도구 불필요
            else:
                return {
                    "messages": state['messages'],
                    "tool_call_needed": False,
                    "tool_name": "",
                    "tool_result": ""
                }
        
        # 5. 도구 실행 노드 (계산 + 검색만)
        def execute_tool_node(state: ToolAgentState):
            """도구 실행"""
            messages = state['messages']
            tool_name = state['tool_name']
            user_message = messages[-1]
            
            if tool_name == "calculator":
                # 숫자와 연산자 추출
                calc_pattern = r'[\d+\-*/.()\s]+'
                matches = re.findall(calc_pattern, user_message)
                if matches:
                    expression = ''.join(matches).strip()
                    result = calculator_tool(expression)
                else:
                    result = "계산식을 찾을 수 없습니다"
            
            elif tool_name == "search":
                # 검색어 추출 (키워드 제거 후)
                query = user_message.replace("검색", "").replace("찾아", "").replace("알려줘", "").strip()
                if not query:
                    query = user_message  # 원본 메시지 사용
                result = search_tool(query)
            
            else:
                result = "도구 실행 오류"
            
            return {
                "messages": state['messages'],
                "tool_call_needed": state['tool_call_needed'],
                "tool_name": tool_name,
                "tool_result": result
            }
        
        # 6. 응답 생성 노드
        def generate_response_node(state: ToolAgentState):
            """도구 결과를 바탕으로 응답 생성"""
            messages = state['messages']
            tool_result = state['tool_result']
            user_message = messages[-1]
            
            if state['tool_call_needed']:
                # 도구 결과를 자연스럽게 변환
                prompt = f"""
                                사용자 질문: {user_message}
                                도구 실행 결과: {tool_result}

                                위 정보를 바탕으로 자연스럽고 친근한 답변을 만들어주세요.
                            """
                response = llm.invoke(prompt)
            else:
                # 일반 대화
                response = llm.invoke(user_message)
            
            return {
                "messages": messages + [response],
                "tool_call_needed": False,
                "tool_name": "",
                "tool_result": ""
            }
        
        # 7. 라우터 함수
        def route_by_tool_need(state: ToolAgentState):
            """도구 필요성에 따라 라우팅"""
            if state['tool_call_needed']:
                return "execute_tool"
            else:
                return "respond"
        
        # 8. Graph 구성
        workflow = StateGraph(ToolAgentState)
        
        # 노드 추가
        workflow.add_node("analyze", analyze_tool_need_node)
        workflow.add_node("execute_tool", execute_tool_node)
        workflow.add_node("respond", generate_response_node)
        
        # 흐름 설정
        workflow.set_entry_point("analyze")
        
        # 조건부 엣지
        workflow.add_conditional_edges(
            "analyze",
            route_by_tool_need,
            {
                "execute_tool": "execute_tool",
                "respond": "respond"
            }
        )
        
        workflow.add_edge("execute_tool", "respond")
        workflow.add_edge("respond", END)
        
        # 컴파일
        app = workflow.compile()
        print("도구 사용 Agent 생성 완료")
        
        return app
        
    except Exception as e:
        print(f"도구 사용 Agent 생성 실패: {e}")
        return None

def test_tool_agent():
    """도구 사용 Agent 테스트"""
    print("\n=== 도구 사용 Agent 테스트 ===")
    
    agent = create_tool_agent()
    if not agent:
        return
    
    test_cases = [
        "125 + 847은?",
        "9월 28일 서울 날씨 검색",
        # "Python 프로그래밍 언어 검색",
        "안녕하세요!"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i} ---")
        print(f"입력: {test_input}")
        
        result = agent.invoke({
            "messages": [test_input],
            "tool_call_needed": False,
            "tool_name": "",
            "tool_result": ""
        })
        
        print(f"도구 사용: {result.get('tool_name', '없음')}")
        print(f"응답: {result['messages'][-1]}")

def explain_tool_concepts():
    """도구 사용 개념 설명"""
    print("\n=== 도구 사용 Agent 핵심 ===")
    
    print("🔧 도구 정의:")
    print("  - calculator_tool: 수학 계산 (+, -, *, /)")
    print("  - search_tool: DuckDuckGo 실제 검색")
    
    print("\n🌐 검색 API:")
    print("  - DuckDuckGo API")
    
    print("\n🤖 처리 흐름:")
    print("  1. 입력 분석 → 계산 vs 검색 vs 일반대화")
    print("  2. 도구 실행 → 계산기 또는 실시간 검색")
    print("  3. 결과 → LLM이 자연어로 변환")

if __name__ == "__main__":
    agent = create_tool_agent()
    if agent:
        test_tool_agent()
        explain_tool_concepts()