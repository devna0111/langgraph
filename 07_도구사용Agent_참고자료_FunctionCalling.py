def create_function_calling_agent():
    """Function Calling을 활용한 간단한 Agent"""
    print("Function Calling Agent 생성 시작")
    
    try:
        import ddgs as DDGS
        from langchain_community.llms import Ollama
        from langchain.agents import initialize_agent, Tool, AgentType
        from langchain.memory import ConversationBufferMemory
        
        # 1. 도구 함수들 정의
        def calculator_function(expression: str) -> str:
            """계산기 함수 - 수학 연산 수행"""
            try:
                # 안전한 계산만 허용
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "허용되지 않은 문자가 포함되어 있습니다"
                
                result = eval(expression)
                return f"계산 결과: {result}"
            except Exception as e:
                return f"계산 오류: {str(e)}"
        
        from duckduckgo_search import DDGS

        def search_function(query, max_results=5):
            with DDGS() as ddgs:
                results = ddgs.text(query)
                return "\n".join([r["body"] for r in results])
        
        # 2. LangChain Tool 객체로 래핑
        tools = [
            Tool(
                name="Calculator",
                func=calculator_function,
                description="수학 계산을 수행합니다. 예: '125 + 847' 또는 '10 * 5'"
            ),
            Tool(
                name="Search", 
                func=search_function,
                description="DuckDuckGo를 통해 정보를 검색합니다. 예: 'Python이란' 또는 '오늘 날씨'"
            )
        ]
        
        # 3. LLM 설정
        llm = Ollama(
            model="devna0111-7b-q4",
            base_url="http://localhost:11434",
            temperature=0.3,
        )
        
        # 4. 메모리 설정
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 5. Agent 초기화
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,  # 내부 과정 출력
            max_iterations=3,  # 최대 반복 횟수
            early_stopping_method="generate"
        )
        
        print("Function Calling Agent 생성 완료")
        return agent
        
    except Exception as e:
        print(f"Function Calling Agent 생성 실패: {e}")
        return None

def test_function_calling_agent():
    """Function Calling Agent 테스트"""
    print("\n=== Function Calling Agent 테스트 ===")
    
    agent = create_function_calling_agent()
    if not agent:
        return
    
    test_cases = [
        "125 + 847을 계산해주세요",
        # "Python 프로그래밍 언어에 대해 알려주세요",
        # "1000 나누기 25는?",
        "내일 날씨를 검색해주세요",
        "안녕하세요!"
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i} ---")
        print(f"질문: {question}")
        
        try:
            # Agent 실행
            response = agent.run(question)
            print(f"답변: {response}")
            
        except Exception as e:
            print(f"오류 발생: {e}")
        
        print("-" * 50)

def simple_function_calling_demo():
    """더 간단한 Function Calling 데모"""
    print("\n=== 간단한 Function Calling 데모 ===")
    
    from langchain_community.llms import Ollama
    
    llm = Ollama(
        model="devna0111-7b-q4",
        base_url="http://localhost:11434",
        temperature=0.3,
    )
    
    def process_with_function_calling(user_input: str):
        """Function Calling 로직"""
        
        # 1. 도구 필요성 판단
        if any(op in user_input for op in ['+', '-', '*', '/', '계산']):
            tool_needed = "calculator"
        elif any(word in user_input for word in ['검색', '찾아', '알려줘', '?']):
            tool_needed = "search"
        else:
            tool_needed = None
        
        # 2. 도구 실행
        if tool_needed == "calculator":
            import re
            # 계산식 추출
            calc_pattern = r'[\d+\-*/.()\s]+'
            matches = re.findall(calc_pattern, user_input)
            if matches:
                expression = ''.join(matches).strip()
                try:
                    result = eval(expression)
                    tool_result = f"계산 결과: {result}"
                except:
                    tool_result = "계산 오류"
            else:
                tool_result = "계산식을 찾을 수 없습니다"
        
        elif tool_needed == "search":
            # 간단한 가짜 검색
            search_db = {
                "python": "Python은 1991년에 만들어진 프로그래밍 언어입니다",
                "ai": "AI는 인공지능 기술로 기계가 인간처럼 사고하게 만드는 기술입니다"
            }
            
            query = user_input.lower()
            tool_result = "관련 정보를 찾지 못했습니다"
            for key, value in search_db.items():
                if key in query:
                    tool_result = value
                    break
        
        else:
            tool_result = None
        
        # 3. 최종 응답 생성
        if tool_result:
            prompt = f"""
                    사용자 질문: {user_input}
                    도구 실행 결과: {tool_result}

                    위 정보를 바탕으로 자연스러운 답변을 해주세요.
                    """
            response = llm.invoke(prompt)
        else:
            response = llm.invoke(user_input)
        
        return response, tool_needed, tool_result
    
    # 테스트
    test_inputs = [
        "100 + 200은?",
        "Python에 대해 알려주세요",
        "안녕하세요"
    ]
    
    for test_input in test_inputs:
        print(f"\n입력: {test_input}")
        response, tool, result = process_with_function_calling(test_input)
        print(f"사용된 도구: {tool or '없음'}")
        if result:
            print(f"도구 결과: {result}")
        print(f"최종 응답: {response}")

def compare_approaches():
    """Langgraph vs Function Calling 비교"""
    print("\n=== Langgraph vs Function Calling 비교 ===")
    
    comparison = {
        "구현 복잡도": {
            "Function Calling": "⭐⭐ 간단함",
            "Langgraph": "⭐⭐⭐⭐ 복잡함"
        },
        "유연성": {
            "Function Calling": "⭐⭐ 제한적",
            "Langgraph": "⭐⭐⭐⭐⭐ 매우 높음"
        },
        "상태 관리": {
            "Function Calling": "⭐ 기본적",
            "Langgraph": "⭐⭐⭐⭐⭐ 강력함"
        },
        "적합한 용도": {
            "Function Calling": "간단한 도구 사용, 1-2단계 작업",
            "Langgraph": "복잡한 워크플로우, 멀티 에이전트, 조건부 분기"
        }
    }
    
    for aspect, details in comparison.items():
        print(f"\n📊 {aspect}:")
        for approach, rating in details.items():
            print(f"  {approach}: {rating}")
    
    print("\n💡 결론:")
    print("  - 간단한 도구 사용 → Function Calling 추천")
    print("  - 복잡한 AI 워크플로우 → Langgraph 추천")

if __name__ == "__main__":
    # LangChain Agent 방식
    agent = create_function_calling_agent()
    if agent:
        test_function_calling_agent()
    
    # 간단한 구현 방식
    # simple_function_calling_demo()
    
    # 비교 분석
    compare_approaches()