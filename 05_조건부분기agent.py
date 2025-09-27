from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from typing import TypedDict

def create_conditional_agent():
    """조건부 분기가 있는 Langgraph Agent 생성"""
    print("조건부 분기 Agent 생성 시작")
    
    try:
        from langgraph.graph import StateGraph, END
        from langchain_community.llms import Ollama
        from typing import TypedDict # 딕셔너리.
        import re
        
        # 1. State 정의 - 더 많은 정보 추가
        class AgentState(TypedDict):
            messages: list
            current_step: str
            user_intent: str  # 사용자 의도 분류
            context: dict     # 추가 컨텍스트
            
        # 2. Ollama 모델 설정
        llm = Ollama(
            model="devna0111-7b-q4",
            base_url="http://localhost:11434",
            temperature=0.7,
        )
        
        # 3. 의도 분석 노드
        def analyze_intent_node(state: AgentState):
            """사용자 입력의 의도를 분석"""
            messages = state['messages']
            last_message = messages[-1] if messages else ""
            
            # 간단한 키워드 기반 의도 분석
            if any(word in last_message.lower() for word in ['안녕', '하이', 'hello']):
                intent = "greeting"
            elif any(word in last_message.lower() for word in ['질문', '궁금', '알고싶', '?']):
                intent = "question"
            elif any(word in last_message.lower() for word in ['도움', 'help', '헬프']):
                intent = "help"
            else:
                intent = "general"
            
            return {
                "messages": state['messages'],
                "current_step": "intent_analyzed",
                "user_intent": intent,
                "context": state.get('context', {})
            }
        
        # 4. 인사 처리 노드
        def greeting_node(state: AgentState):
            """인사 전용 처리"""
            response = "안녕하세요! 저는 조건부 분기 Agent입니다. 무엇을 도와드릴까요?"
            
            return {
                "messages": state['messages'] + [response],
                "current_step": "greeting_complete",
                "user_intent": state['user_intent'],
                "context": {"last_action": "greeting"}
            }
        
        # 5. 질문 처리 노드
        def question_node(state: AgentState):
            """질문 전용 처리"""
            messages = state['messages']
            user_question = messages[-1]
            
            prompt = f"다음 질문에 친절하고 자세히 답해주세요: {user_question}"
            response = llm.invoke(prompt)
            
            return {
                "messages": state['messages'] + [response],
                "current_step": "question_complete", 
                "user_intent": state['user_intent'],
                "context": {"last_action": "answered_question"}
            }
        
        # 6. 도움말 노드
        def help_node(state: AgentState):
            """도움말 전용 처리"""
            help_text = """
            📚 사용 가능한 명령어:
            - 인사: '안녕하세요', '하이' 등
            - 질문: '궁금한 것이 있어요', '질문이 있습니다' 등  
            - 도움: '도움이 필요해요', 'help' 등
            - 일반 대화: 자유롭게 대화하세요!
            """
            
            return {
                "messages": state['messages'] + [help_text],
                "current_step": "help_complete",
                "user_intent": state['user_intent'], 
                "context": {"last_action": "showed_help"}
            }
        
        # 7. 일반 대화 노드
        def general_chat_node(state: AgentState):
            """일반적인 대화 처리"""
            messages = state['messages']
            user_message = messages[-1]
            
            response = llm.invoke(user_message)
            
            return {
                "messages": state['messages'] + [response],
                "current_step": "chat_complete",
                "user_intent": state['user_intent'],
                "context": {"last_action": "general_chat"}
            }
        
        # 8. 라우터 함수 - 핵심!
        def route_by_intent(state: AgentState):
            """의도에 따라 다음 노드 결정"""
            intent = state['user_intent']
            
            if intent == "greeting":
                return "greeting"
            elif intent == "question":
                return "question" 
            elif intent == "help":
                return "help"
            else:
                return "general"
        
        # 9. Graph 생성
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("analyze", analyze_intent_node)
        workflow.add_node("greeting", greeting_node)
        workflow.add_node("question", question_node)
        workflow.add_node("help", help_node)
        workflow.add_node("general", general_chat_node)
        
        # 시작점 설정
        workflow.set_entry_point("analyze") # 의도 분석으로 시작
        
        # 조건부 엣지 추가 - 핵심!
        workflow.add_conditional_edges(
            "analyze",           # 시작 노드
            route_by_intent,     # 라우팅 함수
            {                    # 매핑
                "greeting": "greeting",
                "question": "question", 
                "help": "help",
                "general": "general"
            }
        )
        
        # 모든 처리 노드에서 END => END 포인트
        workflow.add_edge("greeting", END)
        workflow.add_edge("question", END)
        workflow.add_edge("help", END)
        workflow.add_edge("general", END)
        
        # 컴파일
        app = workflow.compile()
        print("조건부 분기 Agent 생성 완료")
        
        return app
        
    except Exception as e:
        print(f"조건부 분기 Agent 생성 실패: {e}")
        return None

def test_conditional_agent():
    """조건부 분기 Agent 테스트"""
    print("\n=== 조건부 분기 Agent 테스트 ===")
    
    agent = create_conditional_agent()
    if not agent:
        return
    
    # 테스트 케이스들
    test_cases = [
        "안녕하세요!",
        # "Langgraph에 대해 질문이 있어요",
        "도움이 필요해요",
        "오늘 날씨가 좋네요",
        "심재성이 누군 지 아나요?"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i} ---")
        print(f"입력: {test_input}")
        
        # Agent 실행
        result = agent.invoke({
            "messages": [test_input],
            "current_step": "start",
            "user_intent": "",
            "context": {}
        })
        
        print(f"의도 분류: {result['user_intent']}")
        print(f"응답: {result['messages'][-1]}...")  # 처음 100자만
        print(f"최종 단계: {result['current_step']}")

def explain_conditional_routing():
    """조건부 라우팅 개념 설명"""
    print("\n=== 조건부 라우팅 핵심 개념 ===")
    
    concepts = {
        "조건부 엣지 (Conditional Edge)": "입력에 따라 다른 노드로 분기",
        "라우터 함수 (Router Function)": "어떤 노드로 갈지 결정하는 함수",
        "의도 분석 (Intent Analysis)": "사용자 입력의 목적을 파악",
        "매핑 (Mapping)": "라우터 결과와 실제 노드를 연결"
    }
    
    for concept, description in concepts.items():
        print(f"• {concept}: {description}")
    
    print("\n=== 흐름 이해 ===")
    print("1. 사용자 입력 → analyze_intent_node")
    print("2. 의도 분석 → route_by_intent 함수")  
    print("3. 의도에 따라 분기 → greeting/question/help/general")
    print("4. 각 노드에서 처리 → END")
    
    print("\n=== 활용 예시 ===")
    print("- 챗봇: 질문/불만/칭찬 분류 처리")
    print("- 업무 자동화: 문서 유형별 처리")
    print("- 고객 지원: 문의 유형별 전문 상담")

if __name__ == "__main__":
    agent = create_conditional_agent()
    if agent:
        test_conditional_agent()
        explain_conditional_routing()