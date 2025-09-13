def create_simple_agent() :
    """ 간단한 Langgraph Agent 생성 """
    print("간단한 Langgraph Agent 생성 시작")
    
    try :
        from langgraph.graph import StateGraph, END
        from langchain_community.llms import Ollama
        from typing import TypedDict
        
        # 1. state 정의
        class AgentState(TypedDict):
            messages : list
            current_step : str
            
        # 2. Ollama 모델 설정
        llm = Ollama(
            model = "devna0111-7b-q4",
            base_url = "http://localhost:11434",
            temperature = 0.7,
        )
        
        # 3. Agent 함수 정의
        def chat_node(state : AgentState):
            '''chat node'''
            messages = state['messages']
            last_message = messages[-1] if messages else '안녕하세요'
            
            # llm 호출
            response = llm.invoke(last_message)
            
            # state 업데이트
            return {"messages" : messages + [response],
                    "current_step" : "chat_complete"}
        
        # 4. graph 생성
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("chat", chat_node)
        
        # 시작점 설정
        workflow.set_entry_point("chat")
        
        # 종료점 설정
        workflow.add_edge("chat", END)
        
        # 그래프 컴파일
        app = workflow.compile()
        print("간단한 Langgraph Agent 생성 완료")
        
        # 5. 테스트 실행
        test_input = {
            'messages' : ['Langgraph에 대해 간단히 설명해주세요'],
            "current_step" : "start"
        }
        
        print("agent 테스트 실행..")
        result = app.invoke(test_input)
        
        print("agent 테스트 완료")
        print(f"응답 : {result['messages'][-1]}")
        
        return app
    
    except Exception as e :
        print(f"간단한 Langgraph Agent 생성 실패 : {e}")
        return None
    
def explain_langgraph() :
    ''' Langgraph 설명 '''
    print("Langgraph 핵심 개념")
    
    concepts = {
        "StateGraph": "상태를 가진 그래프. Agent의 메모리 역할",
        "Node": "작업을 수행하는 함수. 각 단계별 처리 담당", 
        "Edge": "Node 간 연결. 작업 흐름 정의",
        "State": "Agent가 기억하는 정보. 대화 내용, 중간 결과 등",
        "END": "그래프 종료 지점"
    }
    
    print("주요 개념들:")
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n기본 구조:")
    print("  1. State 정의 (무엇을 기억할지)")
    print("  2. Node 함수 작성 (각 단계별 처리)")
    print("  3. Graph 구성 (흐름 설계)")
    print("  4. 컴파일 및 실행")

if __name__ == "__main__":
    agent = create_simple_agent()
    if agent :
        explain_langgraph()