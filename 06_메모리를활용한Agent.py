def create_memory_agent():
    """메모리를 가진 Langgraph Agent 생성"""
    print("메모리 Agent 생성 시작")
    
    try:
        from langgraph.graph import StateGraph, END
        from langchain_community.llms import Ollama
        from langchain.memory import ConversationBufferMemory
        from typing import TypedDict, List
        import json
        from datetime import datetime
        
        # LangChain 메모리 객체 생성
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 1. 메모리가 강화된 State 정의
        class MemoryAgentState(TypedDict):
            messages: List[dict]          # 대화 이력 (구조화)
            langchain_memory: str         # LangChain 메모리 문자열
            user_profile: dict           # 사용자 정보 누적
            session_context: dict        # 세션별 맥락 정보
            current_topic: str           # 현재 대화 주제
            turn_count: int             # 대화 턴 수
            
        # 2. Ollama 모델 설정
        llm = Ollama(
            model="devna0111-7b-q4",
            base_url="http://localhost:11434",
            temperature=0.7,
        )
        
        # 3. 메모리 초기화 노드
        def initialize_memory_node(state: MemoryAgentState):
            """메모리 초기 설정"""
            return {
                "messages": state.get('messages', []),
                "langchain_memory": "",  # 초기값은 빈 문자열
                "user_profile": state.get('user_profile', {}),
                "session_context": {
                    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                },
                "current_topic": "일반대화",
                "turn_count": 0
            }
        
        # 4. 사용자 정보 추출 노드  
        def extract_user_info_node(state: MemoryAgentState):
            """사용자 입력에서 정보 추출하여 프로필 업데이트"""
            messages = state['messages']
            if not messages:
                return state
                
            user_message = messages[-1]['content']
            user_profile = state['user_profile'].copy()
            
            # 간단한 정보 추출 (실제로는 더 정교한 NLP 필요)
            if '이름' in user_message and ('저는' in user_message or '제가' in user_message):
                # "저는 김철수입니다" 같은 패턴
                try:
                    name_part = user_message.split('저는')[1].split('입니다')[0].strip()
                    if name_part:
                        user_profile['name'] = name_part
                except:
                    pass
            
            if any(hobby in user_message for hobby in ['좋아해', '취미', '관심']):
                if 'hobbies' not in user_profile:
                    user_profile['hobbies'] = []
                user_profile['hobbies'].append(user_message)
            
            if any(work in user_message for work in ['직업', '일', '회사']):
                user_profile['work_mention'] = user_message
            
            return {
                "messages": state['messages'],
                "langchain_memory": state['langchain_memory'],
                "user_profile": user_profile,
                "session_context": state['session_context'],
                "current_topic": state['current_topic'],
                "turn_count": state['turn_count']
            }
        
        # 5. LangChain 메모리 업데이트 노드
        def update_langchain_memory_node(state: MemoryAgentState):
            """LangChain 메모리를 업데이트하고 맥락 구성"""
            messages = state['messages']
            user_profile = state['user_profile']
            turn_count = state['turn_count'] + 1
            
            # LangChain 메모리에 대화 추가
            if messages:
                latest_message = messages[-1]
                if latest_message['role'] == 'user':
                    memory.chat_memory.add_user_message(latest_message['content'])
                elif latest_message['role'] == 'assistant':
                    memory.chat_memory.add_ai_message(latest_message['content'])
            
            # LangChain 메모리에서 대화 이력 가져오기
            memory_content = memory.load_memory_variables({})
            langchain_memory = str(memory_content.get('chat_history', []))
            
            # 현재 주제 추측
            current_topic = "일반대화"
            if messages:
                last_message = messages[-1]['content'].lower()
                if any(word in last_message for word in ['날씨', 'weather']):
                    current_topic = "날씨"
                elif any(word in last_message for word in ['음식', '요리', '먹']):
                    current_topic = "음식"
                elif any(word in last_message for word in ['일', '직업', '회사']):
                    current_topic = "업무"
            
            return {
                "messages": state['messages'],
                "langchain_memory": langchain_memory,
                "user_profile": state['user_profile'],
                "session_context": state['session_context'],
                "current_topic": current_topic,
                "turn_count": turn_count
            }
        
        # 6. 맥락 기반 응답 생성 노드
        def contextual_response_node(state: MemoryAgentState):
            """LangChain 메모리와 맥락을 고려한 응답 생성"""
            messages = state['messages']
            langchain_memory = state['langchain_memory']
            user_profile = state['user_profile']
            current_topic = state['current_topic']
            turn_count = state['turn_count']
            
            if not messages:
                response_content = "안녕하세요! 저는 LangChain 메모리를 활용하는 AI입니다."
                # 메모리에 첫 인사 추가
                memory.chat_memory.add_ai_message(response_content)
            else:
                user_message = messages[-1]['content']
                
                # LangChain 메모리를 활용한 맥락이 풍부한 프롬프트 구성
                context_prompt = f"""
LangChain 대화 메모리:
{langchain_memory}

사용자 정보: {json.dumps(user_profile, ensure_ascii=False)}
현재 주제: {current_topic}
대화 턴: {turn_count}

현재 사용자 메시지: {user_message}

위의 LangChain 메모리와 맥락을 고려하여 자연스럽고 연관성 있는 답변을 해주세요.
이전 대화의 흐름을 이어가며 답변하세요.
"""
                
                response_content = llm.invoke(context_prompt)
                # 응답을 메모리에 추가
                memory.chat_memory.add_ai_message(response_content)
            
            # 새 메시지를 대화 이력에 추가
            new_message = {"role": "assistant", "content": response_content}
            updated_messages = messages + [new_message]
            
            # 업데이트된 메모리 가져오기
            updated_memory = str(memory.load_memory_variables({}).get('chat_history', []))
            
            return {
                "messages": updated_messages,
                "langchain_memory": updated_memory,
                "user_profile": state['user_profile'],
                "session_context": state['session_context'],
                "current_topic": state['current_topic'],
                "turn_count": state['turn_count']
            }
        
        # 7. Graph 구성
        workflow = StateGraph(MemoryAgentState)
        
        # 노드 추가
        workflow.add_node("initialize", initialize_memory_node)
        workflow.add_node("extract_info", extract_user_info_node)
        workflow.add_node("update_memory", update_langchain_memory_node)
        workflow.add_node("respond", contextual_response_node)
        
        # 흐름 설정
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "extract_info")
        workflow.add_edge("extract_info", "update_memory") 
        workflow.add_edge("update_memory", "respond")
        workflow.add_edge("respond", END)
        
        # 컴파일
        app = workflow.compile()
        print("메모리 Agent 생성 완료")
        
        return app
        
    except Exception as e:
        print(f"메모리 Agent 생성 실패: {e}")
        return None

def test_memory_agent():
    """메모리 Agent 연속 대화 테스트"""
    print("\n=== 메모리 Agent 연속 대화 테스트 ===")
    
    agent = create_memory_agent()
    if not agent:
        return
    
    # 연속 대화 시뮬레이션
    conversation = [
        "안녕하세요! 저는 김철수입니다.",
        "저는 프로그래밍을 좋아해요.",
        "제가 뭘 좋아한다고 했죠?"
    ]
    
    # 상태 초기화
    current_state = {
        "messages": [],
        "langchain_memory": "",
        "user_profile": {},
        "session_context": {},
        "current_topic": "일반대화",
        "turn_count": 0
    }
    
    for i, user_input in enumerate(conversation, 1):
        print(f"\n--- 대화 턴 {i} ---")
        print(f"사용자: {user_input}")
        
        # 사용자 메시지를 상태에 추가
        user_message = {"role": "user", "content": user_input}
        current_state["messages"].append(user_message)
        
        # Agent 실행
        result = agent.invoke(current_state)
        
        # 상태 업데이트 (메모리 유지)
        current_state = result
        
        # 응답 출력
        assistant_message = result["messages"][-1]["content"]
        print(f"Assistant: {assistant_message}")
        
        # 메모리 상태 출력
        print(f"현재 주제: {result['current_topic']}")
        print(f"사용자 정보: {result['user_profile']}")
        print(f"대화 턴: {result['turn_count']}")
        print(f"LangChain 메모리 크기: {len(result['langchain_memory'])} 문자")

def explain_memory_concepts():
    """메모리 개념 설명"""
    print("\n=== 메모리 Agent 핵심 개념 ===")
    
    concepts = {
        "LangChain Memory": "LangChain의 ConversationBufferMemory 활용",
        "자동 메모리 관리": "대화 자동 저장 및 불러오기",
        "사용자 프로필": "사용자 정보를 누적하여 개인화",
        "세션 맥락": "현재 세션의 맥락 정보 관리", 
        "주제 추적": "대화 주제의 변화를 감지하고 추적",
        "통합 메모리": "LangChain + 커스텀 메모리 조합"
    }
    
    for concept, description in concepts.items():
        print(f"• {concept}: {description}")
    
    print("\n=== LangChain 메모리 활용법 ===")
    print("1. ConversationBufferMemory로 대화 자동 저장")
    print("2. chat_memory.add_user_message() - 사용자 메시지 추가")
    print("3. chat_memory.add_ai_message() - AI 응답 추가")
    print("4. load_memory_variables() - 저장된 대화 불러오기")
    print("5. 커스텀 State와 LangChain 메모리 통합 활용")
    
    print("\n=== 실무 활용 ===")
    print("- 개인 비서: 사용자 선호도 학습")
    print("- 고객 상담: 이전 문의 이력 활용") 
    print("- 교육 봇: 학습 진도 추적")
    print("- 의료 상담: 증상 변화 추적")

if __name__ == "__main__":
    agent = create_memory_agent()
    if agent:
        test_memory_agent()
        explain_memory_concepts()