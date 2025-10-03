from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from typing import TypedDict, Literal

class CustomerState(TypedDict):
    query: str
    category: str
    response: str
    confidence: float



def classify_agent(state: CustomerState) -> CustomerState:
    """고객 문의를 분류하는 에이전트"""
    # llm 설정
    llm = ChatOllama(
            model="devna0111-7b-q4",
            base_url="http://localhost:11434",
            temperature=0.15,
        )
    
    system_prompt = """당신은 고객 문의 분류 전문가입니다.
                        문의를 다음 카테고리로 분류하세요:
                        - technical: 기술적 문제, 버그, 오류
                        - billing: 결제, 환불, 요금
                        - general: 일반 문의, 기타

                        반드시 한 단어로만 답변하세요: technical, billing, general 중 하나"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"문의 내용: {state['query']}")
    ]
    
    response = llm.invoke(messages)
    category = response.content.strip().lower()
    
    # 유효성 검사
    if category not in ["technical", "billing", "general"]:
        category = "general"
    
    return {
        **state,
        "category": category, # state 전체를 언패킹 후 category와 confidence만 업데이트
        "confidence": 0.9  # 향후 평가LLM을 추가해서 LLM 응답에서 추출
    }
    
if __name__ == "__main__":
    # 테스트 1: 기술 문의
    test_state1 = {
        "query": "로그인이 안돼요. 에러 코드 500이 나와요",
        "category": "",
        "response": "",
        "confidence": 0.0
    }

    result1 = classify_agent(test_state1)
    print(f"문의: {result1['query']}")
    print(f"분류: {result1['category']}")
    print(f"신뢰도: {result1['confidence']}\n")

    # 테스트 2: 결제 문의
    test_state2 = {
        "query": "카드 결제가 실패했습니다",
        "category": "",
        "response": "",
        "confidence": 0.0
    }

    result2 = classify_agent(test_state2)
    print(f"문의: {result2['query']}")
    print(f"분류: {result2['category']}")
    print(f"신뢰도: {result2['confidence']}\n")

    # 테스트 3: 일반 문의
    test_state3 = {
        "query": "영업 시간이 어떻게 되나요?",
        "category": "",
        "response": "",
        "confidence": 0.0
    }

    result3 = classify_agent(test_state3)
    print(f"문의: {result3['query']}")
    print(f"분류: {result3['category']}")
    print(f"신뢰도: {result3['confidence']}")