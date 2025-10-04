from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from typing import TypedDict

class CustomerState(TypedDict):
    query: str
    category: str
    response: str
    confidence: float
    
def get_llm(model="devna0111-7b-q4",temperature=0.7) :
    llm = ChatOllama(
        model=model,
        base_url="http://localhost:11434",
        temperature=temperature, # 현재 RAG 등 참고자료가 없어 내용 응답이 불가하여 실습 간 창의적 대답을 유도
    )
    return llm

def evaluate_confidence(query: str, category: str, response: str) -> float:
    """답변의 신뢰도를 평가하는 함수"""
    llm = get_llm(temperature=0.1)
    eval_prompt = f"""
                    다음 고객 문의와 답변을 평가하세요.

                    문의 유형: {category}
                    고객 문의: {query}
                    제공된 답변: {response}

                    다음 기준으로 평가하세요:
                    1. 문의와 답변의 관련성 (0-0.4점)
                    2. 답변의 구체성과 명확성 (0-0.3점)
                    3. 답변의 완성도 (0-0.3점)

                    **반드시 0.0에서 1.0 사이의 숫자만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.**

                    신뢰도 점수:
                    """

    messages = [HumanMessage(content=eval_prompt)]
    
    try:
        result = llm.invoke(messages)
        score_text = result.content.strip()
        
        # 숫자만 추출
        import re
        numbers = re.findall(r'0?\.\d+|[01]\.0|[01]', score_text)
        
        if numbers:
            score = float(numbers[0])
            return max(0.0, min(1.0, score))  # 0~1 범위로 제한
        else:
            return 0.5  # 파싱 실패시 중간값
            
    except Exception as e:
        print(f"평가 오류: {e}")
        return 0.5

def tech_support_agent(state: CustomerState) -> CustomerState:
    """기술 지원 전문 에이전트"""
    llm = get_llm()
    system_prompt = """
                        당신은 기술 지원 전문가입니다.
                        기술적 문제를 진단하고 해결 방법을 제시하세요.
                        단계별로 명확하게 설명하세요.
                        답변은 친화적이고 간결하게 합니다.
                    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"고객 문의: {state['query']}")
    ]
    
    response = llm.invoke(messages)
    confidence = evaluate_confidence(state['query'], state['category'], response.content)
    return {
        **state,
        "response": response.content,
        "confidence":confidence
    }

def billing_agent(state: CustomerState) -> CustomerState:
    """결제/청구 전문 에이전트"""
    llm = get_llm()
    system_prompt = """
                    당신은 결제 및 청구 전문가입니다.
                    결제 문제를 확인하고 해결 방법을 안내하세요.
                    환불 정책과 절차를 명확히 설명하세요.
                    답변은 친화적이고 간결하게 합니다.
                    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"고객 문의: {state['query']}")
    ]
    
    response = llm.invoke(messages)
    confidence = evaluate_confidence(state['query'], state['category'], response.content)
    
    return {
        **state,
        "response": response.content,
        "confidence":confidence
    }

def general_agent(state: CustomerState) -> CustomerState:
    """일반 문의 에이전트"""
    llm = get_llm()
    system_prompt = """
                        당신은 고객 서비스 담당자입니다.
                        친절하고 정중하게 일반 문의에 답변하세요.
                        답변은 친화적이고 간결하게 합니다.
                    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"고객 문의: {state['query']}")
    ]
    
    response = llm.invoke(messages)
    confidence = evaluate_confidence(state['query'], state['category'], response.content)
    
    return {
        **state,
        "response": response.content,
        "confidence":confidence
    }

# 테스트
if __name__ == "__main__":
    # # 테스트 1: 기술 지원
    # tech_state = {
    #     "query": "로그인이 안돼요. 에러 코드 500이 나와요",
    #     "category": "technical",
    #     "response": "",
    #     "confidence": 0.0
    # }
    # result = tech_support_agent(tech_state)
    # print(f"[기술 지원]\n문의: {result['query']}\n답변: {result['response']}\n")
    
    # # 테스트 2: 결제
    # billing_state = {
    #     "query": "카드 결제가 실패했습니다",
    #     "category": "billing",
    #     "response": "",
    #     "confidence": 0.0
    # }
    # result = billing_agent(billing_state)
    # print(f"[결제 지원]\n문의: {result['query']}\n답변: {result['response']}\n")
    
    # 테스트 3: 일반
    general_state = {
        "query": "영업 시간이 어떻게 되나요?",
        "category": "general",
        "response": "",
        "confidence": 0.0
    }
    result = general_agent(general_state)
    print(f"[일반 문의]\n문의: {result['query']}\n답변: {result['response']}\n신뢰도: {result['confidence']:.2f}")