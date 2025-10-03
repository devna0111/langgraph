from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
# agent 모듈 => agent 폴더 속에 구분
from agent.classification import classify_agent
from agent.agents import tech_support_agent, billing_agent, general_agent

# 상태 정의
class CustomerState(TypedDict):
    query: str
    category: str
    response: str
    confidence: float
    retry_count: int # retry 횟수

# confidence 값에 따라 재시도 하는 노드
def retry_node(state: CustomerState) -> CustomerState:
    print(f"신뢰도 낮음 ({state['confidence']:.2f}) - 재시도 중...")
    return {
        **state,
        "retry_count": state["retry_count"] + 1
    }

# 에스컬레이션 노드
def escalate_to_human(state: CustomerState) -> CustomerState:
    print(f"실제였다면 상담원 연결 노드로 상담원 연결 ({state['retry_count']}회 재시도 실패)")
    
    escalation_msg = f"""[자동 답변 실패]

                            고객 문의: {state['query']}
                            분류: {state['category']}
                            신뢰도: {state['confidence']:.2f}

                            010-1234-5678로 전화 연결 부탁드립니다.
                        """
    
    return {
        **state,
        "response": escalation_msg,
        "confidence": 1.0
    }

# 조건부 라우팅 : 분류를 위함
def route_query(state: CustomerState) -> Literal["tech_support", "billing", "general"]:
    category = state["category"]
    if category == "technical":
        return "tech_support"
    elif category == "billing":
        return "billing"
    else:
        return "general"

# 조건부 라우팅: 신뢰도 체크
def check_confidence(state: CustomerState) -> Literal["retry", "escalate", "done"]:
    confidence = state["confidence"]
    retry_count = state["retry_count"]
    
    if confidence >= 0.7:
        return "done"
    elif retry_count < 2:
        return "retry"
    else:
        return "escalate"

# 그래프 생성
workflow = StateGraph(CustomerState)

# 노드 추가
workflow.add_node("classifier", classify_agent)
workflow.add_node("tech_support", tech_support_agent)
workflow.add_node("billing", billing_agent)
workflow.add_node("general", general_agent)
workflow.add_node("retry", retry_node)
workflow.add_node("escalate", escalate_to_human)

# 엣지 설정
workflow.set_entry_point("classifier")

# 조건부 엣지
workflow.add_conditional_edges(
    "classifier",           # 시작 노드
    route_query,            # 라우팅 함수
    {
        "tech_support": "tech_support",
        "billing": "billing",
        "general": "general"
    }
)

# 각 에이전트 후 신뢰도 체크
for agent in ["tech_support", "billing", "general"]:
    workflow.add_conditional_edges(
        agent,
        check_confidence,
        {
            "done": END,
            "retry": "retry",
            "escalate": "escalate"
        }
    )

# 재시도는 분류기로 다시 (카테고리는 유지)
workflow.add_conditional_edges(
    "retry",
    route_query,
    {
        "tech_support": "tech_support",
        "billing": "billing",
        "general": "general"
    }
)

workflow.add_edge("escalate", END)

# 컴파일
app = workflow.compile()

if __name__ == "__main__":
    queries = [
        "로그인이 안돼요. 에러 500 나와요",
        "카드 결제가 실패했어요",
        "영업 시간 알려주세요"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"문의: {query}")
        print(f"{'='*60}")
        
        result = app.invoke({
            "query": query,
            "category": "",
            "response": "",
            "confidence": 0.0,
            "retry_count": 0  # 초기값
        })
        
        print(f"\n분류: {result['category']}")
        print(f"신뢰도: {result['confidence']:.2f}")
        print(f"재시도: {result['retry_count']}회")
        print(f"답변: {result['response'][:150]}...")