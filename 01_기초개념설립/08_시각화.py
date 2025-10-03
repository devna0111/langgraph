from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from agent.classification import classify_agent
from agent.agents import tech_support_agent, billing_agent, general_agent

# 상태 정의
class CustomerState(TypedDict):
    query: str
    category: str
    response: str
    confidence: float
    retry_count: int

# 재시도 노드
def retry_node(state: CustomerState) -> CustomerState:
    return {**state, "retry_count": state["retry_count"] + 1}

# 에스컬레이션 노드
def escalate_to_human(state: CustomerState) -> CustomerState:
    escalation_msg = f"[상담원 연결] 문의: {state['query']}"
    return {**state, "response": escalation_msg, "confidence": 1.0}

# 라우팅 함수들
def route_query(state: CustomerState) -> Literal["tech_support", "billing", "general"]:
    category = state["category"]
    if category == "technical":
        return "tech_support"
    elif category == "billing":
        return "billing"
    else:
        return "general"

def check_confidence(state: CustomerState) -> Literal["retry", "escalate", "done"]:
    conf = state["confidence"]
    retry_count = state["retry_count"]
    
    if conf >= 0.7:
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

workflow.add_conditional_edges(
    "classifier",
    route_query,
    {
        "tech_support": "tech_support",
        "billing": "billing",
        "general": "general"
    }
)

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

# 시각화
if __name__ == "__main__":
    # 방법 1: Mermaid 다이어그램 (텍스트)
    print("=== Mermaid 다이어그램 ===")
    print(app.get_graph().draw_mermaid())
    # 복사해서 https://mermaid.live 에 붙여넣기
    print("\n")
    
    # 방법 2: PNG 이미지 저장
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("img/workflow_graph.png", "wb") as f:
            f.write(png_data)
        print("✅ 그래프 이미지 저장됨: workflow_graph.png")
    except Exception as e:
        print(f"⚠️  PNG 저장 실패: {e}")
        print("Graphviz가 설치되지 않았을 수 있습니다.")
    
    # 방법 3: ASCII 아트 (간단한 텍스트 표현)
    print("\n=== ASCII 그래프 ===")
    # 간단한 텍스트 표현, 터미널에서 바로 확인
    print(app.get_graph().draw_ascii())