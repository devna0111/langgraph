'''
체크포인트_상태저장
 - 서버 재시작 시 대화 이어가기
 - 긴 작업 중단 후 나중에 재개
 - 사용자별 세션관리 

 # 저장
    app.invoke(state, config={"configurable": {"thread_id": "user123"}})

 # 나중에 같은 thread_id로 이어서
    app.invoke(new_input, config={"configurable": {"thread_id": "user123"}})
'''
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # 핵심 모듈
from typing import TypedDict, Literal

# 상태 정의
class CustomerState(TypedDict):
    query: str
    category: str
    response: str
    confidence: float
    retry_count: int

# 간단한 에이전트 (테스트용)
def classify_agent(state: CustomerState) -> CustomerState:
    query = state["query"].lower()
    
    if "로그인" in query or "에러" in query:
        category = "technical"
    elif "결제" in query or "환불" in query:
        category = "billing"
    else:
        category = "general"
    
    return {**state, "category": category}

def answer_agent(state: CustomerState) -> CustomerState:
    category = state["category"]
    responses = {
        "technical": "기술 지원팀이 도와드리겠습니다.",
        "billing": "결제 관련 안내드리겠습니다.",
        "general": "일반 문의 답변드립니다."
    }
    
    return {
        **state,
        "response": responses[category],
        "confidence": 0.85
    }

# 라우팅
def route_to_answer(state: CustomerState) -> Literal["answer"]:
    return "answer"

# 그래프 생성
workflow = StateGraph(CustomerState)

workflow.add_node("classifier", classify_agent)
workflow.add_node("answer", answer_agent)

workflow.set_entry_point("classifier")
workflow.add_conditional_edges("classifier", route_to_answer, {"answer": "answer"})
workflow.add_edge("answer", END)

# 체크포인터 설정 (메모리 저장)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

try:
    img = "img/workflow_graph_체크포인트_상태저장.png"
    png_data = app.get_graph().draw_mermaid_png()
    with open(img, "wb") as f:
        f.write(png_data)
    print(f"그래프 이미지 저장됨: {img}")
except Exception as e:
    print(f"⚠️  PNG 저장 실패: {e}")
    print("Graphviz가 설치되지 않았을 수 있습니다.")

# 테스트
if __name__ == "__main__":
    # 고객 A의 대화
    print("=== 고객 A - 첫 번째 문의 ===")
    config_a = {"configurable": {"thread_id": "customer_A"}}
    
    result_a1 = app.invoke({
        "query": "로그인이 안돼요",
        "category": "",
        "response": "",
        "confidence": 0.0,
        "retry_count": 0
    }, config=config_a)
    
    print(f"문의: {result_a1['query']}")
    print(f"분류: {result_a1['category']}")
    print(f"답변: {result_a1['response']}\n")
    
    # 고객 B의 대화 (별도 세션)
    print("=== 고객 B - 첫 번째 문의 ===")
    config_b = {"configurable": {"thread_id": "customer_B"}}
    
    result_b1 = app.invoke({
        "query": "카드 결제가 실패했어요",
        "category": "",
        "response": "",
        "confidence": 0.0,
        "retry_count": 0
    }, config=config_b)
    
    print(f"문의: {result_b1['query']}")
    print(f"분류: {result_b1['category']}")
    print(f"답변: {result_b1['response']}\n")
    
    # 저장된 상태 확인
    print("=== 저장된 상태 확인 ===")
    
    # 고객 A의 저장된 상태 가져오기
    snapshot_a = app.get_state(config_a)
    print(f"고객 A 마지막 상태: {snapshot_a.values}")
    
    # 고객 B의 저장된 상태 가져오기
    snapshot_b = app.get_state(config_b)
    print(f"고객 B 마지막 상태: {snapshot_b.values}")
