from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Literal

# 상태 정의
class CustomerState(TypedDict):
    query: str
    category: str
    response: str
    confidence: float
    retry_count: int

# 간단한 에이전트들
def classify_agent(state: CustomerState) -> CustomerState:
    query = state["query"].lower()
    
    if "로그인" in query or "에러" in query:
        category = "technical"
    elif "결제" in query or "환불" in query:
        category = "billing"
    else:
        category = "general"
    
    print(f"✅ 분류 완료: {category}")
    return {**state, "category": category}

def answer_agent(state: CustomerState) -> CustomerState:
    category = state["category"]
    responses = {
        "technical": "기술 지원팀이 도와드리겠습니다. 단계별로 안내해드릴게요.",
        "billing": "결제 관련 안내드리겠습니다. 문제를 확인하겠습니다.",
        "general": "일반 문의 답변드립니다. 무엇을 도와드릴까요?"
    }
    
    print(f"✅ 답변 생성 완료")
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

# SQLite 체크포인터 설정
db_path = "db/checkpoints.db"
with SqliteSaver.from_conn_string(db_path) as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
    
    # 테스트
    if __name__ == "__main__":
        print("=== 첫 번째 실행 (저장) ===")
        config = {"configurable": {"thread_id": "user_123"}}
        
        result = app.invoke({
            "query": "로그인이 안돼요",
            "category": "",
            "response": "",
            "confidence": 0.0,
            "retry_count": 0
        }, config=config)
        
        print(f"\n문의: {result['query']}")
        print(f"분류: {result['category']}")
        print(f"답변: {result['response']}")
        print(f"\n💾 DB에 저장됨: {db_path}")
        
        # 저장된 상태 확인
        snapshot = app.get_state(config)
        print(f"\n=== 저장된 상태 확인 ===")
        print(f"thread_id: {config['configurable']['thread_id']}")
        print(f"마지막 상태: {snapshot.values}")

print("\n🔄 이제 프로그램을 종료하고 다시 실행해보세요!")
print("같은 thread_id로 조회하면 저장된 상태가 남아있습니다.")