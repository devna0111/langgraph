from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict

class CustomerState(TypedDict):
    query: str
    category: str
    response: str

def classify_agent(state: CustomerState) -> CustomerState:
    return {**state, "category": "technical"}

def answer_agent(state: CustomerState) -> CustomerState:
    return {**state, "response": "기술 지원 답변"}

workflow = StateGraph(CustomerState)
workflow.add_node("classifier", classify_agent)
workflow.add_node("answer", answer_agent)
workflow.set_entry_point("classifier")
workflow.add_edge("classifier", "answer")
workflow.add_edge("answer", END)

# 불러오기
with SqliteSaver.from_conn_string("db/checkpoints.db") as saver:
    app = workflow.compile(checkpointer=saver)
    
    # 저장된 상태 조회
    snapshot = app.get_state({"configurable": {"thread_id": "user_123"}})
    
    print("=== 저장된 상태 불러오기 ===")
    print(f"문의: {snapshot.values['query']}")
    print(f"분류: {snapshot.values['category']}")
    print(f"답변: {snapshot.values['response']}")