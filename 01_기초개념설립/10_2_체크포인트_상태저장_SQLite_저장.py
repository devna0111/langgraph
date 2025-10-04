from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Literal

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

# 저장
with SqliteSaver.from_conn_string("db/checkpoints.db") as saver:
    app = workflow.compile(checkpointer=saver)
    
    result = app.invoke({
        "query": "로그인 안돼요",
        "category": "",
        "response": ""
    }, config={"configurable": {"thread_id": "user_123"}})
    
    print(f"✅ 저장 완료: {result}")