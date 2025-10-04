from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
import operator

# 상태 정의
class StreamState(TypedDict):
    messages: Annotated[list, operator.add]
    response: str

# LLM
llm = ChatOllama(
    model="devna0111-7b-q4",
    base_url="http://localhost:11434",
    temperature=0.7
)

# 에이전트 노드
def agent_node(state: StreamState) -> StreamState:
    messages = state["messages"]
    
    # 스트리밍 호출
    full_response = ""
    print("\n답변: ", end="", flush=True)
    # 일반적으로 response = llm.invoke(messages)
    for chunk in llm.stream(messages):
        content = chunk.content
        print(content, end="", flush=True) # flush = True : 버퍼 즉시 출력
        full_response += content
    
    print("\n")
    
    return {
        "messages": [AIMessage(content=full_response)],
        "response": full_response
    }

# 그래프
workflow = StateGraph(StreamState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

app = workflow.compile()

# 테스트
if __name__ == "__main__":
    print("=== 스트리밍 테스트 ===")
    
    queries = [
        "랭그래프가 뭔가요?",
        "AI 에이전트의 장점을 3가지 설명해주세요"
    ]
    
    for query in queries:
        print(f"\n질문: {query}")
        
        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "response": ""
        })