from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
import operator

def langgraph_img(app : StateGraph) :
    img = input("파일이름을 입력하세요 : ")
    file_path = f"img/{img}.png"
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"그래프 이미지 저장됨: {file_path}")
    except Exception as e:
        print(f"PNG 저장 실패: {e}")
        print("Graphviz가 설치되지 않았을 수 있습니다.")
        
# 상태 정의
class ParallelState(TypedDict):
    messages: Annotated[list, operator.add]
    topic: str
    web_result: str
    db_result: str
    api_result: str
    final_report: str

llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0.12
)

# Agent 노드 (시작점)
def agent_node(state: ParallelState) -> ParallelState:
    """주제를 분석하고 병렬 조사 시작"""
    messages = state["messages"]
    topic = messages[0].content
    
    print(f"\n[Agent] 주제 '{topic}' 분석 완료")
    print("[Agent] 병렬 조사 시작: Web, DB, API")
    
    return {"topic": topic}

# 3개의 Researcher (병렬 실행) => 실제론 tools로 달아줘야함. 현재는 학습용으로 깡통 LLM
def researcher_web(state: ParallelState) -> ParallelState:
    print("\n[Web Researcher] 웹 조사 시작...")
    prompt = f"웹에서 '{state['topic']}'에 대한 정보를 조사하세요."
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[Web] 완료: {result.content[:50]}...")
    return {"web_result": result.content}

def researcher_db(state: ParallelState) -> ParallelState:
    print("\n[DB Researcher] 데이터베이스 조사 시작...")
    prompt = f"데이터베이스에서 '{state['topic']}'에 대한 데이터를 조사하세요."
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[DB] 완료: {result.content[:50]}...")
    return {"db_result": result.content}

def researcher_api(state: ParallelState) -> ParallelState:
    print("\n[API Researcher] API 조사 시작...")
    prompt = f"API를 통해 '{state['topic']}'에 대한 최신 정보를 조사하세요."
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[API] 완료: {result.content[:50]}...")
    return {"api_result": result.content}

# 통합 노드
def aggregator_node(state: ParallelState) -> ParallelState:
    """3개 Researcher 결과를 통합"""
    print("\n[Aggregator] 결과 통합 중...")
    
    prompt = f"""다음 3가지 조사 결과를 통합하여 종합 보고서를 작성하세요.

                주제: {state['topic']}

                웹 조사 결과:
                {state['web_result']}

                데이터베이스 조사 결과:
                {state['db_result']}

                API 조사 결과:
                {state['api_result']}

                위 내용을 간단히 통합하여 보고서를 작성하세요."""

    result = llm.invoke([HumanMessage(content=prompt)])
    
    print("\n[Aggregator] 통합 완료!")
    
    return {
        "final_report": result.content,
        "messages": [AIMessage(content=result.content)]
    }

# 그래프 생성
workflow = StateGraph(ParallelState)

# 노드 추가
workflow.add_node("agent", agent_node)
workflow.add_node("web", researcher_web)
workflow.add_node("db", researcher_db)
workflow.add_node("api", researcher_api)
workflow.add_node("aggregate", aggregator_node)

# 엣지 설정
workflow.set_entry_point("agent")

# agent → 3개 Researcher로 병렬 분기
workflow.add_edge("agent", "web")
workflow.add_edge("agent", "db")
workflow.add_edge("agent", "api")

# 3개 Researcher → aggregate로 수렴
workflow.add_edge("web", "aggregate")
workflow.add_edge("db", "aggregate")
workflow.add_edge("api", "aggregate")

workflow.add_edge("aggregate", END)

app = workflow.compile()

langgraph_img(app)


# 테스트
if __name__ == "__main__":
    import time
    
    print("=== 병렬 멀티 에이전트 시스템 ===")
    
    topic = input("\n조사할 주제: ")
    
    print(f"\n{'='*60}")
    start = time.time()
    
    result = app.invoke({
        "messages": [HumanMessage(content=topic)],
        "topic": "",
        "web_result": "",
        "db_result": "",
        "api_result": "",
        "final_report": ""
    })
    
    end = time.time()
    
    print(f"\n{'='*60}")
    print("최종 통합 보고서")
    print('='*60)
    print(result["final_report"])
    
    print(f"\n{'='*60}")
    print(f"총 실행 시간: {end - start:.2f}초")