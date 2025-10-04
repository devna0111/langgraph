'''
Manager 에이전트가 여러 Worker 에이전트를 관리하는 구조

[계층구조]
Manager (관리자)
  ├─ Researcher (조사)
  ├─ Analyzer (분석)
  └─ Writer (작성)

[순환 흐름]
Manager → Researcher → Manager → Analyzer → Manager → Writer → END
'''
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator # 수학/논리 연산자를 함수 형태로 제공하는 라이브러리

# 상태 정의
class ResearchState(TypedDict):
    messages: Annotated[list, operator.add] # messages는 리스트이며 각 노드에서 반환되는 message를 add연산 하겠다는 의미
    # 만약 operator.add를 사용하지 않으면 messages를 덮어쓰기함
    topic: str
    research_data: str
    analysis_result: str
    final_report: str
    current_worker: str

# LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Manager 노드
def manager_node(state: ResearchState) -> ResearchState:
    """Manager가 작업을 분석하고 다음 Worker 결정"""
    
    topic = state.get("topic", "")
    research_data = state.get("research_data", "")
    analysis_result = state.get("analysis_result", "")
    
    if not research_data:
        next_worker = "researcher"
        print(f"\n[Manager] 주제 '{topic}' 분석 완료 → Researcher에게 할당")
    elif not analysis_result:
        next_worker = "analyzer"
        print(f"\n[Manager] 조사 완료 → Analyzer에게 할당")
    else:
        next_worker = "writer"
        print(f"\n[Manager] 분석 완료 → Writer에게 할당")
    
    return {"current_worker": next_worker}

# Worker 1: Researcher
def researcher_node(state: ResearchState) -> ResearchState:
    """정보 수집 담당"""
    
    topic = state["topic"]
    print(f"\n[Researcher] '{topic}' 조사 시작...")
    
    prompt = f"주제 '{topic}'에 대한 핵심 정보 3가지를 간단히 조사하세요."
    response = llm.invoke([HumanMessage(content=prompt)])
    
    research_data = response.content
    print(f"조사 결과: {research_data[:80]}...")
    
    return {"research_data": research_data}

# Worker 2: Analyzer
def analyzer_node(state: ResearchState) -> ResearchState:
    """데이터 분석 담당"""
    
    research_data = state["research_data"]
    print(f"\n[Analyzer] 데이터 분석 시작...")
    
    prompt = f"다음 조사 결과를 분석하고 주요 인사이트를 도출하세요:\n\n{research_data}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    analysis_result = response.content
    print(f"분석 결과: {analysis_result[:80]}...")
    
    return {"analysis_result": analysis_result}

# Worker 3: Writer
def writer_node(state: ResearchState) -> ResearchState:
    """보고서 작성 담당"""
    
    topic = state["topic"]
    research_data = state["research_data"]
    analysis_result = state["analysis_result"]
    
    print(f"\n[Writer] 최종 보고서 작성 중...")
    
    prompt = f"""주제: {topic}

조사 결과:
{research_data}

분석 결과:
{analysis_result}

위 내용을 바탕으로 간단한 보고서를 작성하세요."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    final_report = response.content
    print(f"\n보고서 작성 완료!")
    
    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)]
    }

# 라우팅 함수
def route_to_worker(state: ResearchState) -> Literal["researcher", "analyzer", "writer", "end"]:
    """Manager가 결정한 Worker로 라우팅"""
    
    current_worker = state.get("current_worker", "")
    
    if current_worker == "researcher":
        return "researcher"
    elif current_worker == "analyzer":
        return "analyzer"
    elif current_worker == "writer":
        return "writer"
    else:
        return "end"

# 그래프 생성
workflow = StateGraph(ResearchState)

# 노드 추가
workflow.add_node("manager", manager_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("writer", writer_node)

# 엣지 설정
workflow.set_entry_point("manager")

# Manager가 Worker 결정
workflow.add_conditional_edges(
    "manager",
    route_to_worker,
    {
        "researcher": "researcher",
        "analyzer": "analyzer",
        "writer": "writer",
        "end": END
    }
)

# 각 Worker 완료 후 다시 Manager로
workflow.add_edge("researcher", "manager")
workflow.add_edge("analyzer", "manager")
workflow.add_edge("writer", END)

app = workflow.compile()

img = input("파일이름을 입력하세요 : ")
file_path = f"img/{img}"

try:
    png_data = app.get_graph().draw_mermaid_png()
    with open(file_path, "wb") as f:
        f.write(png_data)
    print(f"그래프 이미지 저장됨: {file_path}")
except Exception as e:
    print(f"PNG 저장 실패: {e}")
    print("Graphviz가 설치되지 않았을 수 있습니다.")
    
# 테스트
if __name__ == "__main__":
    print("=== 계층형 멀티 에이전트 시스템 ===")
    
    topic = input("\n조사할 주제를 입력하세요: ")
    
    print(f"\n주제: {topic}")
    print("="*60)
    
    result = app.invoke({
        "messages": [],
        "topic": topic,
        "research_data": "",
        "analysis_result": "",
        "final_report": "",
        "current_worker": ""
    })
    
    print("\n" + "="*60)
    print("최종 보고서")
    print("="*60)
    print(result["final_report"])