'''
휴먼 인 더 루프 방식은 중요 의사 결정 사항에서 인간의 판단이 개입되는 구조
체크포인트의 핵심 활용 사례
[실행흐름]
1. draft → AI 초안 작성
2. wait_approval 전에 INTERRUPT ⏸️
3. [사람 개입] 승인 or 수정 요청
4. update_state로 결정 반영
5. invoke(None) → 재개
6. final → 최종 응답
7. END
'''

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage
import operator

# 상태 정의
class ApprovalState(TypedDict):
    messages: Annotated[list, operator.add]
    draft_response: str  # AI가 작성한 초안
    approved: bool  # 승인 여부
    user_feedback: str  # 사용자 피드백

# LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# 초안 작성 노드
def draft_response_node(state: ApprovalState) -> ApprovalState:
    """AI가 답변 초안을 작성"""
    messages = state["messages"]
    
    system_prompt = """당신은 고객 응대 AI입니다.
                        고객 문의에 대한 답변 초안을 작성하세요.
                        이 답변은 사람의 검토를 거칠 예정입니다."""
    
    full_messages = [HumanMessage(content=system_prompt)] + messages
    response = llm.invoke(full_messages)
    
    print(f"\n📝 AI 초안 작성 완료:")
    print(f"{response.content}\n")
    
    return {
        "draft_response": response.content,
        "approved": False
    }

# 승인 대기 노드
def wait_for_approval_node(state: ApprovalState) -> ApprovalState:
    """사람의 승인을 대기 (실제로는 interrupt 발생)"""
    print("⏸사람의 승인 대기 중...")
    return state

# 최종 응답 노드
def final_response_node(state: ApprovalState) -> ApprovalState:
    """승인된 답변 또는 수정된 답변 반영"""
    
    if state.get("user_feedback"):
        # 피드백이 있으면 수정
        print(f"\n🔄 사용자 피드백 반영: {state['user_feedback']}")
        
        revision_prompt = f"""원래 답변: {state['draft_response']}

                                사용자 피드백: {state['user_feedback']}

                                위 피드백을 반영하여 답변을 수정하세요."""
        
        response = llm.invoke([HumanMessage(content=revision_prompt)])
        final = response.content
    else:
        # 피드백 없으면 초안 그대로
        final = state["draft_response"]
    
    print(f"\n✅ 최종 답변:")
    print(f"{final}\n")
    
    return {
        "messages": [AIMessage(content=final)],
        "approved": True
    }

# 라우팅 함수
def check_approval(state: ApprovalState) -> Literal["approved", "wait"]:
    """승인 여부 확인"""
    if state.get("approved", False):
        return "approved"
    return "wait"

# 그래프 생성
workflow = StateGraph(ApprovalState)

workflow.add_node("draft", draft_response_node)
workflow.add_node("wait_approval", wait_for_approval_node)
workflow.add_node("final", final_response_node)

workflow.set_entry_point("draft")

# draft → wait_approval
workflow.add_edge("draft", "wait_approval")

# wait_approval → interrupt (사람 개입)
# 이 부분은 아래에서 설명

workflow.add_conditional_edges(
    "wait_approval",
    check_approval,
    {
        "approved": "final",
        "wait": END  # 임시로 END (실제로는 interrupt)
    }
)

workflow.add_edge("final", END)

# 메모리 체크포인터
memory = MemorySaver()

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["wait_approval"]  # 이 노드 전에 중단
)

# 테스트
if __name__ == "__main__":
    print("=== 휴먼-인-더-루프 테스트 ===\n")
    
    # 1단계: 초안 작성
    config = {"configurable": {"thread_id": "approval_test_1"}}
    
    initial_state = {
        "messages": [HumanMessage(content="환불 정책이 어떻게 되나요?")],
        "draft_response": "",
        "approved": False,
        "user_feedback": ""
    }
    
    print("[1단계] AI 초안 작성 시작...")
    result = app.invoke(initial_state, config=config)
    
    # 현재 상태 확인
    current_state = app.get_state(config)
    print(f"현재 노드: {current_state.next}")
    print(f"초안: {current_state.values['draft_response'][:100]}...\n")
    
    # 2단계: 사용자 승인 시뮬레이션
    print("\n" + "="*60)
    print("[사용자 선택]")
    print("1. 승인")
    print("2. 수정 요청")
    choice = input("선택 (1 or 2): ")
    
    if choice == "1":
        # 승인
        print("\n[2단계] 승인됨")
        app.update_state(config, {"approved": True})
        
    else:
        # 수정 요청
        feedback = input("\n수정 요청 내용: ")
        print(f"\n[2단계] 수정 요청: {feedback}")
        app.update_state(config, {
            "approved": True,
            "user_feedback": feedback
        })
    
    # 3단계: 최종 응답 생성
    print("\n[3단계] 최종 응답 생성...")
    final_result = app.invoke(None, config=config)
    
    print("\n=== 완료 ===")