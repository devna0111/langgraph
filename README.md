# Langraph를 활용한 AI Agent를 설계하기
## 단계별 학습
- vllm + 양자화 => 서버 구축
- 필요 환경 구축
- 랭그래프 학습
- AI Agent 구현

## vLLM 개념
- vLLM(Virtual Large Language Model)은 대규모 언어 모델의 추론을 효율적으로 처리하기 위한 고성능 서빙 엔진
- 핵심 특징
   1. PagedAttention : 메모리를 효율적으로 관리하여 더 많은 동시 요청 처리
   2. 연속 배치 처리 : 여러 요청을 동시에 처리하여 처리량 향상
   3. 최적화된 CUDA 커널 : GPU 활용도 극대화
   4. 호환성 : HuggingFace Transformers 완벽 호환
- 기존 방식 vs vLLM
   1. 기존 방식 : [요청] -> [처리] -> [응답] -> [다음 요청] -> [처리] -> [응답] ...
   2. vLLM : [요청1,2,3...] -> [병렬처리] -> [응답1,2,3,...][응답] ...

## 양자화 개념(Quantization)
- 양자화는 모델의 가중치를 더 적은 비트로 표현하여 메모리 사용량과 추론 속도를 개선하는 기술
- 양자화의 종류
   1. FP(32) : default, 32비트 부동소수점
   2. FP16 : 16비트 부동소수점(메모리 50% 절약)
   3. INT8 : 8비트 정수 (메모리 75% 절약)
   4. INT4 : 4비트 정수 (메모리 87.5% 절약)
- 양자화 방법
   1. Post-training Quantization : 학습 후 양자화(PTQ), 이미 완성된 모델을 압축
   2. Quantization-aware Training : 학습 중 양자화 고려(QAT), 처음부터 양자화를 염두에 두고 학습하는 방법
- 현재 작업 환경 : RTX4060 VRAM 8gb, 사용 예정 llm은 Qwen2.5:7b 모델로 원활한 서비스를 위해 PTQ로 INT8 양자화(예상 필요 VRAM ~7gb)

## 랭그래프의 기본 개념와 구조
### 랭그래프의 기본 개념
- LangGraph는 자연어 처리(NLP) 기반의 언어 **그래프** 서비스 플랫폼
```
   StateGraph: 상태를 가진 그래프. Agent의 메모리 역할
   Node: 작업을 수행하는 함수. 각 단계별 처리 담당
   Edge: Node 간 연결. 작업 흐름 정의
   State: Agent가 기억하는 정보. 대화 내용, 중간 결과 등
   END: 그래프 종료 지점
```
### 랭그래프의 구조
   1. State 정의 (무엇을 기억할지)
   2. Node 함수 작성 (각 단계별 처리)
   3. Graph 구성 (흐름 설계)
   4. 컴파일 및 실행
   5. 그래프 빌드 순서
   ```
   workflow = StateGraph(CustomerState)  # 1. 그래프 생성
   workflow.add_node()                   # 2. 노드 추가
   workflow.set_entry_point()            # 3. 시작점 설정
   workflow.add_conditional_edges()      # 4. 조건부 엣지
   workflow.add_edge()                   # 5. 일반 엣지
   app = workflow.compile()              # 6. 컴파일
   ```

### **랭그래프의 조건부 엣지(Conditional Edge)**
- 조건부 엣지는 입력에 따라 다른 노드로 분기하는 기능
- add_conditional_edges() 메소드를 사용하여 조건부 엣지를 추가
- 라우터함수가 어떤 노드로 갈 지 결정
```
1. 일반 엣지 vs 조건부 엣지:
   • 일반 엣지: A → B (항상 B로 이동)
   • 조건부 엣지: A → B or C or D (조건에 따라 분기)

2. 핵심 구성 요소:
   • 라우터 함수: 어디로 갈지 결정하는 함수
   • 매핑 딕셔너리: 라우터 결과 → 실제 노드
   • 시작 노드: 조건부 분기가 시작되는 노드

3. 문법 구조:
   workflow.add_conditional_edges(
       source="시작_노드",
       path=라우터_함수, # 판별함수를 사용함. intent classification
       path_map={
           "조건1": "노드1",
           "조건2": "노드2",
           "조건3": "노드3"
       }
   )

```

### 랭그래프의 시각화
- 랭그래프 시각화는 크게 세 가지 방법이 있음
   1. **Mermaid 다이어그램** : 특정 코드를 반환하며 이를 https://mermaid.live에 붙여넣기 하면 시각화 가능
      ```
      print(app.get_graph().draw_mermaid())
      ```
   2. **PNG 이미지 저장** : 파일로 저장하나 WSL의 경우 graphbiz가 필요
      ```
      [저장 방식]
      png_data = app.get_graph().draw_mermaid_png()
      with open("workflow_graph.png", "wb") as f:
         f.write(png_data)

      [Graphviz 설치]
      Linux cmd
         sudo apt-get update
         sudo apt-get install graphviz
      python cmd
         pip install pygraphviz
      ```
   3. ASCII 아트 : 개발자가 확인하기 용이한, 터미널에서 즉시 확인이 가능한 방식
      ```
      print(app.get_graph().draw_ascii())
      ```

### 대화 history 저장을 통한 메모리 agent
- **langchain**과 동일하게 **langchain.memory.ConversationBufferMemory**를 사용하여 대화 자동 저장
```
# 사용자 메시지 저장
memory.chat_memory.add_user_message(user_message)

# AI 응답 저장  
memory.chat_memory.add_ai_message(response)

# 저장된 대화 불러오기
memory_content = memory.load_memory_variables({})
```
- **핵심 장점**
```
   1. 자동 메모리 관리: 수동 구현 없이 LangChain이 처리
   2. 표준화: LangChain 표준 인터페이스 활용
   3. 확장성: 다른 LangChain 컴포넌트와 쉽게 연동
   4. 안정성: 검증된 라이브러리 활용
```

### Tool-Using Agent / Function Calling Agent
- LLM에게 tool을 부여하여 다양한 상호작용으로 유연한 반응을 이끌어 낼 수 있음 : 기존 Function Calling이나 langchain.agents.Tool 객체
- 그러나 랭그래프의 node를 통해서도 tool 기능을 부여할 수 있음 => State에 담아서 전달 하는 등
- 두 방법의 각각의 장단점
```
- Langgraph Node 방식
- 예시 : 연구 Agent(LLM+검색) → 분석 Agent(LLM+계산) → 작가 Agent(LLM+글쓰기)
   - 장점: 각 Agent가 창의적 판단
   - 단점: 리소스 소모 높음 (LLM 호출 9-12회)

- Langchain.agents.Tool 방식
- 예시 : 검색 함수 → LLM 분석 → 계산 함수 → LLM 작성
   - 장점: 리소스 소모 적음 (LLM 호출 2~3회 정도) 
   - 단점: 유연성 부족
```
- 이런 두 방식의 장점을 극대화하는 **하이브리드 방식**을 선호
```
- 예시 : 연구 Agent(LLM+검색) → 계산 함수 → 작가 Agent(LLM+글쓰기)
- 장점: 비용 효율 + 적절한 창의성
- 핵심: 필요한 곳에만 LLM 사용
```
- **실무 결정 기준**

- LLM Agent를 쓸 때:
```
   - 복잡한 판단이 필요할 때
   - 창의적 작업일 때
   - 사용자 의도 파악이 중요할 때
```

- 함수 노드를 쓸 때:
```
   - 정해진 로직으로 처리 가능할 때
   - 속도가 중요할 때
   - 비용을 절약하고 싶을 때
```
### 체크포인트 & 상태 저장
**[체크포인트의 의의]**
- 체크포인트는 서버 재시작 시 대화 이어가기
- 긴 작업 중단 후 나중에 재개
- 사용자별 세션관리 

#### [사용 방식]

**[저장]**
```
workflow = StateGraph(State:TypedDict)
....
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

**[저장 상태 조회]**
```
snapshot = app.get_state(config)

print("=== 저장된 데이터 ===")
print(f"현재 상태: {snapshot.values}")
print(f"다음 노드: {snapshot.next}")
print(f"체크포인트 ID: {snapshot.config['configurable']['checkpoint_id']}")
```

**[전체 히스토리 조회]**
```
print("\n=== 전체 실행 히스토리 ===")
for i, state in enumerate(app.get_state_history(config)):
    print(f"\n[{i}] 체크포인트: {state.config['configurable']['checkpoint_id'][:8]}")
    print(f"    상태: category={state.values.get('category')}, response={state.values.get('response')[:30] if state.values.get('response') else 'None'}...")
```
</br>
아래 활용 예시

- [롤백] 2단계 전으로 돌아가기
   ```
   app.update_state(config, values=None, as_node="classifier")
   ```
- [디버깅] 어느 단계에서 문제가 생겼는지 추적
```
for state in app.get_state_history(config):
   print(state.values)
```
- [작업 재개] 서버 재시작 후에도 같은 지점부터 이어서
```
   snapshot = app.get_state(config)
   app.invoke(None, config=config)  # 이어서 실행
```



### 멀티 에이전트
- LLM -> LLM 연결로 보다 유연하고 강력한 에이전트 군집을 생성할 수 있음.
- 생성 방법은 단순하게 Node에서 State를 체크하고 함수 기능을 작동하여 LLM의 추론이 필요한 Node에 invoke하여 결과를 유도하는 방식
- 워크플로우 예시
```
    사용자 요청
         ↓
    🤖 연구자 Agent (LLM)
    ├─ 검색 필요성 판단
    ├─ 검색 실행
    └─ 데이터 수집
         ↓
    ⚙️ 분석 노드 (함수)
    ├─ 텍스트 분석
    ├─ 키워드 추출
    └─ 통계 계산
         ↓
    ✍️ 작가 Agent (LLM)
    ├─ 종합 분석
    ├─ 콘텐츠 작성
    └─ 구조화
         ↓
    📋 정리 노드 (함수)
    ├─ 포맷팅
    ├─ 메타데이터 추가
    └─ 최종 출력
         ↓
    ✅ 완료된 결과
```
- 다만, LLM 추론 시간이 과정마다 발생하고 활용하는 리소스의 양이 많아지는 등 시간적, 경제적, 물리적 한계가 발생 가능

### 02_멀티에이전트협업.02_실습_고객문의처리시스템.py 아키택쳐 및 실행 흐름 예시
```
[아키택쳐]
   [최초]

      [START]
         ↓
      [분류 에이전트] → category 결정
         ↓
      [조건부 라우팅]
         ├─ technical → [기술지원 에이전트]
         ├─ billing   → [결제 에이전트]
         └─ general   → [일반 에이전트]
         ↓
      [END]
   
   [결과물에 따라 retry 가능하도록 수정]
      retry → route_query → [tech/billing/general] → check_confidence
            ↑_______________retry________________|

[실행흐름 예시]
   [입력]
   state = {
      "query": "결제가 안돼요",
      "category": "",
      "response": "",
      "confidence": 0.0
   }

   [실행 순서]
   1. classifier → category = "billing"
   2. route_query → "billing" 반환
   3. billing_agent → response = "결제 문제 해결 방법..."
   4. LLM에 의한 Confidence 결정
   5. Confidence에 따른 조건부 라우팅(0.7 기준 성공/실패, 2차례 실패시 상담원 연결의 구조 유도)
      [성공 케이스]
      classifier → tech_support (0.85) → END

      [재시도 케이스]
      classifier → tech_support (0.55) → retry → tech_support (0.78) → END

      [에스컬레이션 케이스]
      classifier → tech_support (0.50) → retry → tech_support (0.52) → retry → tech_support (0.48) → escalate → END
   6. END
```

### 학습 할 것
- Option A: 휴먼-인-더-루프 (Human-in-the-Loop)
   - 사람의 승인을 받아야 진행되는 시스템
   - 활용: 중요한 결정, 민감한 답변

- Option B: 체크포인트 & 상태 저장 
   - 대화를 중단하고 나중에 이어서 하기
   - 활용: 긴 작업, 세션 관리

- Option C: 스트리밍 & 실시간 출력
   - LLM 답변을 실시간으로 출력
   - 활용: 사용자 경험 개선

- Option D: 그래프 시각화
   - 현재 워크플로우를 그림으로 보기
   - 활용: 디버깅, 문서화

- Option E: 복잡한 멀티 에이전트 (계층 구조)
   - 매니저 에이전트가 여러 워커 에이전트를 관리
   - 활용: 복잡한 작업 분할