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

## 학습 과정 요약

**주 1-2: 초급 기능**
-기초 다지기
Ollama + LangGraph로 간단한 Agent 구현
Tool 사용법 습득 (계산기, 웹검색 등)

**주 3-4: 중급 기능**
- 멀티 Agent 협업
- RAG 시스템 구축
- 성능 측정 시작

**주 5-6: vLLM 도입**
- HuggingFace 원본 모델로 vLLM 서버 구축
- API 서버 최적화
- 로드 테스트

**주 7-8: 프로덕션 배포**
- Docker 컨테이너화
- 모니터링 시스템
- 실제 서비스 배포
