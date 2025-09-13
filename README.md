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

## Modelfile(01 ~ 03_*.py) 내용 요약
- 기존 원본 모델(FP16/FP32) => Modelfile을 통해 설정만 변경한 것으로 양자화 모델은 아님
- 양자화는 transformers + BitsAndBytes => GGUF => Ollama 방식으로 모델을 서브해야함

## 학습 과정 요약

**주 1-2: 초급 기능**
 |기초 다지기|
Ollama + LangGraph로 간단한 Agent 구현
Tool 사용법 습득 (계산기, 웹검색 등)

**주 3-4: 중급 기능**
|멀티 Agent 협업|
|RAG 시스템 구축|
|성능 측정 시작|

**주 5-6: vLLM 도입**
|HuggingFace 원본 모델로 vLLM 서버 구축|
|API 서버 최적화|
|로드 테스트|

**주 7-8: 프로덕션 배포**
|Docker 컨테이너화|
|모니터링 시스템|
|실제 서비스 배포|
