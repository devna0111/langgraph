from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from typing import TypedDict, List
import re
def create_hybrid_multi_agent():
    """하이브리드 멀티 에이전트 시스템 생성"""
    print("==== 하이브리드 멀티 에이전트 시스템 생성 시작 ====")
    
    try:
        # 1. 공유 상태 정의
        class MultiAgentState(TypedDict):
            user_request: str # 사용자 요청
            research_data: str # 연구자가 수집한 데이터
            analysis_result: str # 분석 결과 (함수 노드)
            draft_content: str # 작가가 작성한 초안
            final_output: str # 최종 결과
            current_step: str # 현재 진행 단계
            metadata: dict # 추가 메타데이터
        
        # 2. LLM 설정
        llm = Ollama(
            model="devna0111-7b-q4",
            base_url="http://localhost:11434",
            temperature=0.15,
        )
        
        # 3. 연구자 Agent (LLM + 실제 검색 도구)
        def researcher_agent(state: MultiAgentState):
            """연구자 Agent - 실제 duckduckgo-search 활용"""
            print("==== 연구자 Agent 작업 시작")
            
            user_request = state['user_request']
            
            # 검색이 필요한지 LLM이 판단
            search_prompt = f"""
                                사용자 요청: {user_request}

                                이 요청에 대해 인터넷 검색이 필요한지 판단하고, 필요하다면 적절한 검색어를 제안해주세요.

                                응답 형식 :
                                - 검색 필요: 예/아니오
                                - 검색어: (필요한 경우만)
                                - 이유: (간단한 설명)
                                """
            
            search_decision = llm.invoke(search_prompt)
            print(f"==== 검색 필요성 판단: {search_decision[:100]}...")
            
            # 실제 검색 실행
            if "검색 필요: 예" in search_decision or "필요: 예" in search_decision:
                try:
                    from ddgs import DDGS # duckduckgo_search
                    
                    # 검색어 추출 (간단한 패턴 매칭)
                    search_keywords = user_request.replace("에 대해", "").replace("알려줘", "").replace("분석해줘", "").strip()
                    print(f"==== 검색어: {search_keywords}")
                    
                    # 실제 DuckDuckGo 검색
                    with DDGS() as ddgs:
                        search_results = list(ddgs.text(search_keywords, max_results=5))
                    
                    if search_results:
                        # 검색 결과 포맷팅
                        formatted_results = []
                        for i, result in enumerate(search_results, 1):
                            title = result.get('title', '제목 없음')
                            body = result.get('body', '내용 없음')
                            url = result.get('href', '')
                            
                            formatted_results.append(f"""
                                                    {i}. {title}
                                                    내용: {body[:200]}...
                                                    출처: {url}
                                                    """)
                        
                        research_data = f"""
                                        ==== '{search_keywords}' 검색 결과:

                                        {''.join(formatted_results)}

                                        총 {len(search_results)}개의 검색 결과를 수집했습니다.
                                        """
                        print(f"==== 검색 완료: {len(search_results)}개 결과")
                    else:
                        research_data = f"'{search_keywords}'에 대한 검색 결과를 찾을 수 없습니다."
                        print("==== 검색 결과 없음")
                
                except ImportError:
                    print("**** duckduckgo-search 라이브러리가 설치되지 않음")
                    research_data = f"""
                                        *** duckduckgo-search 라이브러리가 설치되지 않았습니다.
                                        설치 명령어: pip install duckduckgo-search

                                        '{user_request}'에 대한 기본 정보를 제공합니다:
                                        - 해당 주제는 현재 관심이 높은 분야입니다.
                                        - 더 자세한 정보를 위해서는 실제 검색이 필요합니다.
                                        """
                
                except Exception as e:
                    print(f"**** 검색 오류: {e}")
                    research_data = f"검색 중 오류가 발생했습니다: {str(e)}"
            
            else:
                research_data = f"'{user_request}'에 대한 검색이 불필요하여 기본 지식을 활용합니다."
                search_keywords = None
            
            return {
                "user_request": state['user_request'],
                "research_data": research_data,
                "analysis_result": state.get('analysis_result', ''),
                "draft_content": state.get('draft_content', ''),
                "final_output": state.get('final_output', ''),
                "current_step": "research_complete",
                "metadata": {
                    "search_performed": "검색 필요: 예" in search_decision,
                    "search_keywords": search_keywords,
                    "search_results_count": len(search_results) if "검색 필요: 예" in search_decision and 'search_results' in locals() else 0
                }
            }
        
        # 4. 분석 함수 노드 (순수 함수)
        def analysis_function_node(state: MultiAgentState):
            """분석 함수 노드 - 데이터 처리 및 통계"""
            print("**** 분석 함수 노드 작업 시작")
            
            research_data = state['research_data']
            
            # 간단한 텍스트 분석
            analysis_metrics = {
                "data_length": len(research_data),
                "sentence_count": research_data.count('.') + research_data.count('!') + research_data.count('?'),
                "keyword_frequency": {},
                "sentiment_score": 0.7  # 가짜 감정 점수로 실제 프로젝트 시 감정 분류 모델 등을 활용해서 사용해야함
            }
            
            # 키워드 빈도 분석 (간단 버전)
            words = re.findall(r'\w+', research_data.lower())
            common_words = ['은', '는', '이', '가', '을', '를', '에', '의', '와', '과']
            filtered_words = [word for word in words if word not in common_words and len(word) > 1]
            
            # 상위 3개 키워드 추출
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            analysis_metrics["keyword_frequency"] = dict(top_keywords)
            
            # 분석 결과 요약
            analysis_result = f"""
                                **** 데이터 분석 결과:
                                - 데이터 길이: {analysis_metrics['data_length']} 문자
                                - 문장 수: {analysis_metrics['sentence_count']}개
                                - 주요 키워드: {', '.join([k for k, v in top_keywords])}
                                - 감정 점수: {analysis_metrics['sentiment_score']} (긍정적)
                                - 분석 완료 시간: 현재
                                """
            
            return {
                "user_request": state['user_request'],
                "research_data": state['research_data'],
                "analysis_result": analysis_result,
                "draft_content": state.get('draft_content', ''),
                "final_output": state.get('final_output', ''),
                "current_step": "analysis_complete",
                "metadata": {**state.get('metadata', {}), "analysis_metrics": analysis_metrics}
            }
        
        # 5. 작가 Agent (LLM + 텍스트 생성)
        def writer_agent(state: MultiAgentState):
            """작가 Agent - 콘텐츠 작성 전문"""
            print("==== 작가 Agent 작업 시작")
            
            user_request = state['user_request']
            research_data = state['research_data']
            analysis_result = state['analysis_result']
            
            # 작가가 종합적인 콘텐츠 작성
            writing_prompt = f"""
                                사용자 요청: {user_request}

                                연구 데이터:
                                {research_data}

                                분석 결과:
                                {analysis_result}

                                위 정보를 바탕으로 사용자의 요청에 대한 완성도 높은 답변을 작성해주세요.
                                구조화되고 읽기 쉬우며, 핵심 정보를 포함해야 합니다.
                                논문 형태로 작성해야하며 한글로만 작성합니다.
                                """
            
            draft_content = llm.invoke(writing_prompt)
            
            return {
                "user_request": state['user_request'],
                "research_data": state['research_data'],
                "analysis_result": state['analysis_result'],
                "draft_content": draft_content,
                "final_output": state.get('final_output', ''),
                "current_step": "writing_complete",
                "metadata": state.get('metadata', {})
            }
        
        # 6. 최종 정리 노드 (함수)
        def finalize_output_node(state: MultiAgentState):
            """최종 정리 노드 - 결과 포맷팅"""
            print("==== 최종 정리 노드 작업 시작")
            
            draft_content = state['draft_content']
            metadata = state.get('metadata', {})
            
            # 최종 출력 포맷팅
            final_output = f"""
                            -- 하이브리드 멀티 에이전트 협업 결과 --

                            {draft_content}

                            =========================================
                            **** 작업 정보:
                            • 검색 수행: {'예' if metadata.get('search_performed') else '아니오'}
                            • 검색 키워드: {metadata.get('search_keywords', 'N/A')}
                            • 검색 결과 수: {metadata.get('search_results_count', 0)}개
                            • 분석 완료: 예
                            • 작가 작업: 완료

                            **** 시스템 정보:
                            • 연구자 Agent: LLM + duckduckgo-search
                            • 분석 노드: 순수 함수 (텍스트 분석)
                            • 작가 Agent: LLM + 텍스트 생성  
                            • 최종 정리: 함수 노드

                            **** 사용된 도구:
                            • DuckDuckGo Search API (실시간 검색)
                            • 텍스트 분석 함수
                            • LLM 언어 모델 (Ollama)
                            """
            
            return {
                "user_request": state['user_request'],
                "research_data": state['research_data'], 
                "analysis_result": state['analysis_result'],
                "draft_content": state['draft_content'],
                "final_output": final_output,
                "current_step": "completed",
                "metadata": metadata
            }
        
        # 7. Graph 구성
        workflow = StateGraph(MultiAgentState)
        
        # 노드 추가
        workflow.add_node("researcher", researcher_agent)          # LLM Agent
        workflow.add_node("analyzer", analysis_function_node)      # 함수 노드
        workflow.add_node("writer", writer_agent)                  # LLM Agent  
        workflow.add_node("finalizer", finalize_output_node)       # 함수 노드
        
        # 순차적 흐름 설정 => 조건적 분기가 필요 없는 상황
        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "analyzer")
        workflow.add_edge("analyzer", "writer")
        workflow.add_edge("writer", "finalizer")
        workflow.add_edge("finalizer", END)
        
        # 컴파일
        app = workflow.compile()
        print("✅ 하이브리드 멀티 에이전트 시스템 생성 완료")
        
        return app
        
    except Exception as e:
        print(f"==== 하이브리드 멀티 에이전트 생성 실패: {e}")
        return None

def test_hybrid_multi_agent():
    """하이브리드 멀티 에이전트 테스트"""
    print("\==== 하이브리드 멀티 에이전트 테스트")
    print("="*50)
    
    agent_system = create_hybrid_multi_agent()
    if not agent_system:
        return
    
    test_cases = [
        # "ChatGPT 최신 기능에 대해 분석해줘",
        "2026년 AI 기술 동향을 알려주세요",
        # "duckduckgo-search 라이브러리 사용법은?",
        # "Python과 JavaScript 비교 분석"
    ]
    
    for i, test_request in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"==== 테스트 {i}: {test_request}")
        print("="*60)
        
        # 초기 상태 설정
        initial_state = {
            "user_request": test_request,
            "research_data": "",
            "analysis_result": "",
            "draft_content": "",
            "final_output": "",
            "current_step": "start",
            "metadata": {}
        }
        
        try:
            # 멀티 에이전트 실행
            print("==== 멀티 에이전트 협업 시작...")
            result = agent_system.invoke(initial_state)
            
            # 결과 출력
            print(f"\==== 최종 결과:")
            print(result['final_output'])
            
        except Exception as e:
            print(f"==== 테스트 실패: {e}")
        
        # print(f"\==== 다음 테스트까지 잠시 대기...")
        import time
        time.sleep(2)

def explain_hybrid_architecture():
    """하이브리드 아키텍처 설명"""
    print("\n🏗️ 하이브리드 아키텍처 구조")
    print("="*40)
    
    architecture = {
        "연구자 Agent (LLM)": {
            "역할": "실시간 정보 검색 및 필요성 판단",
            "도구": "duckduckgo-search 라이브러리",
            "특징": "실제 웹 검색, 최신 정보 수집",
            "비용": "💰💰💰"
        },
        "분석 노드 (함수)": {
            "역할": "데이터 분석 및 통계 처리",
            "도구": "텍스트 분석, 키워드 추출",
            "특징": "빠른 처리, 예측 가능한 결과",
            "비용": "💰"
        },
        "작가 Agent (LLM)": {
            "역할": "종합적인 콘텐츠 작성",
            "도구": "텍스트 생성, 구조화",
            "특징": "창의적 작성, 사용자 맞춤",
            "비용": "💰💰💰"
        },
        "정리 노드 (함수)": {
            "역할": "최종 출력 포맷팅",
            "도구": "템플릿 처리, 메타데이터 추가",
            "특징": "일관된 형식, 빠른 처리",
            "비용": "💰"
        }
    }
    
    for component, details in architecture.items():
        print(f"\n🔧 {component}")
        for key, value in details.items():
            print(f"   {key}: {value}")
    
    print(f"\n💡 하이브리드의 장점:")
    print("  • 총 LLM 호출: 2회 (연구자 + 작가)")
    print("  • 빠른 처리: 함수 노드로 분석/정리")
    print("  • 창의성: 필요한 곳에만 LLM 사용")
    print("  • 비용 효율: 순수 함수로 비용 절약")

def show_workflow_diagram():
    """워크플로우 다이어그램"""
    print(f"\n🔄 하이브리드 멀티 에이전트 워크플로우")
    print("="*45)
    
    workflow_diagram = """
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
    """
    
    print(workflow_diagram)
    
    print("\n⏱️ 예상 처리 시간:")
    print("  • 연구자 Agent: 30-60초")
    print("  • 분석 노드: 1-2초")
    print("  • 작가 Agent: 30-60초")
    print("  • 정리 노드: 1초")
    print("  • 총합: 약 1-2분")

if __name__ == "__main__":
    # 아키텍처 설명
    explain_hybrid_architecture()
    
    # 워크플로우 다이어그램
    show_workflow_diagram()
    
    # 실제 시스템 생성 및 테스트
    print(f"\n{'='*80}")
    print("🚀 하이브리드 멀티 에이전트 시스템 테스트")
    print("="*80)
    
    test_hybrid_multi_agent()