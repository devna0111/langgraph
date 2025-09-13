"""
vLLM 라이브러리 다운로드 가정 => 만약 없다면 pip install vllm 필요
Ollama 설치 가정, 미설치 시 홈페이지에서 다운로드 요망
Ollama 설치 시 자동 실행 가정 => 만약 실행중이지 않다면 ollama serve
"""
import subprocess
import json
import requests
import time
def test_ollama_model(model_name:str = 'devna0111-7b-q4') :
    start = time.time()
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": "넌 어떤 일을 할 수 있어?",
        "stream": False,
    }
    
    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        end = time.time()
        print(f"✓ 모델 테스트 완료! 소요시간: {end-start:.2f}초")
        return result.get('response', '응답 없음')
        
    else:
        print(f"✗ 모델 테스트 실패: {response.status_code}")
        end = time.time()
        print(f"✓ 모델 테스트 완료! 소요시간: {end-start:.2f}초")
        
def create_vllm_config(model_name : str = 'devna0111-7b-q4') :
    """vLLM 서버 설정 생성"""
    config = {
        "model": model_name,
        "host": "0.0.0.0",
        "port": 8000,
        "served-model-name": "test",
        "max-model-len": 4096,
        "gpu-memory-utilization": 0.8,
        "trust-remote-code": True
    }
    
    # 설정 파일 저장
    with open("vllm_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ vLLM 설정 파일 생성: vllm_config.json")
    return config

def start_vllm_server(model_name : str = 'devna0111-7b-q4'):
    """vLLM 서버 시작"""
    print(f"vLLM 서버 시작: {model_name}")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--served-model-name", "qwen2.5",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.8",
        "--trust-remote-code"
    ]
    
    print("명령어:", " ".join(cmd))
    print("서버를 시작합니다... (Ctrl+C로 중지)")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n서버가 중지되었습니다.")

def test_vllm_server(model_name : str = 'devna0111-7b-q4'):
    """vLLM 서버 테스트"""
    print("vLLM 서버 테스트 중...")
    
    # 서버 준비 대기
    for i in range(30):
        try:
            response = requests.get("http://localhost:8000/v1/models", timeout=2)
            if response.status_code == 200:
                print("✓ vLLM 서버 응답 확인")
                break
        except:
            print(f"서버 대기 중... ({i+1}/30)")
            time.sleep(2)
    else:
        print("✗ vLLM 서버 응답 없음")
        return False
    
    # 채팅 테스트
    try:
        url = "http://localhost:8000/v1/chat/completions"
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": "안녕하세요"}],
            "max_tokens": 100
        }
        
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print("✓ vLLM 응답:")
            print(f"  {content}")
            return True
        else:
            print(f"✗ vLLM 테스트 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ 테스트 오류: {e}")
        return False

def main():
    """메인 vLLM 설정 프로세스"""
    print("=== vLLM 서버 설정 ===")
    
    model_name = "devna0111-7b-q4"  # 이전에 만든 양자화 모델
    
    config = create_vllm_config(model_name)
    
    print("\n=== 설정 완료 ===")
    print(f"모델: {model_name}")
    print("vLLM 서버 시작 명령어:")
    print(f"python -m vllm.entrypoints.openai.api_server --model {model_name}")
    print("\n수동으로 서버를 시작하려면:")
    print(f"start_vllm_server({model_name})")
    return True

# 테스트 코드
if __name__ == "__main__":
    success = main()
    if success:
        print("\nvLLM 설정 완료!")
        
        # 서버 시작 여부 선택
        choice = input("\n지금 vLLM 서버를 시작하시겠습니까? (y/n): ")
        if choice.lower() == 'y':
            start_vllm_server("devna0111-7b-q4")
    else:
        print("\nvLLM 설정 실패")
    
