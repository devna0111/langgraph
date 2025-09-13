# 03_vLLM서버설정.py
import subprocess
import requests
import time
import json

def install_vllm():
    """vLLM 설치"""
    print("vLLM 설치 중...")
    
    try:
        cmd = ["pip", "install", "vllm"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ vLLM 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ vLLM 설치 실패: {e.stderr}")
        return False

def check_vllm_installed():
    """vLLM 설치 여부 확인"""
    try:
        import vllm
        print(f"✓ vLLM 설치됨 (버전: {vllm.__version__})")
        return True
    except ImportError:
        print("✗ vLLM 설치되지 않음")
        return False

def start_ollama_server():
    """Ollama 서버 시작"""
    print("Ollama 서버 시작 중...")
    
    try:
        # 백그라운드에서 서버 시작
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # 서버 준비 대기
        for i in range(10):
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print("✓ Ollama 서버 실행 중")
                    return True
            except:
                print(f"대기 중... ({i+1}/10)")
                time.sleep(2)
        
        print("✗ Ollama 서버 시작 실패")
        return False
    except Exception as e:
        print(f"✗ 서버 시작 오류: {e}")
        return False

def test_ollama_model(model_name):
    """Ollama 모델 테스트"""
    print(f"모델 테스트: {model_name}")
    
    try:
        # Ollama API로 테스트
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model_name,
            "prompt": "안녕하세요",
            "stream": False
        }
        
        response = requests.post(url, json=data, timeout=30)
        print(f"HTTP 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 모델 응답 성공!")
            
            # 응답 데이터 구조 확인
            print("응답 데이터 키:", list(result.keys()))
            
            # 실제 응답 내용 추출
            ai_response = result.get('response', '응답 없음')
            print(f"AI 응답 길이: {len(ai_response)}글자")
            print(f"AI 응답 (처음 100글자): {ai_response[:100]}...")
            
            return True
        else:
            print(f"✗ HTTP 오류: {response.status_code}")
            print(f"오류 내용: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ 응답 시간 초과 (30초)")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ 연결 실패 (Ollama 서버가 실행되지 않음)")
        return False
    except Exception as e:
        print(f"✗ 테스트 오류: {e}")
        return False

def create_vllm_config(model_name):
    """vLLM 서버 설정 생성"""
    config = {
        "model": model_name,
        "host": "0.0.0.0",
        "port": 8000,
        "served-model-name": "qwen2.5",
        "max-model-len": 4096,
        "gpu-memory-utilization": 0.8,
        "trust-remote-code": True
    }
    
    # 설정 파일 저장
    with open("vllm_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ vLLM 설정 파일 생성: vllm_config.json")
    return config

def start_vllm_server(model_name):
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

def test_vllm_server():
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
            "model": "qwen2.5",
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
    
    model_name = "qwen2.5-7b-q4"  # 이전에 만든 양자화 모델
    
    # 1. vLLM 설치 확인
    if not check_vllm_installed():
        if not install_vllm():
            return False
    
    # 2. Ollama 서버 시작
    if not start_ollama_server():
        return False
    
    # 3. 모델 테스트
    if not test_ollama_model(model_name):
        print(f"모델 {model_name}을 먼저 생성하세요")
        return False
    
    # 4. vLLM 설정 생성
    config = create_vllm_config(model_name)
    
    print("\n=== 설정 완료 ===")
    print(f"모델: {model_name}")
    print("vLLM 서버 시작 명령어:")
    print(f"python -m vllm.entrypoints.openai.api_server --model {model_name}")
    print("\n수동으로 서버를 시작하려면:")
    print("start_vllm_server('qwen2.5-7b-q4')")
    
    return True

# 테스트 코드
if __name__ == "__main__":
    success = main()
    if success:
        print("\nvLLM 설정 완료!")
        
        # 서버 시작 여부 선택
        choice = input("\n지금 vLLM 서버를 시작하시겠습니까? (y/n): ")
        if choice.lower() == 'y':
            start_vllm_server("qwen2.5-7b-q4")
    else:
        print("\nvLLM 설정 실패")