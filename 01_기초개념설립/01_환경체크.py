# 01_환경체크.py
import subprocess
import sys
import torch

def check_python_version():
    """Python 버전 체크 (3.8+ 필요)"""
    version = sys.version_info
    print(f"Python 버전: {version.major}.{version.minor}.{version.micro}")
    return version.major >= 3 and version.minor >= 8

def check_cuda():
    """CUDA 사용 가능 여부 체크"""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA: 사용 가능")
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram:.1f}GB")
        return True, vram
    else:
        print("CUDA: 사용 불가")
        return False, 0

def check_ollama():
    """Ollama 설치 및 실행 상태 체크"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Ollama: 설치됨 ({result.stdout.strip()})")
            return True
        else:
            print("Ollama: 설치되지 않음")
            return False
    except FileNotFoundError:
        print("Ollama: 설치되지 않음")
        return False

def check_model_exists(model_name):
    """Ollama 모델 존재 여부 체크"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True)
        if model_name in result.stdout:
            print(f"모델: {model_name} 존재함")
            return True
        else:
            print(f"모델: {model_name} 존재하지 않음")
            return False
    except Exception as e:
        print(f"모델 체크 오류: {e}")
        return False

def main():
    """전체 환경 체크"""
    print("=== 환경 체크 시작 ===")
    
    # 1. Python 버전
    python_ok = check_python_version()
    
    # 2. CUDA 체크
    cuda_ok, vram = check_cuda()
    
    # 3. Ollama 체크
    ollama_ok = check_ollama()
    
    # 4. 모델 체크
    model_name = "anpigon/qwen2.5-7b-instruct-kowiki:latest"
    model_ok = check_model_exists(model_name)
    
    print("\n=== 체크 결과 ===")
    print(f"Python 3.8+: {'✓' if python_ok else '✗'}")
    print(f"CUDA: {'✓' if cuda_ok else '✗'}")
    print(f"VRAM: {vram:.1f}GB")
    print(f"Ollama: {'✓' if ollama_ok else '✗'}")
    print(f"모델: {'✓' if model_ok else '✗'}")
    
    return all([python_ok, cuda_ok, ollama_ok, model_ok])

# 테스트 코드
if __name__ == "__main__":
    if main():
        print("\n모든 환경 준비 완료!")
    else:
        print("\n환경 설정이 필요합니다.")