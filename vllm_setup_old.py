import subprocess
import sys
import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig :
    ''' 모델 설정 정보를 담는 데이터 클래스'''
    name : str
    size : str
    quantization : str
    memory_requirement : float # GB
    context_length : int
    download_url : Optional[str] = None
    
class VLLMSetup:
    ''' vLLM과 Qwen2.5 모델 설정을 담당하는 클래스'''
    def __init__(self, base_dir : str = './models') :
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # RTX 4060(환경 컴퓨터) 최적화 된 모델 설정
        self.available_models = {
            "qwen2.5-7b-instruct-kowiki-q8_0" : ModelConfig(
                name = "qwen2.5-7b-instruct-kowiki-q8_0",
                size = "7B",
                quantization = "Q8_0", # INT8 양자화
                memory_requirement = 7.5, # GB
                context_length = 32768, # ??
            ),
            "qwen2.5-7b-instruct-kowiki-q4_k_m":ModelConfig(
                name = "qwen2.5-7b-instruct-kowiki-q4_k_m",
                size = "7B",
                quantization = "Q4_K_M", # INT4 양자화 : 메모리 절약
                memory_requirement = 4.5, # GB
                context_length = 32768, # ??
            )
        }
        
        self.selected_model = None
        self.vllm_server_process = None

    def check_ollama_installed(self) -> Tuple[bool, Optional[str]] :
        ''' ollama 설치 여부를 확인 '''
        try :
            result = subrocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.strip()
            return True, version
        except (subprocess.CalledProcessError, FileNotFoundError) :
            return False, None
    
    def install_ollama(self) -> bool :
        ''' ollama 설치 (Ubuntu, Linux 환경) '''
        print("ollama 설치 시작")
        
        try :
            # Ollama 설치 스크립트 실행
            install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            result = subprocess.run(
                install_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            print("*****Ollama 설치 완료*****")
            return True
        except subprocess.CalledProcessError as e :
            print(f"Ollama 설치 실패 : {e}")
            print("https://ollama.com/download 방문 후 Linux 버젼 설치 요망")
            return False
    
    def start_ollama_server(self)->bool :
        ''' ollama 서비스 시작 '''
        try :
            # 백그라운드에서 Ollama 서비스 실행
            subprocess.Popen(
                ['ollama','serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            # 서비스가 시작할 때 까지 대기
            for i in range(10) :
                try :
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200 :
                        print("Ollama 서비스 시작!!!!")
                        return True
                except requests.exceptions.RequestException as e :
                    print(f"Ollama 서비스 시작 대기중 ({i+1}/10)...")
                    time.sleep(2)
            print("Ollama 서비스 시작 실패")
            return False
        except Exception as e :
            print(f"Ollama 서비스 시작 중 오류 : {e}")
            return False
    def select_optimal_model(self, available_vram: float) -> ModelConfig:
        """사용 가능한 VRAM에 따라 최적 모델 선택"""
        print(f"🎯 사용 가능한 VRAM: {available_vram:.1f}GB")
        
        # VRAM에 따른 모델 선택
        if available_vram >= 7.5:
            model = self.available_models["qwen2.5:7b-instruct-q8_0"]
            print("✅ Q8_0 (INT8) 양자화 모델 선택 - 최고 품질")
        elif available_vram >= 4.5:
            model = self.available_models["qwen2.5:7b-instruct-q4_k_m"] 
            print("✅ Q4_K_M (INT4) 양자화 모델 선택 - 메모리 효율")
        else:
            model = self.available_models["qwen2.5:7b-instruct-q4_k_m"]
            print("⚠️  VRAM이 부족하지만 Q4_K_M 모델로 시도합니다.")
        
        self.selected_model = model
        return model
    def download_model_with_ollama(self, model_config : ModelConfig) -> bool:
        """ Ollama 모델 다운로드"""
        print(f"{model_config.name} 모델 다운로드 시작(size : {model_config.size}, quantization : {model_config.quantization})")
        print(f"예상 메모리 : {model_config.memory_requirement:.1f}GB")
        
        try :
            # 모델 다운로드
            result = subprocess.run(
                ['ollama', 'pull', model_config.name],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text=True,
            )
            
            # 실시간 진행 상황 출력
            while True :
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # process 대기 완료
            return_code = process.wait()
            if return_code == 0:
                print(f"{model_config.name} 모델 다운로드 완료")
                return True
            else:
                print(f"{model_config.name} 모델 다운로드 실패")
                return False
        except Exception as e :
            print(f"Ollama 모델 다운로드 실패 : {e}")
            return False
    def verify_model_downloaded(self,model_name :str) -> bool :
        """ 모델 정상 다운로드 점검 """
        try :
            result = subprocess.run(
                ['ollama','list'],
                capture_output=True,
                text=True,
                check=True
            )
            # 다운로드 된 모델 목록에서 확인
            downloaded_models = result.stdout
            if model_name in downloaded_models :
                print(f"{model_name} 모델 다운로드 완료")
                return True
            else :
                print(f"{model_name} 모델 다운로드 실패")
                return False
        except Exception as e :
            print(f"모델 확인 중 실패오류 : {e}")
            return False
    def install_vllm(self) -> bool :
        '''vLLM 설치'''
        print("vLLM 설치 시작")
        
        # CUDA 버젼에 맞는 Pytorch 설치 확인
        try :
            import torch
            if torch.cuda.is_available():
                print("CUDA 정상")
            else :
                print("CUDA 에러, CPU 모드 실행")
        except ImportError:
            print("torch 설치 요망")
        
        # vllm 설치 명령어들
        install_commands = [
            # 기본 vllm 설치
            [sys.executable, '-m', 'pip', 'install', 'vllm'],
            # 추가 의존성 설치
            [sys.executable, '-m', 'pip', 'install', 'transformers', 'accelerate'],
        ]
        
        for cmd in install_commands:
            try :
                print(f"실행 중 : {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("설치 완료")
            except subprocess.CalledProcessError as e :
                print(f"설치 실패 : {e}")
                print(f"오류 출력 : {e.stderr}")
                return False
        
        # 설치 확인
        try : 
            import vllm
            print("vLLM 설치 완료", vllm.__version__)
            return True
        except ImportError:
            print("vLLM 설치 실패")
            return False
    
# 모듈 2: vLLM 설치 및 Ollama Qwen2.5:7b 설정
# File: vllm_setup.py

import subprocess
import sys
import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """모델 설정 정보를 담는 데이터클래스"""
    name: str
    size: str
    quantization: str
    memory_requirement: float  # GB
    context_length: int
    download_url: Optional[str] = None

class VLLMSetup:
    """vLLM과 Qwen2.5 모델 설정을 담당하는 클래스"""
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # RTX 4060 8GB 환경에 최적화된 모델 설정
        self.available_models = {
            "qwen2.5:7b-instruct-q8_0": ModelConfig(
                name="qwen2.5:7b-instruct-q8_0",
                size="7B",
                quantization="Q8_0",  # INT8 양자화
                memory_requirement=7.5,  # GB
                context_length=32768
            ),
            "qwen2.5:7b-instruct-q4_k_m": ModelConfig(
                name="qwen2.5:7b-instruct-q4_k_m", 
                size="7B",
                quantization="Q4_K_M",  # INT4 양자화 (더 절약)
                memory_requirement=4.5,  # GB
                context_length=32768
            )
        }
        
        self.selected_model = None
        self.vllm_server_process = None
        
    def check_ollama_installed(self) -> Tuple[bool, Optional[str]]:
        """Ollama 설치 여부 확인"""
        try:
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            version = result.stdout.strip()
            return True, version
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, None
    
    def install_ollama(self) -> bool:
        """Ollama 설치 (Ubuntu/Linux 환경)"""
        print("🚀 Ollama 설치를 시작합니다...")
        
        try:
            # Ollama 설치 스크립트 실행
            install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            result = subprocess.run(
                install_cmd,
                shell=True,
                check=True,
                text=True,
                capture_output=True
            )
            
            print("✅ Ollama 설치가 완료되었습니다!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Ollama 설치 실패: {e}")
            print("수동 설치 가이드:")
            print("1. https://ollama.com/download 방문")
            print("2. Linux 버전 다운로드 후 설치")
            return False
    
    def start_ollama_service(self) -> bool:
        """Ollama 서비스 시작"""
        try:
            # 백그라운드에서 Ollama 서비스 시작
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 서비스가 시작될 때까지 대기
            for i in range(10):
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("✅ Ollama 서비스가 시작되었습니다!")
                        return True
                except requests.exceptions.RequestException:
                    print(f"⏳ Ollama 서비스 시작 대기 중... ({i+1}/10)")
                    time.sleep(2)
            
            print("❌ Ollama 서비스 시작에 실패했습니다.")
            return False
            
        except Exception as e:
            print(f"❌ Ollama 서비스 시작 중 오류: {e}")
            return False
    
    def select_optimal_model(self, available_vram: float) -> ModelConfig:
        """사용 가능한 VRAM에 따라 최적 모델 선택"""
        print(f"🎯 사용 가능한 VRAM: {available_vram:.1f}GB")
        
        # VRAM에 따른 모델 선택
        if available_vram >= 7.5:
            model = self.available_models["qwen2.5:7b-instruct-q8_0"]
            print("✅ Q8_0 (INT8) 양자화 모델 선택 - 최고 품질")
        elif available_vram >= 4.5:
            model = self.available_models["qwen2.5:7b-instruct-q4_k_m"] 
            print("✅ Q4_K_M (INT4) 양자화 모델 선택 - 메모리 효율")
        else:
            model = self.available_models["qwen2.5:7b-instruct-q4_k_m"]
            print("⚠️  VRAM이 부족하지만 Q4_K_M 모델로 시도합니다.")
        
        self.selected_model = model
        return model
    
    def download_model_with_ollama(self, model_config: ModelConfig) -> bool:
        """Ollama를 통해 모델 다운로드"""
        print(f"📥 {model_config.name} 모델을 다운로드합니다...")
        print(f"   크기: {model_config.size}, 양자화: {model_config.quantization}")
        print(f"   예상 메모리: {model_config.memory_requirement}GB")
        
        try:
            # Ollama pull 명령어 실행
            process = subprocess.Popen(
                ["ollama", "pull", model_config.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 실시간 진행상황 출력
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"   {output.strip()}")
            
            # 프로세스 완료 대기
            return_code = process.wait()
            
            if return_code == 0:
                print("✅ 모델 다운로드가 완료되었습니다!")
                return True
            else:
                error_output = process.stderr.read()
                print(f"❌ 모델 다운로드 실패: {error_output}")
                return False
                
        except Exception as e:
            print(f"❌ 모델 다운로드 중 오류: {e}")
            return False
    
    def verify_model_download(self, model_name: str) -> bool:
        """모델이 제대로 다운로드되었는지 확인"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # 다운로드된 모델 목록에서 확인
            downloaded_models = result.stdout
            if model_name in downloaded_models:
                print(f"✅ {model_name} 모델 확인 완료!")
                return True
            else:
                print(f"❌ {model_name} 모델을 찾을 수 없습니다.")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 모델 확인 중 오류: {e}")
            return False
    
    def install_vllm(self) -> bool:
        """vLLM 설치"""
        print("🚀 vLLM 설치를 시작합니다...")
        
        # CUDA 버전에 맞는 PyTorch 설치 확인
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA PyTorch 감지: {torch.version.cuda}")
            else:
                print("⚠️  CUDA PyTorch가 감지되지 않았습니다. CPU 모드로 실행됩니다.")
        except ImportError:
            print("❌ PyTorch가 설치되지 않았습니다!")
            return False
        
        # vLLM 설치 명령어들
        install_commands = [
            # 기본 vLLM 설치
            [sys.executable, "-m", "pip", "install", "vllm"],
            # 추가 의존성 설치  
            [sys.executable, "-m", "pip", "install", "transformers", "accelerate"]
        ]
        
        for cmd in install_commands:
            try:
                print(f"실행 중: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("✅ 설치 완료!")
            except subprocess.CalledProcessError as e:
                print(f"❌ 설치 실패: {e}")
                print(f"오류 출력: {e.stderr}")
                return False
        
        # 설치 확인
        try:
            import vllm
            print(f"✅ vLLM 설치 확인 완료! 버전: {vllm.__version__}")
            return True
        except ImportError:
            print("❌ vLLM 설치 확인 실패!")
            return False
    
    def create_vllm_config(self, model_config: ModelConfig) -> Dict:
        """vLLM 서버 설정 생성"""
        config = {
            "model": model_config.name,
            "host": "0.0.0.0",
            "port": 8000,
            "gpu-memory-utilization": 0.8,  # GPU 메모리의 80% 사용
            "max-model-len": min(model_config.context_length, 8192),  # 컨텍스트 길이 제한
            "trust-remote-code": True,
            "enforce-eager": True,  # 메모리 효율성을 위한 설정
        }
        
        # RTX 4060 최적화 설정
        if model_config.quantization.startswith("Q4"):
            config.update({
                "quantization": "awq",  # 4비트 양자화 활용
                "max-num-batched-tokens": 2048,  # 배치 크기 제한
            })
        
        return config
    
    def save_config(self, config: Dict, filename: str = "vllm_config.json"):
        """설정을 JSON 파일로 저장"""
        config_path = self.base_dir / filename
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"설정 파일 저장: {config_path}")
        return config_path
    
    def setup_complete_environment(self, available_vram: float = 8.0) -> bool:
        """전체 환경 설정을 자동으로 수행"""
        print("vLLM + Qwen2.5 환경 설정 시작!")
        print("=" * 60)
        
        # 1. Ollama 확인 및 설치
        print("Ollama 확인")
        ollama_installed, version = self.check_ollama_installed()
        
        if not ollama_installed:
            if not self.install_ollama():
                return False
        else:
            print(f"Ollama 이미 설치됨: {version}")
        
        # 2. Ollama 서비스 시작
        print("\n Ollama 서비스 시작 중...")
        if not self.start_ollama_service():
            return False
        
        # 3. 최적 모델 선택
        print("\n 최적 모델 선택 중...")
        model_config = self.select_optimal_model(available_vram)
        
        # 4. 모델 다운로드
        print("\n 모델 다운로드 중...")
        if not self.download_model_with_ollama(model_config):
            return False
        
        # 5. 모델 다운로드 확인
        print("\n 모델 확인 중...")
        if not self.verify_model_download(model_config.name):
            return False
        
        # 6. vLLM 설치
        print("\n vLLM 설치 중...")
        if not self.install_vllm():
            return False
        
        # 7. 설정 파일 생성
        print("\n 설정 파일 생성 중...")
        vllm_config = self.create_vllm_config(model_config)
        self.save_config(vllm_config)
        
        print("\n 모든 설정 완료")
        print(f"선택된 모델: {model_config.name}")
        print(f"메모리 사용량: {model_config.memory_requirement}GB")
        print("다음 단계에서 vLLM 서버를 시작할 수 있습니다.")
        
        return True

if __name__ == "__main__":
    # RTX 4060 8GB 환경 기준으로 설정
    setup = VLLMSetup()
    
    # 환경 설정 실행
    success = setup.setup_complete_environment(available_vram=8.0)
    
    if success:
        print("\n🚀 설정 완료! 다음 명령어로 vLLM 서버를 시작할 수 있습니다:")
        print("python -m vllm.entrypoints.openai.api_server --config vllm_config.json")
    else:
        print("\n❌ 설정 중 오류가 발생했습니다. 로그를 확인하세요.")