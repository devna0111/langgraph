"""
- 환경확인 및 기본 설정
"""

# 필요 라이브러리 : 설치 완료
import subprocess 
import sys
import torch
import platform
import psutil
import os
from typing import Dict, Tuple, Optional

class EnvironmentChecker:
    '''시스템 환경을 확인하고 vLLM 설정에 필요한 정보를 수집하는 클래스'''
    def __init__(self):
        self.system_info = {}
        self.gpu_info = {}
        self.requirements = {
            'python_version' : (3,8),
            'cuda_version' : 12.0,
            'min_vram' : 4, # GB
            'max_vram' : 8, # GB
        }
    def check_python_version(self) -> Tuple[bool, str]:
        '''python 버전 체크'''
        current_version = sys.version_info
        min_version = self.requirements['python_version']
        
        version_str = f"{current_version.major}.{current_version.micro}"
        is_compatible = current_version >= min_version
        
        return is_compatible, version_str
    
    def check_cuda_availability(self) -> Tuple[bool, Optional[str]]:
        '''CUDA 설치 및 Pytorch CUDA 지원 확인'''
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            return True, cuda_version
        else:
            return False, None
    def check_gpu_memory(self) -> Tuple[bool, Dict]:
        """GPU  메모리 정보 확인"""
        if not torch.cuda.is_available():
            return False, {}
        
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            total_memory_gb = total_memory / (1024 ** 3)
            gpu_info[f"gpu_{i}"] = {
                'name': gpu_name,
                'total_memory_gb': round(total_memory_gb,2),
                'is_sufficient': total_memory_gb >= self.requirements['min_vram'],
            }
        return True, gpu_info
    
    def check_system_memory(self) -> Tuple[bool, float]:
        """시스템 ram 확인"""
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        is_sufficient = total_ram_gb >= self.requirements['min_vram']
        
        return is_sufficient, round(total_ram_gb,2)
    
    def check_disk_space(self,path:str ='.') -> Tuple[bool, float]:
        '''디스크 공간 확인(모델 로컬 저장을 위함)'''
        disk_usage = psutil.disk_usage(path)
        fee_space = disk_usage.free / (1024 ** 3)
        # 최소 50gb 이상 권장 (모델 + 캐시)
        is_sufficient = fee_space >= 50
        
        return is_sufficient, round(fee_space,2)
    def get_system_info(self) -> Dict:
        '''시스템 정보 수집'''
        python_ok, python_version = self.check_python_version()
        cuda_ok, cuda_version = self.check_cuda_availability()
        gpu_ok, gpu_details  = self.check_gpu_memory()
        ram_ok, ram_size = self.check_system_memory()
        disk_ok, disk_space = self.check_disk_space()
        system_info = {
            'platform' : platform.system(),
            'platform_version' : platform.version(),
            'architecture' : platform.machine(),
            'python' : {
                'version' : python_version,
                'compatible' : python_ok,
            },
            'cuda' : {
                'available' : cuda_ok,
                'version' : cuda_version,
                'compatible' : cuda_ok and cuda_version is not None,
            },
            'gpu' : {
                'available' : gpu_ok,
                'details' : gpu_details,
            },
            'memory' : {
                'ram_gb' : ram_size,
                'ram_sufficient' : ram_ok,
                'disk_free_gb' : disk_space,
                'disk_sufficient' : disk_ok,
            }
        }
        return system_info
    
    def print_environment_report(self):
        """환경 정보 출력"""
        info = self.get_system_info()
        
        print("환경 확인 보고서")
        print("=" * 50)
        
        # 기본 시스템 정보
        print(f"운영체제: {info['platform']} ({info['architecture']})")
        print(f"Python: {info['python']['version']} {'ok' if info['python']['compatible'] else 'ERROR'}")
        
        # CUDA 정보
        cuda_status = "ok" if info['cuda']['compatible'] else "ERROR"
        print(f"**CUDA: {info['cuda']['version'] or 'Not Available'} {cuda_status}")
        
        # GPU 정보
        if info['gpu']['available']:
            print("***GPU 정보:")
            for gpu_id, gpu_data in info['gpu']['details'].items():
                status = "ok" if gpu_data['is_sufficient'] else "ERROR"
                print(f"   - {gpu_data['name']}: {gpu_data['total_memory_gb']}GB {status}")
        else:
            print("****GPU: 사용 불가 ERROR")
        
        # 메모리 정보
        ram_status = "ok" if info['memory']['ram_sufficient'] else "ERROR"
        disk_status = "ok" if info['memory']['disk_sufficient'] else "ERROR"
        print(f"****RAM: {info['memory']['ram_gb']}GB {ram_status}")
        print(f"**여유 디스크: {info['memory']['disk_free_gb']}GB {disk_status}")
        
        # 전체 호환성 체크
        print("\n**vLLM 호환성 체크:")
        all_compatible = all([
            info['python']['compatible'],
            info['cuda']['compatible'],
            any(gpu['is_sufficient'] for gpu in info['gpu']['details'].values()) if info['gpu']['details'] else False,
            info['memory']['ram_sufficient'],
            info['memory']['disk_sufficient']
        ])
        
        if all_compatible:
            print("모든 요구사항 만족, vLLM 설치를 진행 가능")
        else:
            print("일부 요구사항을 만족하지 않습니다. 아래 권장사항을 확인하세요:")
            if not info['python']['compatible']:
                print("   - Python 3.8 이상 필요")
            if not info['cuda']['compatible']:
                print("   - CUDA 설치 또는 PyTorch CUDA 버전 확인 필요")
            if not any(gpu['is_sufficient'] for gpu in info['gpu']['details'].values()) if info['gpu']['details'] else True:
                print("   - 최소 4GB VRAM의 GPU 필요")
            if not info['memory']['ram_sufficient']:
                print("   - 최소 16GB RAM 권장")
            if not info['memory']['disk_sufficient']:
                print("   - 최소 50GB 여유 디스크 공간 필요")
                
def create_requirements_file(filename:str='requirements.txt'):
    """프로젝트에 필요한 라이브러리 목록을 requirements.txt로 생성"""
    requirements = [
        "# vLLM 및 관련 라이브러리(torch는 cuda 버젼 체크하면서 설치 필요)",
        "vllm>=0.2.7",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "tokenizers>=0.14.0",
        "",
        "# LangGraph 및 LangChain",
        "langgraph>=0.0.30",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "",
        "# 유틸리티",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "",
        "# 개발 및 테스트",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.1.0",
        "",
        "# 모니터링",
        "psutil>=5.9.0",
        "nvidia-ml-py>=12.535.0"
    ]
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(requirements))
    
    print("📄 requirements.txt 파일이 생성되었습니다.")
    
if __name__ == "__main__":
    # 환경 체커 실행
    checker = EnvironmentChecker()
    checker.print_environment_report()
    
    # 필요 라이브러리 생성
    print('='*50)
    create_requirements_file('standard_requirements.txt')