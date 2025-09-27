import subprocess
import json

def create_quantized_model(original_model, quantization_type, new_model_name) :
    ''' ollama 모델 양자화 함수 '''
    # Modelfile 생성 : 양자화 설정
    modelfile_content = f"""FROM {original_model}
                            # 양자화 설정
                            PARAMETER num_predict 512
                            PARAMETER num_gpu 1
                            PARAMETER num_thread 4
                            PARAMETER temperature 0.7
                            PARAMETER top_p 0.9

                            # 시스템 프롬프트
                            SYSTEM '당신은 devna0111[정종혁]이 개발한 ai agent입니다. 모든 답변은 한국어로만 대답합니다.친절하고 정확한 정보를 제공해주세요.중국어 답변은 금지입니다.'
                        """
    # Modelfile 저장
    with open("Modelfile",'w',encoding='utf-8') as f :
        f.write(modelfile_content)
    
    print(f"양자화 모델 생성 : {new_model_name}")
    print(f"원본 모델 : {original_model}")
    print(f"양자화 타입 : {quantization_type}")
    
    try :
        # ollama create 명령어로 양자화 모델 생성
        cmd =['ollama','create', new_model_name, "-f", "Modelfile"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print('양자화 모델 생성')
        return True
    except subprocess.CalledProcessError as e :
        print(f"양자화 모델 생성 실패 : {e}")
        return False

def test_model(model_name, test_prompt="안녕하세요") :
    '''모델 테스트'''
    try :
        cmd = ['ollama','run',model_name,test_prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 :
            print("***모델 응답***")
            print(result.stdout)
            return True
        else :
            print("***모델 응답 없음***")
            return False
    except subprocess.TimeoutExpired :
        print("모델 응답 시간 초과")
        return False
    except Exception as e :
        print(f"모델 테스트 오류 : {e}")
        return False

def get_model_info(model_name) :
    '''모델 정보 출력'''
    try :
        cmd = ['ollama', 'show', model_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 :
            print(f"***모델 정보 : {model_name}***")
            # 모델 크기 정보 파싱
            lines = result.stdout.split('\n')
            for line in lines :
                if 'parameters' in line.lower() or 'size' in line.lower() :
                    print(line)
            return True
        else :
            print(f"모델 정보 출력 실패 : {result.stderr}")
            return False
    except Exception as e :
        print(f"모델 정보 출력 오류 : {e}")
        return False

def main() :
    '''모델 양자화 프로세스'''
    original_model = "anpigon/qwen2.5-7b-instruct-kowiki:q6_k"
    # Q4 모델 생성
    q4_model_name = "devna0111-7b-q4"
    
    print(f"\n Q4양자화 모델 생성...")
    q4_success = create_quantized_model(original_model, "Q4", q4_model_name)
    
    if q4_success:
        print(f"\n2. 모델 정보 확인...")
        get_model_info(q4_model_name)
        
        print(f"\n3. 모델 테스트...")
        test_success = test_model(q4_model_name, "간단한 인사말을 해주세요.")
        
        if test_success:
            print(f"\n✓ 양자화 완료! 사용 가능한 모델: {q4_model_name}")
            return q4_model_name
        else:
            print(f"\n✗ 모델 테스트 실패")
            return None
    else:
        print(f"\n✗ 양자화 실패")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"다음 모듈에서 사용할 모델: {result}")
    else:
        print("양자화 과정을 다시 확인하세요.")