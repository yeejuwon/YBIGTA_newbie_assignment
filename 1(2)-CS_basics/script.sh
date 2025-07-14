
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
if ! command -v curl &> /dev/null; then
    echo "[INFO] curl이 설치되어 있지 않아 설치를 진행합니다..."
    apt update && apt install -y curl
fi

echo "[INFO] Anaconda(또는 Miniconda)가 설치되어 있지 않으면 자동으로 설치를 진행합니다..."

if command -v conda &> /dev/null; then
    echo "[INFO] Conda가 이미 설치되어 있습니다."
else
    echo "[INFO] Conda가 설치되어 있지 않습니다. Miniconda를 설치합니다..."
    
    # 시스템 아키텍처 확인
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            ARCH="x86_64"
            ;;
        aarch64|arm64)
            ARCH="aarch64"
            ;;
        *)
            echo "[ERROR] 지원하지 않는 아키텍처입니다: $ARCH"
            exit 1
            ;;
    esac
    
    # Miniconda 설치 파일 다운로드
    INSTALLER_NAME="Miniconda3-latest-Linux-${ARCH}.sh"
    DOWNLOAD_URL="https://repo.anaconda.com/miniconda/${INSTALLER_NAME}"
    
    echo "[INFO] Miniconda 설치 파일을 다운로드합니다..."
    if ! curl -L -o "$INSTALLER_NAME" "$DOWNLOAD_URL"; then
        echo "[ERROR] 다운로드에 실패했습니다."
        exit 1
    fi
    
    # Miniconda 설치
    echo "[INFO] Miniconda를 설치합니다..."
    bash "$INSTALLER_NAME" -b -p "$HOME/miniconda3"
    
    # 설치 파일 정리
    rm -f "$INSTALLER_NAME"
    
    # PATH 설정
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # conda 초기화 및 사용 가능하도록 설정
    "$HOME/miniconda3/bin/conda" init bash
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    echo "[INFO] Miniconda 설치가 완료되었습니다!"
fi

# Conda 환경 생성 및 활성화
## TODO
conda create -n myenv python=3.10
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
echo "[INFO] 필요한 패키지를 설치합니다..."
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    ## TODO
    name="${file%.py}"
    input_file="../input/${name}_input"
    output_file="../output/${name}_output"

    echo "[INFO} 실행 중: $file -> $output_file"
    python "$file" < "$input_file" > "$output_file"
done

# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
mypy . > ../mypy_log.txt 2>&1
echo "[INFO] mypy 테스트 완료 및 결과 mypy_log.txt에 저장"

# conda.yml 파일 생성
## TODO
conda env export --no-builds > ../conda.yml
echo "[INFO] conda.yml 파일 생성 완료"

# 가상환경 비활성화
## TODO
conda deactivate
echo "[INFO] 가상환경 비활성화: 성공"