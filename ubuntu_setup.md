# 🐧 Ubuntu 환경 설정 가이드

Hardware-Data-Parameter Co-Design Framework를 Ubuntu 환경에서 실행하기 위한 완전 가이드입니다.

## 🚀 빠른 시작 (Dry Run)

### 1단계: 저장소 클론 및 진입
```bash
git clone <repository-url>
cd YunMin-mamba-v1
```

### 2단계: Dry Run 실행 (자동 설정)
```bash
# 실행 권한 부여
chmod +x ubuntu_dry_run.sh

# Dry run 실행 (모든 설정 자동화)
./ubuntu_dry_run.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
- ✅ 시스템 환경 확인
- ✅ Python 및 의존성 설치
- ✅ GPU 상태 확인
- ✅ 가상환경 생성 및 활성화
- ✅ 프로젝트 구조 검증
- ✅ 모델 임포트 테스트
- ✅ 설정 파일 검증
- ✅ 최소 성능 벤치마크

## 📋 시스템 요구사항

### 최소 요구사항
- **OS**: Ubuntu 18.04+ (20.04/22.04 권장)
- **RAM**: 8GB 이상
- **저장공간**: 10GB 이상
- **Python**: 3.8+ (3.9+ 권장)

### 권장 요구사항
- **OS**: Ubuntu 22.04 LTS
- **RAM**: 32GB 이상
- **GPU**: NVIDIA GPU (CUDA 지원)
- **CUDA**: 11.8+ 또는 12.1+
- **저장공간**: 50GB+ (모델 체크포인트용)

## 🔧 수동 설정 (고급 사용자용)

### 1. 시스템 패키지 설치
```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 기본 개발 도구
sudo apt install -y python3 python3-pip python3-venv git curl wget

# GPU 사용시 (NVIDIA)
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit
```

### 2. Python 가상환경 설정
```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip
```

### 3. 의존성 설치

#### CPU 버전 (테스트용)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets pyyaml numpy psutil
```

#### GPU 버전 (CUDA 12.1)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pyyaml numpy psutil
```

#### 전체 의존성 (requirements.txt 사용)
```bash
pip install -r requirements.txt
```

## 🎯 실행 옵션

### Option 1: Dry Run (빠른 테스트)
```bash
# 설정 확인 및 최소 테스트
python3 main.py --config configs/dry_run_config.yaml --mode full_pipeline --debug

# 또는 개별 테스트
python3 train.py --config configs/dry_run_config.yaml --phase pretrain --model baseline
```

### Option 2: CPU 실행 (GPU 없는 환경)
```bash
# unified_config.yaml에서 device: "cpu" 설정 후
python3 main.py --config configs/unified_config.yaml --mode full_pipeline
```

### Option 3: GPU 실행 (풀 스케일)
```bash
# unified_config.yaml에서 device: "cuda" 설정 후
python3 main.py --config configs/unified_config.yaml --mode full_pipeline
```

## 🔍 환경 검증

### GPU 확인
```bash
# NVIDIA GPU 상태 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version

# PyTorch CUDA 지원 확인
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 메모리 확인
```bash
# RAM 확인
free -h

# 디스크 확인
df -h

# GPU 메모리 확인 (NVIDIA)
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

## 📊 성능 모니터링

### 기본 모니터링
```bash
# CPU/메모리 모니터링
htop

# GPU 모니터링 (실시간)
watch -n 1 nvidia-smi

# 디스크 I/O 모니터링
iotop
```

### 고급 모니터링
```bash
# 프로세스별 GPU 사용량
nvidia-smi pmon

# 상세 GPU 정보
nvidia-smi -l 1 --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. CUDA 관련 오류
```bash
# CUDA 버전 불일치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 라이브러리 경로 문제
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 2. 메모리 부족 오류
```bash
# configs/unified_config.yaml에서 배치 크기 줄이기
micro_batch_size: 4  # 8에서 4로 줄이기

# 또는 CPU 모드로 전환
device: "cpu"
```

#### 3. 의존성 설치 오류
```bash
# 가상환경 재생성
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### 4. 권한 오류
```bash
# 스크립트 실행 권한
chmod +x ubuntu_dry_run.sh

# 디렉토리 권한
sudo chown -R $USER:$USER ./
```

### 로그 확인
```bash
# 실험 로그 확인
tail -f experiments/*/pipeline.log

# 시스템 로그 확인
dmesg | grep -i cuda
journalctl -u nvidia-persistenced
```

## 📈 성능 최적화

### 시스템 최적화
```bash
# CPU 성능 모드 설정
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# GPU 성능 모드 설정
nvidia-smi -pm 1
nvidia-smi -ac 1215,1410  # 메모리,그래픽 클럭 (GPU 모델에 따라 조정)
```

### 환경 변수 최적화
```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## 🔄 자동화 스크립트

### 전체 파이프라인 실행
```bash
#!/bin/bash
# run_full_pipeline.sh

set -e

echo "🚀 Starting Full Pipeline on Ubuntu"

# 환경 활성화
source venv/bin/activate

# GPU 확인
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU detected"
    CONFIG="configs/unified_config.yaml"
else
    echo "⚠️  No GPU, using CPU"
    CONFIG="configs/dry_run_config.yaml"
fi

# 파이프라인 실행
python3 main.py --config $CONFIG --mode full_pipeline --experiment_name ubuntu_$(date +%Y%m%d_%H%M%S)

echo "✅ Pipeline completed!"
```

### 모니터링 스크립트
```bash
#!/bin/bash
# monitor.sh

# 터미널 분할하여 모니터링
tmux new-session -d -s monitor
tmux split-window -h
tmux select-pane -t 0
tmux send-keys 'watch -n 1 nvidia-smi' C-m
tmux select-pane -t 1
tmux send-keys 'htop' C-m
tmux attach-session -t monitor
```

## 📚 추가 리소스

### 유용한 명령어 모음
```bash
# 모델 크기 확인
python3 -c "from models.baseline_ssm import BaselineSSM; m=BaselineSSM(768,12,50257,16,4); print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')"

# 설정 파일 검증
python3 -c "import yaml; print(yaml.safe_load(open('configs/unified_config.yaml')))"

# 실험 결과 요약
ls -la experiments/*/results.json
```

### 디버깅 도구
```bash
# Python 디버거 사용
python3 -m pdb main.py --config configs/dry_run_config.yaml --mode full_pipeline

# 메모리 프로파일링
python3 -m memory_profiler main.py --config configs/dry_run_config.yaml --mode full_pipeline

# GPU 메모리 프로파일링
python3 -c "import torch; print(torch.cuda.memory_summary())"
```

---

## 🎉 완료!

Ubuntu 환경 설정이 완료되었습니다. 다음 단계로 진행하세요:

1. **Dry Run 실행**: `./ubuntu_dry_run.sh`
2. **전체 파이프라인**: `python3 main.py --config configs/unified_config.yaml --mode full_pipeline`
3. **결과 확인**: `cat ubuntu_dry_run_report.txt`

문제가 발생하면 위의 문제 해결 섹션을 참조하거나 로그를 확인하세요! 🐧✨ 