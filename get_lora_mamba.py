# get_lora_mamba.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# 1. 기본 모델 로드
base_model_id = "state-spaces/mamba-130m"
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True
)
# Mamba 모델은 GPTNeoX tokenizer 사용
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# 2. LoRA 대상 모듈 지정 (SSM 관련 linear layer만 선택)
target_modules = [
    "mixer.in_proj",   # input -> hidden + gate
    "mixer.x_proj",    # hidden -> (time, B, C)
    "mixer.dt_proj",   # time step projection
    "mixer.out_proj",  # final output
]

# 3. LoRA 구성 설정
lora_config = LoraConfig(
    r=8,                          # low-rank dimension
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 4. LoRA 모델 생성
lora_model = get_peft_model(model, lora_config)

# 5. 학습 가능한 파라미터 수 확인
lora_model.print_trainable_parameters() 