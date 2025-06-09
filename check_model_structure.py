# check_model_structure.py
from transformers import AutoModelForCausalLM
import torch

# 모델 로드
base_model_id = "state-spaces/mamba-130m"
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True
)

print("🔍 모델 구조 내 모든 Linear 레이어 확인:")
print("=" * 60)

linear_modules = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_modules.append(name)
        print(f"📍 {name}")

print("\n🎯 SSM 관련 Linear 레이어 (mixer 포함):")
print("=" * 60)
ssm_modules = [name for name in linear_modules if "mixer" in name]
for name in ssm_modules:
    print(f"✅ {name}")

print(f"\n📊 전체 Linear 레이어 수: {len(linear_modules)}")
print(f"📊 SSM Linear 레이어 수: {len(ssm_modules)}")

# target_modules 검증
target_modules = [
    "mixer.in_proj",
    "mixer.x_proj", 
    "mixer.dt_proj",
    "mixer.out_proj"
]

print(f"\n🧪 target_modules 검증:")
print("=" * 60)
for target in target_modules:
    found = any(target in name for name in linear_modules)
    status = "✅ FOUND" if found else "❌ NOT FOUND"
    print(f"{status}: {target}")

print(f"\n📋 실제 존재하는 정확한 모듈명:")
print("=" * 60)
for target in target_modules:
    matches = [name for name in linear_modules if target.split('.')[-1] in name and "mixer" in name]
    if matches:
        for match in matches:
            print(f"🎯 {target} → {match}")
    else:
        print(f"❌ {target} → NOT FOUND") 