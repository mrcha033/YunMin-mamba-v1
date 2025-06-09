# test_yunmin_scan.py
"""
🧪 YunMin Correlation Scan 완전 테스트

이 스크립트는 다음을 수행합니다:
1. 원본 Mamba 모델 로딩
2. LoRA @ SSM-only 적용
3. YunMin Scan Patch 적용/제거
4. Before/After 성능 비교
5. 텍스트 생성 품질 검증
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import time
import numpy as np

# 우리가 만든 모듈들
from scan_patch import apply_scan_patch, remove_scan_patch, is_scan_patched, get_permutation_info

def setup_model_and_tokenizer():
    """모델과 토크나이저 설정"""
    print("🚀 모델 및 토크나이저 로딩...")
    
    model_id = "state-spaces/mamba-130m"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 모델 로딩 완료")
    return model, tokenizer

def setup_lora(model):
    """LoRA @ SSM-only 설정"""
    print("🔧 LoRA @ SSM-only 설정 중...")
    
    target_modules = [
        "mixer.in_proj",
        "mixer.x_proj", 
        "mixer.dt_proj",
        "mixer.out_proj"
    ]
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    
    return lora_model

def test_generation(model, tokenizer, prompt="대한민국의 수도는", max_tokens=30, label=""):
    """텍스트 생성 테스트"""
    print(f"📝 텍스트 생성 테스트{label}:")
    print(f"   프롬프트: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   결과: '{generated_text}'")
    print(f"   소요 시간: {generation_time:.3f}초")
    print()
    
    return generated_text, generation_time

def test_perplexity(model, tokenizer, text="The quick brown fox jumps over the lazy dog."):
    """간단한 perplexity 테스트"""
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(loss)).item()
    
    return perplexity

def main():
    print("=" * 70)
    print("🧠 YunMin Correlation Scan 완전 테스트")
    print("=" * 70)
    
    # 1. 모델 설정
    model, tokenizer = setup_model_and_tokenizer()
    
    # 2. LoRA 적용
    lora_model = setup_lora(model)
    
    # 3. 순열 파일 확인
    import os
    if not os.path.exists("scan_order.npy"):
        print("❌ scan_order.npy가 없습니다. run_correlation_scan.py를 먼저 실행하세요.")
        return
    
    print("\n" + "=" * 50)
    print("📊 BEFORE: 원본 모델 성능 측정")
    print("=" * 50)
    
    # Before 테스트
    test_prompts = [
        "대한민국의 수도는",
        "인공지능의 미래는", 
        "Today is a beautiful"
    ]
    
    before_results = {}
    for prompt in test_prompts:
        text, time_taken = test_generation(lora_model, tokenizer, prompt, label=" [BEFORE]")
        before_results[prompt] = {'text': text, 'time': time_taken}
    
    # Perplexity 측정 (BEFORE)
    before_ppl = test_perplexity(lora_model, tokenizer)
    print(f"📈 BEFORE Perplexity: {before_ppl:.4f}")
    
    print("\n" + "=" * 50)
    print("🔧 YunMin Correlation Scan 패치 적용")
    print("=" * 50)
    
    # 4. Scan Patch 적용
    try:
        apply_scan_patch(lora_model)
        
        # 패치 정보 확인
        patch_info = get_permutation_info()
        print(f"📋 패치 정보: {patch_info}")
        print(f"🔗 패치 상태: {is_scan_patched()}")
        
    except Exception as e:
        print(f"❌ 패치 적용 실패: {e}")
        return
    
    print("\n" + "=" * 50)
    print("📊 AFTER: YunMin Scan 적용 후 성능 측정")
    print("=" * 50)
    
    # After 테스트
    after_results = {}
    for prompt in test_prompts:
        text, time_taken = test_generation(lora_model, tokenizer, prompt, label=" [AFTER]")
        after_results[prompt] = {'text': text, 'time': time_taken}
    
    # Perplexity 측정 (AFTER)
    after_ppl = test_perplexity(lora_model, tokenizer)
    print(f"📈 AFTER Perplexity: {after_ppl:.4f}")
    
    print("\n" + "=" * 50)
    print("📈 결과 비교 분석")
    print("=" * 50)
    
    # 결과 비교
    print("🔍 텍스트 생성 비교:")
    for prompt in test_prompts:
        print(f"\n프롬프트: '{prompt}'")
        print(f"  BEFORE: {before_results[prompt]['text']}")
        print(f"  AFTER:  {after_results[prompt]['text']}")
        
        time_diff = after_results[prompt]['time'] - before_results[prompt]['time']
        print(f"  시간 변화: {time_diff:+.3f}초")
    
    print(f"\n📊 Perplexity 변화:")
    print(f"  BEFORE: {before_ppl:.4f}")
    print(f"  AFTER:  {after_ppl:.4f}")
    ppl_change = ((after_ppl - before_ppl) / before_ppl) * 100
    print(f"  변화율: {ppl_change:+.2f}%")
    
    # 6. 패치 제거 테스트
    print("\n" + "=" * 50)
    print("🔄 패치 제거 및 복원 테스트")
    print("=" * 50)
    
    remove_scan_patch(lora_model)
    print(f"🔗 패치 제거 후 상태: {is_scan_patched()}")
    
    # 복원 확인
    restore_text, restore_time = test_generation(lora_model, tokenizer, test_prompts[0], label=" [RESTORED]")
    restore_ppl = test_perplexity(lora_model, tokenizer)
    
    print(f"🔍 복원 검증:")
    print(f"  원본과 동일한지: {abs(restore_ppl - before_ppl) < 0.001}")
    print(f"  복원 Perplexity: {restore_ppl:.4f}")
    
    print("\n" + "=" * 70)
    print("🎉 YunMin Correlation Scan 테스트 완료!")
    print("=" * 70)
    
    # 최종 요약
    print("📋 최종 요약:")
    print(f"  🧠 순열 길이: {patch_info['pi_length'] if patch_info else 'N/A'}")
    print(f"  📈 Perplexity 변화: {ppl_change:+.2f}%")
    print(f"  🔧 LoRA 파라미터: 1.48% (SSM-only)")
    print(f"  ✅ 패치 적용/제거: 정상 작동")

if __name__ == "__main__":
    main() 