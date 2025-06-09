# extract_hidden_states.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def extract_hidden_states():
    print("🚀 [Step 1] Hidden State Extraction 시작...")
    
    # 모델 로딩
    model_id = "state-spaces/mamba-130m"
    print(f"📥 모델 로딩: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True
    )
    
    # Mamba는 GPTNeoX tokenizer 사용
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()

    # WikiText-2 소규모 샘플 로딩
    print("📚 WikiText-2 데이터셋 로딩...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    hidden_states_list = []

    def capture_hook(module, input, output):
        """hidden states를 캡처하는 훅 함수"""
        # input[0]은 hidden_states (shape: [batch_size, seq_len, hidden_size])
        hidden_states_list.append(input[0].detach().cpu())

    # 마지막 레이어의 mixer 입력을 훅으로 캡처
    # Mamba 구조: model.backbone.layers[-1].mixer
    target_layer = model.backbone.layers[-1].mixer
    hook = target_layer.register_forward_hook(capture_hook)
    
    print("🎯 훅 등록 완료: 마지막 레이어 mixer 입력 캡처")

    # 최대 10개 문장만 처리
    processed_count = 0
    for i, item in enumerate(dataset):
        if processed_count >= 10:
            break
            
        text = item["text"].strip()
        if len(text) < 10:  # 너무 짧은 텍스트 스킵
            continue
            
        print(f"🔄 처리 중: {processed_count + 1}/10 - '{text[:50]}...'")
        
        try:
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256,  # 메모리 절약을 위해 축소
                padding=True
            )
            
            with torch.no_grad():
                _ = model(**inputs)
                
            processed_count += 1
            
        except Exception as e:
            print(f"⚠️ 텍스트 처리 실패: {e}")
            continue

    hook.remove()
    print("🔌 훅 제거 완료")

    if not hidden_states_list:
        raise ValueError("❌ Hidden states가 캡처되지 않았습니다!")

    # 모든 hidden state를 [총_토큰수, hidden_size]로 결합
    all_h_list = []
    for h in hidden_states_list:
        # h shape: [batch_size, seq_len, hidden_size]
        # -> [seq_len, hidden_size]
        all_h_list.append(h.squeeze(0))
    
    all_h = torch.cat(all_h_list, dim=0)  # [total_tokens, hidden_size]
    print(f"✅ [INFO] Total hidden states: {all_h.shape}")

    # 저장
    torch.save(all_h, "hidden_states.pt")
    print(f"💾 저장 완료: hidden_states.pt ({all_h.shape})")
    
    return all_h.shape

if __name__ == "__main__":
    extract_hidden_states() 