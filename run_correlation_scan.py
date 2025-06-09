# run_correlation_scan.py
"""
🧠 YunMin Correlation Scan 전체 파이프라인

이 스크립트는 다음 작업을 자동으로 수행합니다:
1. Mamba 모델에서 hidden states 추출
2. 상관계수 기반 최적 순열 계산
3. 결과 파일 생성 및 검증
"""

import os
import time
from extract_hidden_states import extract_hidden_states
from calculate_scan_order import calculate_scan_order

def main():
    print("=" * 60)
    print("🧠 YunMin Correlation Scan Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Hidden States 추출
        print("\n🚀 STEP 1: Hidden States 추출")
        print("-" * 40)
        
        if os.path.exists("hidden_states.pt"):
            print("📄 hidden_states.pt가 이미 존재합니다.")
            overwrite = input("다시 추출하시겠습니까? (y/N): ").lower().strip()
            if overwrite == 'y':
                os.remove("hidden_states.pt")
                hidden_shape = extract_hidden_states()
            else:
                import torch
                hidden_states = torch.load("hidden_states.pt")
                hidden_shape = hidden_states.shape
                print(f"✅ 기존 파일 사용: {hidden_shape}")
        else:
            hidden_shape = extract_hidden_states()
            
        print(f"✅ Step 1 완료: {hidden_shape}")
        
        # Step 2: 순열 계산
        print("\n🧠 STEP 2: Correlation Scan 순열 계산")
        print("-" * 40)
        
        scan_results = calculate_scan_order()
        print(f"✅ Step 2 완료")
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 PIPELINE 완료 - 결과 요약")
        print("=" * 60)
        
        elapsed_time = time.time() - start_time
        
        print(f"⏱️  총 실행 시간: {elapsed_time:.2f}초")
        print(f"📊 Hidden States: {hidden_shape}")
        print(f"📊 순열 길이: {scan_results['permutation_length']}")
        print(f"📈 개선율: {scan_results['improvement_rate']:.2f}%")
        
        # 생성된 파일 확인
        print(f"\n📂 생성된 파일:")
        files = ["hidden_states.pt", "scan_order.npy", "scan_order_inv.npy"]
        for file in files:
            if os.path.exists(file):
                size_mb = os.path.getsize(file) / 1024 / 1024
                print(f"   ✅ {file} ({size_mb:.2f}MB)")
            else:
                print(f"   ❌ {file} (생성되지 않음)")
        
        print(f"\n🎉 YunMin Correlation Scan이 성공적으로 완료되었습니다!")
        print(f"   다음 단계: Mamba 모델에 순열 로직 삽입")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print(f"📞 문제 해결이 필요합니다.")
        raise

if __name__ == "__main__":
    main() 