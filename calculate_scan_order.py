# calculate_scan_order.py
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os

def calculate_scan_order():
    print("🧠 [Step 2] Correlation Scan 순열 계산 시작...")
    
    # hidden states 로딩
    if not os.path.exists("hidden_states.pt"):
        raise FileNotFoundError("❌ hidden_states.pt 파일을 찾을 수 없습니다. extract_hidden_states.py를 먼저 실행하세요.")
    
    print("📥 Hidden states 로딩...")
    hidden_states = torch.load("hidden_states.pt")  # shape: [L, D]
    print(f"📊 원본 크기: {hidden_states.shape}")
    
    # 메모리 절약을 위해 크기 제한
    max_tokens = 512  # 메모리 한계 고려
    if hidden_states.shape[0] > max_tokens:
        print(f"⚡ 메모리 절약을 위해 {max_tokens}개 토큰으로 축소")
        # 균등 샘플링
        indices = np.linspace(0, hidden_states.shape[0]-1, max_tokens, dtype=int)
        hidden_states = hidden_states[indices]
    
    print(f"🎯 최종 크기: {hidden_states.shape}")

    # Pearson 상관계수 기반 거리 행렬 생성
    print("🔗 상관계수 행렬 계산 중...")
    X = hidden_states.numpy()
    
    # 각 토큰을 정규화 (zero mean, unit variance)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    
    # 상관계수 행렬 계산
    corr = np.corrcoef(X)  # shape: [L, L]
    
    # NaN 처리
    corr = np.nan_to_num(corr, nan=0.0)
    
    # 거리 행렬: 높은 상관 → 짧은 거리
    dist = 1 - np.abs(corr)  # 절대값 사용으로 음의 상관도 고려
    
    print(f"📏 거리 행렬 생성 완료: {dist.shape}")
    print(f"📈 평균 거리: {dist.mean():.4f}, 표준편차: {dist.std():.4f}")

    # Nearest Neighbor 순회 (TSP heuristic)
    print("🚶 Nearest Neighbor TSP 순회 시작...")
    L = dist.shape[0]
    visited = [0]  # 첫 번째 토큰부터 시작
    unvisited = set(range(1, L))

    for step in range(L - 1):
        if step % 50 == 0:
            print(f"  진행률: {step+1}/{L-1} ({(step+1)/(L-1)*100:.1f}%)")
            
        last = visited[-1]
        # 가장 가까운(상관도 높은) 다음 토큰 선택
        next_idx = min(unvisited, key=lambda j: dist[last][j])
        visited.append(next_idx)
        unvisited.remove(next_idx)

    pi = np.array(visited)  # 순열
    pi_inv = np.argsort(pi)  # 역순열

    print(f"✅ 순열 계산 완료!")
    print(f"📊 순열 π: 길이 {len(pi)}, 범위 [{pi.min()}, {pi.max()}]")
    print(f"📊 역순열 π⁻¹: 길이 {len(pi_inv)}, 범위 [{pi_inv.min()}, {pi_inv.max()}]")

    # 순열 품질 평가
    original_total_dist = sum(dist[i, i+1] for i in range(L-1))  # 원본 순서
    optimized_total_dist = sum(dist[pi[i], pi[i+1]] for i in range(L-1))  # 최적화된 순서
    
    improvement = (original_total_dist - optimized_total_dist) / original_total_dist * 100
    print(f"🎯 순열 품질:")
    print(f"   원본 총 거리: {original_total_dist:.4f}")
    print(f"   최적화 총 거리: {optimized_total_dist:.4f}")
    print(f"   개선율: {improvement:.2f}%")

    # 저장
    np.save("scan_order.npy", pi)
    np.save("scan_order_inv.npy", pi_inv)
    
    print("💾 저장 완료:")
    print(f"   📄 scan_order.npy: {pi.shape}")
    print(f"   📄 scan_order_inv.npy: {pi_inv.shape}")
    
    # 순열 미리보기
    print(f"🔍 순열 미리보기 (처음 10개): {pi[:10]}")
    print(f"🔍 역순열 미리보기 (처음 10개): {pi_inv[:10]}")
    
    return {
        'permutation_length': len(pi),
        'improvement_rate': improvement,
        'total_distance_reduction': original_total_dist - optimized_total_dist
    }

if __name__ == "__main__":
    calculate_scan_order() 