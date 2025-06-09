# scan_patch.py
"""
🧠 YunMin Correlation Scan 패치 모듈

이 모듈은 사전 계산된 순열(π)과 역순열(π⁻¹)을 Mamba 모델에 적용하여
Correlation Scan 기법을 실현합니다.

사용법:
    from scan_patch import apply_scan_patch, remove_scan_patch
    
    # 순열 적용
    apply_scan_patch(model)
    
    # 순열 제거 (원본 복원)
    remove_scan_patch(model)
"""

import torch
import numpy as np
import os
from typing import Optional, Dict, Any

class ScanPatcher:
    def __init__(self):
        self.original_forwards = {}  # 원본 forward 함수 백업
        self.is_patched = False
        self.pi = None
        self.pi_inv = None
        
    def load_permutations(self, scan_path="scan_order.npy", reverse_path="scan_order_inv.npy"):
        """순열 파일들을 로딩"""
        if not os.path.exists(scan_path):
            raise FileNotFoundError(f"❌ 순열 파일을 찾을 수 없습니다: {scan_path}")
        if not os.path.exists(reverse_path):
            raise FileNotFoundError(f"❌ 역순열 파일을 찾을 수 없습니다: {reverse_path}")
            
        pi_np = np.load(scan_path)
        pi_inv_np = np.load(reverse_path)
        
        self.pi = torch.tensor(pi_np, dtype=torch.long)
        self.pi_inv = torch.tensor(pi_inv_np, dtype=torch.long)
        
        print(f"✅ [SCAN PATCH] 순열 로딩 완료:")
        print(f"   π.shape: {self.pi.shape}")
        print(f"   π⁻¹.shape: {self.pi_inv.shape}")
        print(f"   순열 미리보기: {self.pi[:10].tolist()}")
        
    def _create_patched_forward(self, original_forward, layer_idx):
        """패치된 forward 함수 생성"""
        def patched_forward(hidden_states, *args, **kwargs):
            batch_size, seq_len, hidden_size = hidden_states.shape
            device = hidden_states.device
            
            # 디바이스 이동
            pi_device = self.pi.to(device)
            pi_inv_device = self.pi_inv.to(device)
            
            # 시퀀스 길이 검증 및 조정
            perm_len = len(pi_device)
            
            if seq_len <= perm_len:
                # 시퀀스가 순열보다 같거나 짧은 경우: 유효한 인덱스만 사용
                valid_mask = pi_device < seq_len
                pi_filtered = pi_device[valid_mask]
                
                if len(pi_filtered) == 0:
                    # 유효한 인덱스가 없으면 순열 적용하지 않음
                    hidden_states_permuted = hidden_states
                    pi_inv_for_output = torch.arange(seq_len, device=device)
                else:
                    # 순열 적용
                    hidden_states_permuted = hidden_states[:, pi_filtered, :]
                    pi_inv_for_output = torch.argsort(pi_filtered)
                    
            else:
                # 시퀀스가 순열보다 긴 경우: 순열에 없는 부분은 그대로 추가
                remaining_indices = torch.arange(perm_len, seq_len, device=device)
                pi_extended = torch.cat([pi_device, remaining_indices])
                pi_inv_for_output = torch.argsort(pi_extended)
                
                hidden_states_permuted = hidden_states[:, pi_extended, :]
            
            # 원본 forward 실행
            output = original_forward(hidden_states_permuted, *args, **kwargs)
            
            # 출력 역순열 적용 (크기가 일치하는 경우에만)
            if output.shape[1] == hidden_states_permuted.shape[1] and len(pi_inv_for_output) == output.shape[1]:
                output_restored = output[:, pi_inv_for_output, :]
            else:
                # 출력 크기가 다른 경우 또는 인덱스 불일치시 그대로 반환
                output_restored = output
                
            return output_restored
            
        return patched_forward
    
    def apply_patch(self, model, target_layers="all"):
        """모델에 순열 패치 적용"""
        if self.pi is None or self.pi_inv is None:
            raise ValueError("❌ 순열이 로딩되지 않았습니다. load_permutations()를 먼저 호출하세요.")
            
        if self.is_patched:
            print("⚠️ 이미 패치가 적용되어 있습니다. 먼저 remove_patch()를 호출하세요.")
            return
            
        print(f"🔧 [SCAN PATCH] 순열 패치 적용 시작...")
        
        # Mamba 모델 구조 확인
        if hasattr(model, 'backbone'):
            layers = model.backbone.layers  # MambaForCausalLM
        elif hasattr(model, 'layers'):
            layers = model.layers  # MambaModel
        else:
            raise ValueError("❌ 지원되지 않는 모델 구조입니다.")
            
        patched_count = 0
        
        for i, layer in enumerate(layers):
            if target_layers != "all" and i not in target_layers:
                continue
                
            mixer = layer.mixer
            layer_id = f"layer_{i}_mixer"
            
            # 원본 forward 백업
            self.original_forwards[layer_id] = mixer.forward
            
            # 패치된 forward로 교체
            mixer.forward = self._create_patched_forward(mixer.forward, i)
            patched_count += 1
            
        self.is_patched = True
        print(f"✅ [SCAN PATCH] 패치 적용 완료: {patched_count}개 레이어")
        
    def remove_patch(self, model):
        """패치 제거하고 원본 복원"""
        if not self.is_patched:
            print("⚠️ 패치가 적용되지 않은 상태입니다.")
            return
            
        print(f"🔄 [SCAN PATCH] 패치 제거 중...")
        
        # Mamba 모델 구조 확인
        if hasattr(model, 'backbone'):
            layers = model.backbone.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise ValueError("❌ 지원되지 않는 모델 구조입니다.")
            
        restored_count = 0
        
        for i, layer in enumerate(layers):
            layer_id = f"layer_{i}_mixer"
            if layer_id in self.original_forwards:
                layer.mixer.forward = self.original_forwards[layer_id]
                restored_count += 1
                
        self.original_forwards.clear()
        self.is_patched = False
        print(f"✅ [SCAN PATCH] 패치 제거 완료: {restored_count}개 레이어 복원")

# 전역 패처 인스턴스
_global_patcher = ScanPatcher()

def apply_scan_patch(model, scan_path="scan_order.npy", reverse_path="scan_order_inv.npy", target_layers="all"):
    """
    모델에 Correlation Scan 패치 적용
    
    Args:
        model: Mamba 모델 인스턴스
        scan_path: 순열 파일 경로
        reverse_path: 역순열 파일 경로
        target_layers: 패치를 적용할 레이어 ("all" 또는 레이어 인덱스 리스트)
    """
    _global_patcher.load_permutations(scan_path, reverse_path)
    _global_patcher.apply_patch(model, target_layers)

def remove_scan_patch(model):
    """모델에서 Correlation Scan 패치 제거"""
    _global_patcher.remove_patch(model)

def is_scan_patched():
    """현재 패치 적용 상태 확인"""
    return _global_patcher.is_patched

def get_permutation_info():
    """현재 로딩된 순열 정보 반환"""
    if _global_patcher.pi is None:
        return None
    return {
        'pi_length': len(_global_patcher.pi),
        'pi_preview': _global_patcher.pi[:10].tolist() if len(_global_patcher.pi) >= 10 else _global_patcher.pi.tolist()
    } 