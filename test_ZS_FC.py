import torch
from bin_int_convert import *

def zeroPointShifting_fc(wq_int, w_bitwidth=8, group_size=4, num_pruned_column=4, const_bitwidth=5, device='cpu'):
    """
    FC 레이어용 ZS 메인 함수: [K, C] 형상을 처리합니다.
    """
    wq_int = wq_int.to(device)
    K, C = wq_int.size() # 출력 노드, 입력 노드
    NUM_GROUP = (K * C) // group_size
    wq_int_flat = wq_int.reshape(NUM_GROUP, group_size)

    # 오프셋 탐색 범위 설정
    offset_min = -2**int(const_bitwidth-1)
    offset_max = 2**int(const_bitwidth-1)

    best_error = torch.full([NUM_GROUP], 1e7, device=device)
    final_wq = torch.zeros_like(wq_int_flat)

    # 최적의 오프셋 탐색
    for offset in range(offset_min, offset_max):
        shifted_wq = wq_int_flat + float(offset)
        
        # Sign-Magnitude 변환 및 프루닝 시뮬레이션
        wqb_sm = int_to_signMagnitude(shifted_wq, w_bitwidth=w_bitwidth, device=device)
        
        # 상위 비트 프루닝 (부호 비트 제외 index 1부터 num_pruned_column만큼)
        wqb_pruned = wqb_sm.clone()
        wqb_pruned[1 : 1 + num_pruned_column] = 0 
        
        # 복원 및 오차 계산
        restored_shifted = signMagnitude_to_int(wqb_pruned, w_bitwidth=w_bitwidth, device=device)
        restored_orig = restored_shifted - float(offset)
        
        current_error = torch.sum((restored_orig - wq_int_flat)**2, dim=-1)

        # 최소 오차 그룹 업데이트
        mask = current_error < best_error
        best_error[mask] = current_error[mask]
        final_wq[mask] = restored_orig[mask]

    return final_wq.reshape(K, C)

def run_zs_debug(input_list, num_pruned_column=4, const_bitwidth=5, w_bitwidth=8):
    """
    @param input_list: 테스트할 4개의 정수 리스트 [n1, n2, n3, n4]
    @param num_pruned_column: 목표로 하는 비트 희소성(제거할 열 개수)
    @param const_bitwidth: 시프트 상수 C_bbs를 표현할 비트 수 (탐색 범위 결정)
    """
    device = 'cpu'
    wq_int = torch.tensor(input_list, dtype=torch.float32).to(device)
    group_size = len(input_list)
    
    print(f" 입력 데이터: {input_list}")
    print(f" 목표 희소성: {num_pruned_column} columns | 시프트 비트: {const_bitwidth}-bit")

    # 1. 시프트 상수(C_bbs) 탐색 범위 설정
    offset_min = -2**int(const_bitwidth-1)
    offset_max = 2**int(const_bitwidth-1)
    print(f"1. 탐색 범위 설정: {offset_min} ~ {offset_max-1} (총 {offset_max-offset_min}개 후보)")

    best_error = 1e7
    best_offset = 0
    best_wq_final = None

    # 2. 모든 오프셋 후보에 대해 테스트 (이미지의 루프 과정)
    print(f"\n2. 최적의 Zero-point 시프트 탐색 시작")
    
    for offset in range(offset_min, offset_max):
        # (1) 가중치 시프트 (w + C)
        shifted_wq = wq_int + float(offset)
        
        # (2) 비트 평면 변환 (Sign-Magnitude 방식 - 논문 권장)
        # ZS는 상위 비트 0을 찾기 위해 부호-절대값 방식을 주로 사용합니다.
        wqb_sm = int_to_signMagnitude(shifted_wq, w_bitwidth=w_bitwidth, device=device)
        
        # (3) 비트 프루닝 시뮬레이션
        # 목표 열(num_pruned_column)만큼을 0으로 만들었을 때의 값 계산
        # 여기서는 단순화를 위해 하위 비트를 날린 후 복원한 값과 비교합니다.
        # prune_idx = w_bitwidth - num_pruned_column
        # wqb_pruned = wqb_sm.clone()
        # wqb_pruned[prune_idx:] = 0 # 목표 열만큼 0으로 강제 (희소성 생성)
        
        prune_idx = num_pruned_column + 1 
        wqb_pruned = wqb_sm.clone()
        wqb_pruned[1:prune_idx] = 0
        # (4) 복원 및 오차 계산 (이미지의 Error check)
        restored_shifted_wq = signMagnitude_to_int(wqb_pruned, w_bitwidth=w_bitwidth, device=device)
        restored_original_wq = restored_shifted_wq - float(offset) # 시프트 복구 (w' - C)
        
        current_error = torch.sum((restored_original_wq - wq_int)**2).item()
        
        # 특정 오프셋들에 대한 로그 출력 (전부 다 찍으면 너무 많으므로 샘플링)
        if offset in [offset_min, 0, offset_max-1] or (current_error < best_error):
            
            print(f"   > Offset {offset:3d}: 복원값 {restored_original_wq.tolist()} | MSE: {current_error/group_size:.4f}")

        # 최소 오차 업데이트
        if current_error < best_error:
            best_error = current_error
            best_offset = offset
            best_wq_final = restored_original_wq

    # 3. 최종 결과 출력
    print(f"\n3. ZS 결과")
    print(f"   - 최적 오프셋(C_bbs): {best_offset}")
    print(f"   - 최종 변환 데이터: {best_wq_final.tolist()}")
    print(f"   - 최종 평균 MSE: {best_error/group_size:.4f}")


if __name__ == "__main__":
    run_zs_debug([18, 21, 23, 16], num_pruned_column=4)
    
