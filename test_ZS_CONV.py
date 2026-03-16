import torch
from bin_int_convert import *

# 1. 메인 함수: CONV 전용 (4차원 텐서 처리)
def zeroPointShifting_conv_trace(wq_int, w_bitwidth=8, group_size=4, num_pruned_column=4, const_bitwidth=5, device='cpu'):
    """
    CONV 레이어용 ZS 메인 함수: [K, C, H, W] 형상을 처리합니다.
    """
    wq_int = wq_int.to(device)
    K, C, H, W = wq_int.size()
    NUM_GROUP = (K * W * H * C) // group_size
    
    # 4차원 -> 2차원 평면으로 변환 (채널 방향 정렬)
    wq_int_flat = wq_int.permute([0, 2, 3, 1]).reshape(NUM_GROUP, group_size)

    # 1. 탐색 범위 설정
    offset_min = -2**int(const_bitwidth-1)
    offset_max = 2**int(const_bitwidth-1)
    print(f"STEP 1. 오프셋 탐색 범위: {offset_min} ~ {offset_max-1}")

    best_error = torch.full([NUM_GROUP], 1e7, device=device)
    final_wq_flat = torch.zeros_like(wq_int_flat)

    # 2. 최적의 시프트 상수(C) 탐색
    print(f"STEP 2. 최적 오프셋 탐색 시작...")
    for offset in range(offset_min, offset_max):
        shifted_wq = wq_int_flat + float(offset)
        
        # Sign-Magnitude 변환
        wqb_sm = int_to_signMagnitude(shifted_wq, w_bitwidth=w_bitwidth, device=device)
        
        # [핵심] ZS 프루닝 시뮬레이션: 상위 비트를 0으로 날렸을 때의 값 계산
        wqb_pruned = wqb_sm.clone()
        # 부호 비트(0번) 제외, 설정된 컬럼 수만큼 0으로 강제
        wqb_pruned[1 : 1 + num_pruned_column] = 0 
        
        # 복원 및 시프트 복구
        restored_shifted = signMagnitude_to_int(wqb_pruned, w_bitwidth=w_bitwidth, device=device)
        restored_orig = restored_shifted - float(offset)
        
        # 그룹별 MSE 계산
        current_error = torch.mean((restored_orig - wq_int_flat)**2, dim=-1)

        # 최소 오차 업데이트
        mask = current_error < best_error
        best_error[mask] = current_error[mask]
        final_wq_flat[mask] = restored_orig[mask]

    # 3. 2차원 -> 4차원 원래 모양으로 복구
    wq_int_new = final_wq_flat.reshape(K, H, W, C).permute(0, 3, 1, 2)
    print(f"STEP 3. ZS 완료 | 최종 평균 MSE: {torch.mean(best_error).item():.6f}")
    
    return wq_int_new

# 2. 실행부: CONV 가중치 디버그 함수
def run_zs_conv_debug(input_tensor, num_pruned_column=4, const_bitwidth=5):
    """
    CONV 가중치를 입력받아 ZS 메인 함수를 호출하고 결과를 출력합니다.
    """
    device = 'cpu'
    if isinstance(input_tensor, list):
        # 입력을 4차원 [K, C, H, W]로 변환
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).reshape(1, len(input_tensor), 1, 1)
    
    print(f"\n[CONV ZS Debug] 입력 데이터: {input_tensor.view(-1).tolist()}")

    # 메인 함수 호출 (이름 일치: zeroPointShifting_conv_trace)
    result_conv = zeroPointShifting_conv_trace(
        input_tensor, 
        group_size=input_tensor.size(1), 
        num_pruned_column=num_pruned_column, 
        const_bitwidth=const_bitwidth,
        device=device
    )
    
    print(f"결과 복원 데이터: {result_conv.view(-1).tolist()}")
    return result_conv

# --- 실행부 ---
if __name__ == "__main__":
    # 테스트 데이터: [18, 21, 23, 16]
    test_conv_data = [18, 21, 23, 16] 
    run_zs_conv_debug(test_conv_data, num_pruned_column=4)