import torch
from bin_int_convert import *

def roundAvg_conv_with_trace(wq_int, w_bitwidth: int=8, group_size: int=16, 
                             num_pruned_column: int=4, device='cpu'):
    """
    Round-Avg 알고리즘: 프루닝되는 하위 비트들을 
    그룹 평균값으로 Rounding하여 채워넣는 과정을 추적합니다.
    """
    # 0. 입력이 리스트인 경우 처리
    if isinstance(wq_int, list):
        wq_int = torch.tensor(wq_int, dtype=torch.float32).reshape(1, len(wq_int), 1, 1)

    wq_int = wq_int.to(device)
    K, C, H, W = wq_int.size()
    if C < group_size: group_size = C
    NUM_GROUP = K*W*H*C // group_size
    
    # [시작 헤더]
    print(f" 입력 데이터: {wq_int.view(-1)[:group_size].tolist()}")
    print(f" 목표 희소성: {num_pruned_column} columns (2's Complement 방식)")


    # [1] 데이터 변환 및 2의 보수 변환
    wq_int_flat = wq_int.permute([0, 2, 3, 1]).reshape(NUM_GROUP, group_size)
    wqb_twosComplement = int_to_twosComplement(wq_int_flat, w_bitwidth=w_bitwidth, device=device)
    print(f"1. 2의 보수(2's Complement) 비트 평면 생성 완료")

    # [2] 프루닝 포인터 결정 (MSB가 동일한 비트열 보호)
    prune_until = torch.full([NUM_GROUP], w_bitwidth - num_pruned_column, device=device)
    eq_msb_column = torch.ones([NUM_GROUP], dtype=torch.bool, device=device)
    
    # MSB와 동일한 값인 비트들은 부호 확장이므로 프루닝 범위에서 제외(뒤로 미룸)
    for i in range(1, w_bitwidth - 4):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        prune_until[eq_msb_column] += 1
    
    print(f"2. 프루닝 시작 지점 결정: index {prune_until[0].item()} (0부터 시작)")

    # [3] Round-Avg 핵심 루프
    print(f"\n3. 평균값 기반 비트 대체(Round-Avg) 시작")
    
    # 추적용 변수
    column_test = torch.zeros_like(wqb_twosComplement, dtype=torch.float32, device=device)
    
    for prune_idx in range(w_bitwidth - num_pruned_column, w_bitwidth):
        mask_group = torch.eq(prune_until, prune_idx)
        if not mask_group.any(): continue
        
        mask_value = mask_group.unsqueeze(-1).expand(-1, group_size)
        
        # (1) 프루닝될 비트들을 추출
        column_test[prune_idx:, mask_value] = wqb_twosComplement[prune_idx:, mask_value]
        
        # (2) 해당 비트들의 정수값 계산
        value_test = binary_to_int(column_test[prune_idx:], w_bitwidth=w_bitwidth-prune_idx, device=device)
        
        # (3) 그룹별 평균 계산 및 반올림(Round)
        v_mean_orig = torch.mean(value_test, dim=-1)
        value_mean = torch.round(v_mean_orig)
        
        # 로그 출력
        print(f"   > Prune Index {prune_idx}:")
        print(f"     - 대상 값들의 합산용 정수: {value_test[mask_group].view(-1).tolist()}")
        print(f"     - 평균값: {v_mean_orig[mask_group].item():.4f} -> Rounding: {value_mean[mask_group].item()}")
        
        # (4) 평균값을 다시 비트로 변환하여 채워넣기
        value_mean_exp = value_mean.unsqueeze(-1).expand(-1, group_size)
        column_new = int_to_binary(value_mean_exp, w_bitwidth=w_bitwidth-prune_idx, device=device)
        wqb_twosComplement[prune_idx:, mask_value] = column_new[:, mask_value]

    # [4] 최종 복원
    wq_int_new = twosComplement_to_int(wqb_twosComplement, w_bitwidth=w_bitwidth, device=device)
    
    # 결과 요약
    mse = torch.mean((wq_int_new - wq_int_flat)**2).item()
    print(f"\n4. RA 결과")
    print(f"   - 최종 변환 데이터: {wq_int_new[0].tolist()}")
    print(f"   - 최종 평균 MSE: {mse:.4f}")
 

    # 원래 차원으로 복구
    return wq_int_new.reshape(K, H, W, C).permute(0, 3, 1, 2)

# --- 실행부 ---
if __name__ == "__main__":
    test_data = [18, 21, 23, 16]
    # RA 알고리즘 실행 (4개 비트 프루닝 가정)
    result = roundAvg_conv_with_trace(test_data, group_size=4, num_pruned_column=4)