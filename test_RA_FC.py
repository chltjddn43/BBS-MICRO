import torch
from bin_int_convert import *

def roundAvg_fc_with_trace(wq_int, w_bitwidth: int=8, group_size: int=16, 
                            num_pruned_column: int=4, device='cpu'):
    """
    FC(Fully Connected) 레이어 전용 Round-Avg 알고리즘 추적 함수입니다.
    2차원 행렬 [Output_Nodes, Input_Nodes] 구조를 처리합니다.
    """
    # 0. 입력 처리: 리스트면 2차원 텐서로, 텐서면 모양 확인
    if isinstance(wq_int, list):
        wq_int = torch.tensor(wq_int, dtype=torch.float32).reshape(1, -1) # [1, N] 행렬로 변환

    wq_int = wq_int.to(device)
    K, C = wq_int.size() # K: 출력 노드, C: 입력 노드
    
    if C < group_size: group_size = C
    NUM_GROUP = (K * C) // group_size
    
    # [시작 헤더]
    print(f"\n[FC RA Trace] 입력 데이터(첫 그룹): {wq_int.view(-1)[:group_size].tolist()}")
    print(f"설정: {w_bitwidth}-bit, {num_pruned_column} columns 프루닝, 그룹크기 {group_size}")

    # [1] 데이터 변환: 2차원을 그룹 단위로 reshape (permute 필요 없음)
    wq_int_flat = wq_int.reshape(NUM_GROUP, group_size)
    wqb_twosComplement = int_to_twosComplement(wq_int_flat, w_bitwidth=w_bitwidth, device=device)
    print(f"1. 2의 보수 변환 완료 (Shape: {wqb_twosComplement.shape})")

    # [2] 프루닝 포인터 결정
    prune_until = torch.full([NUM_GROUP], w_bitwidth - num_pruned_column, device=device)
    eq_msb_column = torch.ones([NUM_GROUP], dtype=torch.bool, device=device)
    
    for i in range(1, w_bitwidth - 4):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        prune_until[eq_msb_column] += 1
    
    print(f"2. 프루닝 시작 지점: Index {prune_until[0].item()}")

    # [3] Round-Avg 핵심 루프
    print(f"\n3. RA 연산 시작")
    column_test = torch.zeros_like(wqb_twosComplement, dtype=torch.float32, device=device)
    
    for prune_idx in range(w_bitwidth - num_pruned_column, w_bitwidth):
        mask_group = torch.eq(prune_until, prune_idx)
        if not mask_group.any(): continue
        
        mask_value = mask_group.unsqueeze(-1).expand(-1, group_size)
        
        # 하위 비트 추출 및 정수화
        column_test[prune_idx:, mask_value] = wqb_twosComplement[prune_idx:, mask_value]
        value_test = binary_to_int(column_test[prune_idx:], w_bitwidth=w_bitwidth-prune_idx, device=device)
        
        # 평균 및 반올림
        v_mean_orig = torch.mean(value_test, dim=-1)
        value_mean = torch.round(v_mean_orig)
        
        print(f"   > Bit Index {prune_idx} | 평균: {v_mean_orig[mask_group][0].item():.2f} -> Round: {value_mean[mask_group][0].item()}")
        
        # 평균값 비트로 대체
        value_mean_exp = value_mean.unsqueeze(-1).expand(-1, group_size)
        column_new = int_to_binary(value_mean_exp, w_bitwidth=w_bitwidth-prune_idx, device=device)
        wqb_twosComplement[prune_idx:, mask_value] = column_new[:, mask_value]

    # [4] 결과 복원
    wq_int_new = twosComplement_to_int(wqb_twosComplement, w_bitwidth=w_bitwidth, device=device)
    mse = torch.mean((wq_int_new - wq_int_flat)**2).item()
    
    print(f"\n4. 최종 결과")
    print(f"   - 변환 데이터: {wq_int_new[0].tolist()}")
    print(f"   - MSE: {mse:.4f}")

    # 원래 2차원 모양 [K, C]로 복구
    return wq_int_new.reshape(K, C)

# --- 실행부 ---
if __name__ == "__main__":
    test_data = [18, 21, 23, 16]
    # FC 방식으로 실행
    result = roundAvg_fc_with_trace(test_data, group_size=4, num_pruned_column=4)