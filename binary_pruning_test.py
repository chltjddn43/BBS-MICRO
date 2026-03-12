import torch
import torch.nn as nn
import torch.nn.functional as F
from bin_int_convert import *

'''
Definition of function parameters:
    @param wq_int: baseline quantized integer weight
    @param w_bitwidth: bit width of the baseline quantized integer weight
    @param group_size: group size for binary pruning
    @param num_pruned_column: number of desired bi-directional sparse bit-columns in every weight group
    @param const_bitwidth: bit width of the BBS constant for the Zero-Point Shifting algorithm
    @param device: 'cpu' or 'cuda'  
'''

def roundAvg_conv(wq_int, w_bitwidth: int=8, group_size: int=16, num_pruned_column: int=4, device='cpu'):
    print(f"\n--- [RA 절차 시작] ---")
    wq_int = wq_int.to(device)
    K, C, H, W = wq_int.size()
    
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    
    # 1. 차원 재구성 (Flattening)
    # [K, C, H, W] -> [그룹 수, 그룹 사이즈]로 변형하여 연산 준비
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)
    print(f"단계 1: 가중치 재구성 완료 | 형상: {wq_int.shape} (그룹 수: {NUM_GROUP}, 그룹 크기: {group_size})")

    # 2. 정수 -> 2의 보수(Binary) 변환
    # 각 정수가 8개의 비트 평면(Bit-planes)으로 펼쳐집니다.
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    print(f"단계 2: 비트 변환 완료 | 형상: {wqb_twosComplement.shape} (비트너비, 그룹수, 그룹크기)")
    # 예시: 첫 번째 그룹의 첫 번째 값의 비트 출력
    print(f"      샘플 비트(첫 값): {wqb_twosComplement[:, 0, 0].cpu().numpy()}")

    # 3. Pruning 범위 결정
    # MSB(최상위 비트)들이 같은 경우 더 많은 비트를 Pruning할 수 있는지 체크합니다.
    prune_until = torch.full([NUM_GROUP], w_bitwidth-num_pruned_column, device=device)
    eq_msb_column = torch.ones([NUM_GROUP], dtype=torch.bool, device=device)
    
    for i in range(1, w_bitwidth-4):
        # 모든 비트가 같은지 확인하여 prune_until 포인터를 이동
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        prune_until[eq_msb_column] += 1
    print(f"단계 3: 비트 Pruning 포인터 결정 완료 | 평균 포인터 위치: {prune_until.float().mean().item():.2f}")

    # 4. 반올림 평균화 (Core Logic)
    # 지정된 비트 컬럼들을 뽑아 정수로 바꾸고, 평균을 낸 뒤 다시 비트로 덮어씁니다.
    column_test = torch.zeros_like(wqb_twosComplement, dtype=torch.float32, device=device)
    
    print(f"단계 4: 비트 컬럼별 평균화 진행 중...")
    for prune_idx in range(w_bitwidth-num_pruned_column, w_bitwidth):
        mask_group = torch.eq(prune_until, prune_idx)
        mask_value = mask_group.unsqueeze(-1).expand(-1, group_size)
        
        # 특정 비트 위치의 값을 정수로 추출
        column_test[prune_idx:, mask_value] = wqb_twosComplement[prune_idx:, mask_value]
        value_test = binary_to_int(column_test[prune_idx:], w_bitwidth=w_bitwidth-prune_idx, device=device)
        
        # 그룹 내 평균 계산 및 반올림
        value_mean = torch.round(torch.mean(value_test, dim=-1))
        
        # 평균값으로 비트 재구성 및 덮어쓰기
        value_mean_exp = value_mean.unsqueeze(-1).expand(-1, group_size)
        column_new = int_to_binary(value_mean_exp, w_bitwidth=w_bitwidth-prune_idx, device=device)
        wqb_twosComplement[prune_idx:, mask_value] = column_new[:, mask_value]
        
        if mask_group.any():
            print(f"      - [{prune_idx}번 비트 컬럼] 평균화 적용 그룹 수: {mask_group.sum().item()}")

    # 5. 비트 -> 정수 복원 및 형상 복구
    wq_int_new = twosComplement_to_int(wqb_twosComplement, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.reshape(K, H, W, C).permute(0, 3, 1, 2)
    print(f"단계 5: 최종 정수 복원 완료 | 최종 형상: {wq_int_new.shape}")
    print(f"--- [RA 절차 종료] ---\n")

    return wq_int_new

def roundAvg_fc(wq_int, w_bitwidth: int=8, group_size: int=16, num_pruned_column: int=4, device='cpu'):
    print(f"\n--- [RA-FC 절차 시작] ---")
    wq_int = wq_int.to(device)
    
    # K: 출력 노드 수, C: 입력 노드 수
    K, C = wq_int.size() 
    
    if C < group_size:
        group_size = C
    
    # 전체 요소를 그룹 크기로 나누어 총 그룹 수를 계산
    NUM_GROUP = K * C // group_size
    
    # 1. 차원 재구성 (Flattening)
    # [K, C] 형상을 [그룹 수, 그룹 사이즈]로 변경하여 병렬 연산 준비
    wq_int = wq_int.reshape(NUM_GROUP, group_size)
    print(f"단계 1: 가중치 재구성 완료 | 형상: {wq_int.shape} (그룹 수: {NUM_GROUP}, 그룹 크기: {group_size})")

    # 2. 정수 -> 2의 보수(Binary) 비트 평면으로 변환
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    print(f"단계 2: 비트 변환 완료 | 형상: {wqb_twosComplement.shape} (비트너비, 그룹수, 그룹크기)")
    # 첫 번째 그룹의 첫 번째 가중치가 어떤 비트로 표현되었는지 확인
    print(f"      샘플 비트 패턴: {wqb_twosComplement[:, 0, 0].cpu().numpy()}")

    # 3. 비트 Pruning 시작점(Pointer) 결정
    # 기본적으로 (전체 비트 - 목표 희소 비트 수) 위치에서 시작
    prune_until = torch.full([NUM_GROUP], w_bitwidth - num_pruned_column, device=device)
    eq_msb_column = torch.ones([NUM_GROUP], dtype=torch.bool, device=device)
    
    # 상위 비트들이 일치하는 경우, 더 많은 비트를 Pruning할 수 있는지 탐색
    for i in range(1, w_bitwidth - 4):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        prune_until[eq_msb_column] += 1
    print(f"단계 3: 비트 Pruning 포인터 결정 완료 | 평균 포인터 위치: {prune_until.float().mean().item():.2f}")

    # 4. 반올림 평균화 (Rounded Averaging) 핵심 로직
    column_test = torch.zeros_like(wqb_twosComplement, dtype=torch.float32, device=device)
    
    print(f"단계 4: 비트 컬럼별 평균화(Rounding) 진행 중...")
    for prune_idx in range(w_bitwidth - num_pruned_column, w_bitwidth):
        # 현재 처리할 비트 위치에 해당하는 그룹들만 필터링(Masking)
        mask_group = torch.eq(prune_until, prune_idx)
        mask_value = mask_group.unsqueeze(-1).expand(-1, group_size)
        
        # 해당 비트 컬럼들의 값을 정수로 일시 변환
        column_test[prune_idx:, mask_value] = wqb_twosComplement[prune_idx:, mask_value]
        value_test = binary_to_int(column_test[prune_idx:], w_bitwidth=w_bitwidth-prune_idx, device=device)
        
        # 핵심: 그룹 내 가중치들의 평균을 구하고 반올림하여 '대표값' 생성
        value_mean = torch.round(torch.mean(value_test, dim=-1))
        
        # 생성된 대표값을 다시 비트로 변환하여 원래 자리에 덮어씌움 (희소성 강제 생성)
        value_mean_exp = value_mean.unsqueeze(-1).expand(-1, group_size)
        column_new = int_to_binary(value_mean_exp, w_bitwidth=w_bitwidth-prune_idx, device=device)
        wqb_twosComplement[prune_idx:, mask_value] = column_new[:, mask_value]
        
        if mask_group.any():
            print(f"      - [비트 {prune_idx} 위치] {mask_group.sum().item()}개 그룹 평균화 완료")

    # 5. 비트 -> 정수 복원 및 원래 형상으로 복구
    wq_int_new = twosComplement_to_int(wqb_twosComplement, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.reshape(K, C)
    
    print(f"단계 5: 최종 정수 복원 완료 | 결과 형상: {wq_int_new.shape}")
    print(f"--- [RA-FC 절차 종료] ---\n")

    return wq_int_new
import torch

def zeroPointShifting_conv(wq_int, w_bitwidth: int=8, group_size: int=16, 
                           num_pruned_column: int=4, const_bitwidth: int=5, device='cpu'):
    print(f"\n--- [ZS-Conv 절차 시작] ---")
    wq_int = wq_int.to(device)
    K, C, H, W = wq_int.size()
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    
    # 1. 차원 재구성 (Flattening)
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)
    print(f"단계 1: 데이터 재구성 완료 | 그룹 수: {NUM_GROUP}")

    # 2. 클리핑 및 오프셋(Offset) 후보군 설정
    v_max = 2.**(w_bitwidth-1) - 1
    v_min = -v_max
    offset_min = -2**int(const_bitwidth-1)
    offset_max = 2**int(const_bitwidth-1)
    offsets = list(range(offset_min, offset_max)) # 예: [-16, -15, ..., 15]
    rp_factor = len(offsets)
    print(f"단계 2: 오프셋 탐색 범위: {offset_min} ~ {offset_max-1} (총 {rp_factor}개 후보)")

    # 모든 오프셋 후보에 대해 병렬적으로 가중치를 시프팅함
    wq_int_rp = wq_int.unsqueeze(0).repeat(rp_factor, 1, 1)
    for i, offset in enumerate(offsets):
        wq_int_rp[i] = wq_int_rp[i] + float(offset)
    
    wq_int_rp[wq_int_rp.lt(v_min)] = v_min
    wq_int_rp[wq_int_rp.gt(v_max)] = v_max

    # 3. 비트 변환 (Sign-Magnitude 방식)
    # ZS는 부호-절대값 방식을 사용하여 MSB 0의 개수를 계산합니다.
    from binary_pruning import int_to_signMagnitude, binary_to_int, int_to_binary, signMagnitude_to_int
    wqb_signMagnitude = int_to_signMagnitude(wq_int_rp, w_bitwidth=w_bitwidth, device=device)
    print(f"단계 3: Sign-Magnitude 비트 변환 완료")

    # 4. 비트 Pruning/Test 포인터 계산
    prune_until = torch.full([rp_factor, NUM_GROUP], int(w_bitwidth-num_pruned_column), device=device)
    test_until  = torch.full([rp_factor, NUM_GROUP], 1, device=device)
    is_msb_zero = torch.ones([rp_factor, NUM_GROUP], dtype=torch.bool, device=device)
    
    for i in range(1, w_bitwidth):
        is_current_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
        is_msb_zero = torch.logical_and(is_msb_zero, is_current_zero)
        prune_until[is_msb_zero] +=  1
        test_until[is_msb_zero]  +=  1
    print(f"단계 4: 비트 구조 분석 완료")

    # 5. 최적 비트 패턴 탐색 (MSE 최소화 루프)
    value_test = torch.zeros_like(wq_int_rp, dtype=torch.float32, device=device)
    value_new = torch.zeros_like(wq_int_rp, dtype=torch.float32, device=device)

    for test_idx in range(1, w_bitwidth):
        mask = torch.eq(test_until, test_idx)
        column_test = wqb_signMagnitude[test_idx:, mask, :]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_idx, device=device)
        value_test[mask] = int_test
    
    print(f"단계 5: 최적 비트 탐색 중 (시간이 다소 소요될 수 있습니다)...")
    for prune_idx in range(w_bitwidth-num_pruned_column, w_bitwidth):
        for test_idx in range(1, prune_idx):
            mask_group = torch.logical_and(torch.eq(test_until, test_idx), torch.eq(prune_until, prune_idx))
            mask_group_exp = mask_group.unsqueeze(-1).expand(-1, -1, group_size)
            error = torch.full([rp_factor, NUM_GROUP, group_size], 1e7, device=device)
            
            for n in range(2**(prune_idx-test_idx)):
                tmp_value = n * 2.**(w_bitwidth-prune_idx)
                new_error = (tmp_value - value_test) ** 2
                mask_value = torch.logical_and(torch.lt(new_error, error), mask_group_exp)
                error[mask_value] = new_error[mask_value]
                value_new[mask_value] = tmp_value
            
            column_new = int_to_binary(value_new, w_bitwidth=w_bitwidth-test_idx, device=device)
            wqb_signMagnitude[test_idx:, mask_group_exp] = column_new[:, mask_group_exp]
        print(f"      - {prune_idx}비트 영역 최적화 완료")

    # 6. 복원 및 최적 상수(Offset) 결정
    wq_int_pruned = signMagnitude_to_int(wqb_signMagnitude, w_bitwidth=w_bitwidth, device=device)
    for i, offset in enumerate(offsets):
        wq_int_pruned[i] = wq_int_pruned[i] - float(offset)
    
    wq_int_original = wq_int.to(torch.float32)
    wq_int_new = torch.zeros_like(wq_int_original, dtype=torch.float32, device=device)
    
    error = torch.full([NUM_GROUP], 1e7, device=device)
    best_offsets = torch.zeros([NUM_GROUP], device=device) # 각 그룹별 최적 상수 저장

    print(f"단계 6: 모든 오프셋 비교 후 최적 상수(Constant) 선택 중...")
    for i, offset in enumerate(offsets):
        new_error = torch.sum((wq_int_pruned[i] - wq_int_original)**2, dim=-1)
        mask_value = torch.lt(new_error, error)
        error[mask_value] = new_error[mask_value]
        wq_int_new[mask_value] = wq_int_pruned[i][mask_value]
        best_offsets[mask_value] = float(offset) # 여기서 상수가 결정됩니다.

    print(f"      - 선택된 상수(Offset) 샘플 (첫 4개 그룹): {best_offsets[:4].cpu().numpy()}")
    print(f"      - 전체 그룹의 평균 오프셋: {best_offsets.mean().item():.4f}")
    
    wq_int_new = wq_int_new.reshape(K, H, W, C).permute(0, 3, 1, 2)
    print(f"--- [ZS-Conv 절차 종료] ---\n")
    return wq_int_new
def zeroPointShifting_fc(wq_int, w_bitwidth: int=8, group_size: int=16, 
                         num_pruned_column: int=4, const_bitwidth: int=5, device='cpu'):
    print(f"\n--- [ZS-FC 절차 시작] ---")
    wq_int = wq_int.to(device)
    K, C = wq_int.size()
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    
    # 1. 차원 재구성
    wq_int = wq_int.reshape(NUM_GROUP, group_size)
    print(f"단계 1: FC 데이터 재구성 완료 | 그룹 수: {NUM_GROUP}")

    # 2. 클리핑 및 오프셋 후보군 설정
    offset_min = -2**int(const_bitwidth-1)
    offset_max = 2**int(const_bitwidth-1)
    offsets = list(range(offset_min, offset_max))
    rp_factor = len(offsets)
    
    v_max = 2.**(w_bitwidth-1) - 1
    v_min = -v_max
    
    wq_int_rp = wq_int.unsqueeze(0).repeat(rp_factor, 1, 1)
    for i, offset in enumerate(offsets):
        wq_int_rp[i] = wq_int_rp[i] + float(offset)
    
    wq_int_rp[wq_int_rp.lt(v_min)] = v_min
    wq_int_rp[wq_int_rp.gt(v_max)] = v_max

    # 3. 비트 변환
    from binary_pruning import int_to_signMagnitude, binary_to_int, int_to_binary, signMagnitude_to_int
    wqb_signMagnitude = int_to_signMagnitude(wq_int_rp, w_bitwidth=w_bitwidth, device=device)
    print(f"단계 2~3: 오프셋 후보 생성 및 비트 변환 완료")

    # 4. Pruning 포인터 분석
    prune_until = torch.full([rp_factor, NUM_GROUP], w_bitwidth-num_pruned_column, device=device)
    test_until  = torch.full([rp_factor, NUM_GROUP], 1, device=device)
    is_msb_zero = torch.ones([rp_factor, NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(1, w_bitwidth):
        is_current_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
        is_msb_zero = torch.logical_and(is_msb_zero, is_current_zero)
        prune_until[is_msb_zero] +=  1
        test_until[is_msb_zero]  +=  1

    # 5. 최적값 탐색 (MSE 최소화)
    value_test = torch.zeros_like(wq_int_rp, dtype=torch.float32, device=device)
    value_new = torch.zeros_like(wq_int_rp, dtype=torch.float32, device=device)

    for test_idx in range(1, w_bitwidth):
        mask = torch.eq(test_until, test_idx)
        column_test = wqb_signMagnitude[test_idx:, mask, :]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_idx, device=device)
        value_test[mask] = int_test

    print(f"단계 5: 최적 비트 패턴 탐색 중...")
    for prune_idx in range(w_bitwidth-num_pruned_column, w_bitwidth):
        for test_idx in range(1, prune_idx):
            mask_group = torch.logical_and(torch.eq(test_until, test_idx), torch.eq(prune_until, prune_idx))
            mask_group_exp = mask_group.unsqueeze(-1).expand(-1, -1, group_size)
            error = torch.full([rp_factor, NUM_GROUP, group_size], 1e7, device=device)
            for n in range(2**(prune_idx-test_idx)):
                tmp_value = n * 2.**(w_bitwidth-prune_idx)
                new_error = (tmp_value - value_test) ** 2
                mask_value = torch.logical_and(torch.lt(new_error, error), mask_group_exp)
                error[mask_value] = new_error[mask_value]
                value_new[mask_value] = tmp_value
            column_new = int_to_binary(value_new, w_bitwidth=w_bitwidth-test_idx, device=device)
            wqb_signMagnitude[test_idx:, mask_group_exp] = column_new[:, mask_group_exp]

    # 6. 복원 및 최적 상수(Offset) 선택
    wq_int_pruned = signMagnitude_to_int(wqb_signMagnitude, w_bitwidth=w_bitwidth, device=device)
    for i, offset in enumerate(offsets):
        wq_int_pruned[i] = wq_int_pruned[i] - float(offset)
    
    wq_int_original = wq_int.to(torch.float32)
    wq_int_new = torch.zeros_like(wq_int_original, dtype=torch.float32, device=device)
    error = torch.full([NUM_GROUP], 1e7, device=device)
    best_offsets = torch.zeros([NUM_GROUP], device=device)

    print(f"단계 6: 최종 가중치 복원 및 최적 상수 선택 완료")
    for i, offset in enumerate(offsets):
        new_error = torch.sum((wq_int_pruned[i] - wq_int_original)**2, dim=-1)
        mask_value = torch.lt(new_error, error)
        error[mask_value] = new_error[mask_value]
        wq_int_new[mask_value] = wq_int_pruned[i][mask_value]
        best_offsets[mask_value] = float(offset)

    print(f"      - 선정된 상수 샘플 (첫 4개 그룹): {best_offsets[:4].cpu().numpy()}")
    
    wq_int_new = wq_int_new.reshape(K, C)
    print(f"--- [ZS-FC 절차 종료] ---\n")
    return wq_int_new