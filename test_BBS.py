import torch
import numpy as np
from binary_pruning import roundAvg_conv, zeroPointShifting_conv

def run_bbs_test():
    # 설정값
    BIT_WIDTH = 8
    GROUP_SIZE = 16
    NUM_PRUNED = 4  # 8비트 중 4개 비트 컬럼을 희소화(Pruning) 타겟으로 설정
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"--- [실험 시작] 사용 디바이스: {DEVICE} ---")

    # 1. 가상의 8-bit Quantized 가중치 생성 (Conv 레이어 형태: [K, C, H, W])
    # -128 ~ 127 사이의 정수형 텐서 생성
    # torch.manual_seed(42)

    original_wq = torch.randint(-64, 65, (64, 32, 3, 3), dtype=torch.float32).to(DEVICE)
    
    print(f"가중치 형상: {original_wq.shape}")
    print(f"원본 샘플 (첫 4개): {original_wq.flatten()[:4].cpu().numpy()}")

    # 2. Zero-Point Shifting (ZS) 실험
    # ZS는 가중치 분포를 미세하게 이동시켜 MSE를 최소화하며 희소성을 확보하기 위한 최적의 오프셋을 찾습니다.
    print(f"\n[1/2] Zero-Point Shifting 적용 중...")
    zs_weight = zeroPointShifting_conv(
        original_wq, 
        w_bitwidth=BIT_WIDTH, 
        group_size=GROUP_SIZE, 
        num_pruned_column=NUM_PRUNED,
        device=DEVICE
    )
    
    mse_zs = torch.mean((original_wq - zs_weight) ** 2).item()
    print(f" - ZS 적용 후 MSE: {mse_zs:.4f}")

    # 3. Rounded Averaging (RA) 실험
    # ZS 결과물에 RA를 적용하여 비트 컬럼들을 평균화(희소화)합니다.
    print(f"\n[2/2] Rounded Averaging 적용 중...")
    final_weight = roundAvg_conv(
        zs_weight, 
        w_bitwidth=BIT_WIDTH, 
        group_size=GROUP_SIZE, 
        num_pruned_column=NUM_PRUNED,
        device=DEVICE
    )

    mse_final = torch.mean((original_wq - final_weight) ** 2).item()
    print(f" - 최종(ZS+RA) 적용 후 MSE: {mse_final:.4f}")

    # 4. 결과 분석
    print(f"\n--- [실험 결과 요약] ---")
    print(f"원본 대비 최종 가중치 변화 샘플:")
    print(f"  원본: {original_wq.flatten()[:4].cpu().numpy()}")
    print(f"  결과: {final_weight.flatten()[:4].cpu().numpy()}")
    
    # 5. 작동 검증 로직
    diff_count = torch.sum(original_wq != final_weight).item()
    total_count = original_wq.numel()
    print(f"\n변경된 값의 비율: {(diff_count/total_count)*100:.2f}%")
    print("설명: BBS는 비트 수준에서 값을 조정하므로, 정수 값이 미세하게 변하면서도")
    print("      비트 표현 시 공통된 비트 패턴(Sparsity)을 가지게 됩니다.")

if __name__ == "__main__":
    run_bbs_test() 
