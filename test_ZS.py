import torch
from binary_pruning_test import zeroPointShifting_conv

# 1. 원본 가중치 생성
# torch.manual_seed(42)
original_wq = torch.randint(-64, 65, (64, 1, 1, 1), dtype=torch.float32)
total_elements = original_wq.numel()
custom_group_size = total_elements // 4
# 2. ZS만 적용
# const_bitwidth=5: 시프팅 오프셋의 범위를 결정 (보통 5비트 내외)
zs_only_weight = zeroPointShifting_conv(
    original_wq.clone(), 
    w_bitwidth=8, 
    num_pruned_column=4, 
    const_bitwidth=4
)

print("\n--- [ZS 단독 테스트] ---")
print(f"원본 샘플: {original_wq.flatten()[:4].numpy()}")
print(f"ZS 결과 : {zs_only_weight.flatten()[:4].numpy()}")
print(f"평균 오차(MSE): {torch.mean((original_wq - zs_only_weight)**2).item():.4f}")
