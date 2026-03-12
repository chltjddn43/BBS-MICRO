import torch
from binary_pruning_test import roundAvg_conv

# 1. 원본 가중치 생성
# torch.manual_seed(42)
original_wq = torch.randint(-64, 65, (64, 32, 3, 3), dtype=torch.float32)

# 2. RA만 적용
# num_pruned_column=4: 8비트 중 뒤쪽 4개 비트 컬럼을 희소화 대상으로 지정
ra_only_weight = roundAvg_conv(original_wq.clone(), w_bitwidth=8, group_size=16, num_pruned_column=4)

print(f"원본 샘플: {original_wq.flatten()[:4].numpy()}")
print(f"RA 결과 : {ra_only_weight.flatten()[:4].numpy()}")
print(f"평균 오차(MSE): {torch.mean((original_wq - ra_only_weight)**2).item():.4f}")
