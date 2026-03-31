import torch
import pandas as pd
import os
import sys

sys.path.append(r'C:\Users\CS\Desktop\BBS-MICRO')
from binary_pruning import roundAvg_fc, zeroPointShifting_fc

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
PT_PATH        = r'C:\Users\CS\Desktop\BBS-MICRO\gptq\opt125m-8bit.pt'
OUT_PATH       = r'C:\Users\CS\Desktop\BBS-MICRO\bbs_results_2\summary_fixed.xlsx'
W_BITWIDTH     = 8
GROUP_SIZE     = 16
NUM_PRUNED_COL = 4
CONST_BITWIDTH = 5
DEVICE         = 'cuda'

os.makedirs(r'C:\Users\CS\Desktop\BBS-MICRO\bbs_results_2', exist_ok=True)

# ─────────────────────────────────────────
# 1. float → INT8 정수 변환 함수
# ─────────────────────────────────────────
def float_to_int8(state_dict, key):
    weight   = state_dict[key].float()
    zero_key = key.replace('.weight', '.zeros')

    if zero_key in state_dict:
        zeros = state_dict[zero_key].float()  # [K, 1]
        w_int = (weight - zeros).clamp(-128, 127)
    else:
        w_int = weight.clamp(-128, 127)

    return w_int
# ─────────────────────────────────────────
# 2. 모델 로드 및 FC 키 필터링
# ─────────────────────────────────────────
print("모델 로드 중...")
state_dict = torch.load(PT_PATH, map_location='cpu')

fc_keys = [k for k in state_dict.keys()
           if any(x in k for x in ['q_proj', 'k_proj', 'v_proj',
                                    'out_proj', 'fc1', 'fc2'])
           and k.endswith('.weight')
           and any(f'layers.{i}.' in k for i in [0, 1])]
print(f"FC 레이어 수: {len(fc_keys)}")

# ─────────────────────────────────────────
# 3. 엑셀 저장
# ─────────────────────────────────────────
summary_rows = []

with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:

    for key in fc_keys:
        print(f"\n처리 중: {key}")

        # INT8 정수값으로 변환
        tensor = float_to_int8(state_dict, key)  # [K, C] 정수

        # RA / ZS 적용
        ra_result = roundAvg_fc(
            tensor, w_bitwidth=W_BITWIDTH, group_size=GROUP_SIZE,
            num_pruned_column=NUM_PRUNED_COL, device=DEVICE
        ).cpu()

        zs_result = zeroPointShifting_fc(
            tensor, w_bitwidth=W_BITWIDTH, group_size=GROUP_SIZE,
            num_pruned_column=NUM_PRUNED_COL, const_bitwidth=CONST_BITWIDTH,
            device=DEVICE
        ).cpu()

        tensor = tensor.cpu()

        # ── 그룹별 상세 비교 (GROUP_SIZE 개씩 묶어서)
        flat_orig = tensor.flatten().tolist()
        flat_ra   = ra_result.flatten().tolist()
        flat_zs   = zs_result.flatten().tolist()

        total     = len(flat_orig)
        num_groups = total // GROUP_SIZE

        detail_rows = []
        for g in range(num_groups):
            s = g * GROUP_SIZE
            e = s + GROUP_SIZE

            orig_group = [round(x) for x in flat_orig[s:e]]
            ra_group   = [round(x) for x in flat_ra[s:e]]
            zs_group   = [round(x) for x in flat_zs[s:e]]

            ra_mse_g = sum((o-r)**2 for o,r in zip(orig_group, ra_group)) / GROUP_SIZE
            zs_mse_g = sum((o-z)**2 for o,z in zip(orig_group, zs_group)) / GROUP_SIZE

            detail_rows.append({
                'Group'     : g,
                'Original'  : str(orig_group),
                'RA_Result' : str(ra_group),
                'ZS_Result' : str(zs_group),
                'RA_MSE'    : round(ra_mse_g, 4),
                'ZS_MSE'    : round(zs_mse_g, 4),
                'Winner'    : 'ZS' if zs_mse_g < ra_mse_g else 'RA'
            })

        df_detail = pd.DataFrame(detail_rows)

        # 시트 이름 (최대 31자)
        sheet_name = key.replace('model.decoder.layers.', 'L') \
                        .replace('.self_attn.', '.') \
                        .replace('.weight', '')[:31]

        df_detail.to_excel(writer, index=False, sheet_name=sheet_name)

        # 컬럼 너비 조정
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
            ws.column_dimensions[col[0].column_letter].width = max_len + 4

        # 요약 통계
        ra_mse_total = torch.mean((ra_result - tensor.float())**2).item()
        zs_mse_total = torch.mean((zs_result - tensor.float())**2).item()

        print(f"  RA_MSE={ra_mse_total:.4f} | ZS_MSE={zs_mse_total:.4f}")

        summary_rows.append({
            'Layer'  : key,
            'Shape'  : str(tuple(tensor.shape)),
            'RA_MSE' : round(ra_mse_total, 6),
            'ZS_MSE' : round(zs_mse_total, 6),
            'Winner' : 'ZS' if zs_mse_total < ra_mse_total else 'RA'
        })

    # Summary 시트
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_excel(writer, index=False, sheet_name='Summary_fixed')
    ws = writer.sheets['Summary_fixed']
    for col in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 4

print(f"\n저장 완료: {OUT_PATH}")