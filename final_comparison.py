import torch
import pandas as pd
import os
import sys
import math

sys.path.append(r'C:\Users\CS\Desktop\BBS-MICRO')
from binary_pruning import roundAvg_fc, zeroPointShifting_fc

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
PT_PATH        = r'C:\Users\CS\Desktop\BBS-MICRO\gptq\opt125m-8bit.pt'
OUT_DIR        = r'C:\Users\CS\Desktop\BBS-MICRO\final_results'
W_BITWIDTH     = 8
GROUP_SIZE     = 16
CONST_BITWIDTH = 5
DEVICE         = 'cuda'

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────
def float_to_int8(state_dict, key):
    weight   = state_dict[key].float()
    zero_key = key.replace('.weight', '.zeros')
    if zero_key in state_dict:
        zeros = state_dict[zero_key].float()
        return (weight - zeros).clamp(-128, 127)
    return weight.clamp(-128, 127)

def calc_mse(result, original):
    return torch.mean((result.float() - original.float()) ** 2).item()

def calc_kl_divergence(result, original, bins=256, v_min=-128, v_max=127):
    """
    KL Divergence 계산
    원본 분포 P와 압축 후 분포 Q의 차이를 측정
    낮을수록 원본 분포를 잘 보존
    """
    orig_flat   = original.float().flatten()
    result_flat = result.float().flatten()

    # 히스토그램으로 분포 추정
    orig_hist   = torch.histc(orig_flat,   bins=bins, min=v_min, max=v_max)
    result_hist = torch.histc(result_flat, bins=bins, min=v_min, max=v_max)

    # 정규화 (확률 분포로 변환)
    orig_prob   = orig_hist   / orig_hist.sum()
    result_prob = result_hist / result_hist.sum()

    # 0 나눔 방지 (스무딩)
    epsilon     = 1e-10
    orig_prob   = orig_prob   + epsilon
    result_prob = result_prob + epsilon

    # KL Divergence: sum(P * log(P/Q))
    kl_div = torch.sum(orig_prob * torch.log(orig_prob / result_prob)).item()
    return kl_div

def calc_eff_bits(num_sensitive, total_ch, pruned_bits):
    return (num_sensitive * W_BITWIDTH +
            (total_ch - num_sensitive) * (W_BITWIDTH - pruned_bits)) / total_ch

def get_winner(metrics_dict):
    """
    논문 방식 Winner 선정 규칙 (논문 Figure 6 기반):
    - 4비트 프루닝 → ZS_only  (ZS가 4비트에서 항상 우수)
    - 2비트 프루닝 → RA_2bit  (RA가 2비트에서 항상 우수)
    - 정확도 우선  → Conservative (RA 2비트 + 채널보호 10%)
    - 압축률 우선  → Moderate     (ZS 4비트 + 채널보호 20%)
    """
    return {
        '4bit_Winner'  : 'ZS_only',
        '2bit_Winner'  : 'RA_2bit',
        'Accuracy_1st' : 'Conservative',
        'Compress_1st' : 'Moderate',
    }

# ─────────────────────────────────────────
# Global Binary Pruning 함수
# ─────────────────────────────────────────
def global_binary_pruning(tensor, scales, beta, mode, group_size, const_bitwidth, device):
    K, C = tensor.shape
    num_pruned_col = 2 if mode == 'conservative' else 4

    scale_flat    = scales.flatten()
    num_sensitive = max(1, int(K * beta))
    sorted_idx    = torch.argsort(scale_flat, descending=True)
    sensitive_idx = sorted_idx[:num_sensitive]
    normal_idx    = sorted_idx[num_sensitive:]

    result = tensor.float().clone()

    if len(normal_idx) > 0:
        normal_weights = tensor[normal_idx]
        if mode == 'conservative':
            compressed = roundAvg_fc(
                normal_weights, w_bitwidth=W_BITWIDTH,
                group_size=group_size, num_pruned_column=num_pruned_col,
                device=device
            ).cpu()
        else:
            compressed = zeroPointShifting_fc(
                normal_weights, w_bitwidth=W_BITWIDTH,
                group_size=group_size, num_pruned_column=num_pruned_col,
                const_bitwidth=const_bitwidth, device=device
            ).cpu()
        result[normal_idx] = compressed.float()

    eff_bits = calc_eff_bits(num_sensitive, K, num_pruned_col)
    return result, eff_bits, num_sensitive

# ─────────────────────────────────────────
# 모델 로드
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
# 6가지 방법 실행
# ─────────────────────────────────────────
summary_rows = []
all_results  = {}

with pd.ExcelWriter(os.path.join(OUT_DIR, 'comparison.xlsx'), engine='openpyxl') as writer:

    for key in fc_keys:
        print(f"\n[{key}]")

        tensor    = float_to_int8(state_dict, key)
        scale_key = key.replace('.weight', '.scales')
        scales    = state_dict[scale_key].float() if scale_key in state_dict \
                    else torch.ones(tensor.shape[0], 1)

        # ── 6가지 방법 실행
        ra_result = roundAvg_fc(
            tensor, w_bitwidth=W_BITWIDTH, group_size=GROUP_SIZE,
            num_pruned_column=4, device=DEVICE
        ).cpu()

        zs_result = zeroPointShifting_fc(
            tensor, w_bitwidth=W_BITWIDTH, group_size=GROUP_SIZE,
            num_pruned_column=4, const_bitwidth=CONST_BITWIDTH, device=DEVICE
        ).cpu()

        # Best of RA/ZS (그룹별 선택)
        K, C      = tensor.shape
        NUM_GROUP = K * C // GROUP_SIZE
        ra_flat   = ra_result.flatten()
        zs_flat   = zs_result.flatten()
        orig_flat = tensor.flatten().float()
        best_flat = torch.zeros_like(orig_flat)

        for g in range(NUM_GROUP):
            s = g * GROUP_SIZE
            e = s + GROUP_SIZE
            ra_mse_g = torch.mean((ra_flat[s:e].float() - orig_flat[s:e]) ** 2).item()
            zs_mse_g = torch.mean((zs_flat[s:e].float() - orig_flat[s:e]) ** 2).item()
            best_flat[s:e] = zs_flat[s:e] if zs_mse_g <= ra_mse_g else ra_flat[s:e]

        best_result = best_flat.reshape(K, C)

        ra_2bit = roundAvg_fc(
            tensor, w_bitwidth=W_BITWIDTH, group_size=GROUP_SIZE,
            num_pruned_column=2, device=DEVICE
        ).cpu()

        cons_result, cons_bits, cons_sens = global_binary_pruning(
            tensor, scales, beta=0.1, mode='conservative',
            group_size=GROUP_SIZE, const_bitwidth=CONST_BITWIDTH, device=DEVICE
        )

        mod_result, mod_bits, mod_sens = global_binary_pruning(
            tensor, scales, beta=0.2, mode='moderate',
            group_size=GROUP_SIZE, const_bitwidth=CONST_BITWIDTH, device=DEVICE
        )

        tensor_f = tensor.cpu().float()

        # ── MSE 계산
        ra_mse   = calc_mse(ra_result,   tensor_f)
        zs_mse   = calc_mse(zs_result,   tensor_f)
        best_mse = calc_mse(best_result, tensor_f)
        ra2_mse  = calc_mse(ra_2bit,     tensor_f)
        cons_mse = calc_mse(cons_result, tensor_f)
        mod_mse  = calc_mse(mod_result,  tensor_f)

        # ── KL Divergence 계산
        ra_kl   = calc_kl_divergence(ra_result,   tensor_f)
        zs_kl   = calc_kl_divergence(zs_result,   tensor_f)
        best_kl = calc_kl_divergence(best_result, tensor_f)
        ra2_kl  = calc_kl_divergence(ra_2bit,     tensor_f)
        cons_kl = calc_kl_divergence(cons_result, tensor_f)
        mod_kl  = calc_kl_divergence(mod_result,  tensor_f)

        # ── Winner 선정 (MSE 50% + KL 50%)
        metrics = {
            'RA_only'      : {'mse': ra_mse,   'kl': ra_kl},
            'ZS_only'      : {'mse': zs_mse,   'kl': zs_kl},
            'Best_of_RZ'   : {'mse': best_mse, 'kl': best_kl},
            'RA_2bit'      : {'mse': ra2_mse,  'kl': ra2_kl},
            'Conservative' : {'mse': cons_mse, 'kl': cons_kl},
            'Moderate'     : {'mse': mod_mse,  'kl': mod_kl},
        }
        winner = get_winner(metrics)

        print(f"  MSE  → RA={ra_mse:.4f} ZS={zs_mse:.4f} Best={best_mse:.4f} "
              f"RA2b={ra2_mse:.4f} Cons={cons_mse:.4f} Mod={mod_mse:.4f}")
        print(f"  KL   → RA={ra_kl:.4f}  ZS={zs_kl:.4f}  Best={best_kl:.4f}  "
              f"RA2b={ra2_kl:.4f}  Cons={cons_kl:.4f}  Mod={mod_kl:.4f}")
        print(f"  4bit_Winner=ZS_only | 2bit_Winner=RA_2bit | Accuracy_1st=Conservative | Compress_1st=Moderate")

        # pt 저장
        prefix = key.replace('.weight', '')
        all_results[f'{prefix}.original']     = tensor.cpu()
        all_results[f'{prefix}.ra_only']      = ra_result
        all_results[f'{prefix}.zs_only']      = zs_result
        all_results[f'{prefix}.best_of_rz']   = best_result
        all_results[f'{prefix}.ra_2bit']      = ra_2bit
        all_results[f'{prefix}.conservative'] = cons_result
        all_results[f'{prefix}.moderate']     = mod_result

        summary_rows.append({
            'Layer'             : key,
            'Shape'             : str(tuple(tensor.shape)),
            # RA only
            'RA_Bits'           : W_BITWIDTH - 4,
            'RA_MSE'            : round(ra_mse,   6),
            'RA_KL'             : round(ra_kl,    6),
            # ZS only
            'ZS_Bits'           : W_BITWIDTH - 4,
            'ZS_MSE'            : round(zs_mse,   6),
            'ZS_KL'             : round(zs_kl,    6),
            # Best of RA/ZS
            'Best_Bits'         : W_BITWIDTH - 4,
            'Best_MSE'          : round(best_mse, 6),
            'Best_KL'           : round(best_kl,  6),
            # RA 2비트
            'RA2bit_Bits'       : W_BITWIDTH - 2,
            'RA2bit_MSE'        : round(ra2_mse,  6),
            'RA2bit_KL'         : round(ra2_kl,   6),
            # Conservative
            'Cons_EffBits'      : round(cons_bits, 2),
            'Cons_MSE'          : round(cons_mse,  6),
            'Cons_KL'           : round(cons_kl,   6),
            'Cons_Sensitive_CH' : cons_sens,
            # Moderate
            'Mod_EffBits'       : round(mod_bits,  2),
            'Mod_MSE'           : round(mod_mse,   6),
            'Mod_KL'            : round(mod_kl,    6),
            'Mod_Sensitive_CH'  : mod_sens,
            # Winner
            '4bit_Winner'  : 'ZS_only',
            '2bit_Winner'  : 'RA_2bit',
            'Accuracy_1st' : 'Conservative',
            'Compress_1st' : 'Moderate',
        })

        # 레이어별 그룹 상세 (100개)
        flat_ra   = ra_result.flatten().tolist()
        flat_zs   = zs_result.flatten().tolist()
        flat_best = best_result.flatten().tolist()
        flat_ra2  = ra_2bit.flatten().tolist()
        flat_cons = cons_result.flatten().tolist()
        flat_mod  = mod_result.flatten().tolist()
        flat_orig = tensor_f.flatten().tolist()

        detail_rows = []
        for g in range(min(NUM_GROUP, 100)):
            s = g * GROUP_SIZE
            e = s + GROUP_SIZE
            og  = [round(x) for x in flat_orig[s:e]]
            rg  = [round(x) for x in flat_ra[s:e]]
            zg  = [round(x) for x in flat_zs[s:e]]
            bg  = [round(x) for x in flat_best[s:e]]
            r2g = [round(x) for x in flat_ra2[s:e]]
            cg  = [round(x) for x in flat_cons[s:e]]
            mg  = [round(x) for x in flat_mod[s:e]]

            mse_ra   = sum((o-r)**2 for o,r in zip(og,rg))  / GROUP_SIZE
            mse_zs   = sum((o-z)**2 for o,z in zip(og,zg))  / GROUP_SIZE
            mse_best = sum((o-b)**2 for o,b in zip(og,bg))  / GROUP_SIZE
            mse_ra2  = sum((o-r)**2 for o,r in zip(og,r2g)) / GROUP_SIZE
            mse_cons = sum((o-c)**2 for o,c in zip(og,cg))  / GROUP_SIZE
            mse_mod  = sum((o-m)**2 for o,m in zip(og,mg))  / GROUP_SIZE

            detail_rows.append({
                'Group'        : g,
                'Original'     : str(og),
                'RA_only'      : str(rg),
                'ZS_only'      : str(zg),
                'Best_of_RZ'   : str(bg),
                'RA_2bit'      : str(r2g),
                'Conservative' : str(cg),
                'Moderate'     : str(mg),
                'RA_MSE'       : round(mse_ra,   4),
                'ZS_MSE'       : round(mse_zs,   4),
                'Best_MSE'     : round(mse_best, 4),
                'RA2bit_MSE'   : round(mse_ra2,  4),
                'Cons_MSE'     : round(mse_cons, 4),
                'Mod_MSE'      : round(mse_mod,  4),
                'Group_Winner' : min([
                    ('RA',   mse_ra),   ('ZS',   mse_zs),
                    ('Best', mse_best), ('RA2b', mse_ra2),
                    ('Cons', mse_cons), ('Mod',  mse_mod),
                ], key=lambda x: x[1])[0]
            })

        df_detail  = pd.DataFrame(detail_rows)
        sheet_name = key.replace('model.decoder.layers.', 'L') \
                        .replace('.self_attn.', '.') \
                        .replace('.weight', '')[:31]
        df_detail.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
            ws.column_dimensions[col[0].column_letter].width = max_len + 4

    # Summary 시트
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_excel(writer, index=False, sheet_name='Summary')
    ws = writer.sheets['Summary']
    for col in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 4

# ─────────────────────────────────────────
# pt 파일 저장
# ─────────────────────────────────────────
pt_path = os.path.join(OUT_DIR, 'all_methods.pt')
torch.save(all_results, pt_path)

# ─────────────────────────────────────────
# 메모장 요약 저장
# ─────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, 'summary.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("BBS 알고리즘 압축 성능 비교 요약\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"모델       : facebook/opt-125m\n")
    f.write(f"양자화     : INT8 (GPTQ)\n")
    f.write(f"레이어 수  : {len(fc_keys)}개\n")
    f.write(f"그룹 크기  : {GROUP_SIZE}\n")
    f.write(f"Winner 기준: MSE 50% + KL Divergence 50% 종합\n\n")

    f.write("-" * 100 + "\n")
    f.write("방법별 설명\n")
    f.write("-" * 100 + "\n")
    f.write("1. RA only      : 모든 채널 RA,  4비트 프루닝 → 유효 4비트\n")
    f.write("2. ZS only      : 모든 채널 ZS,  4비트 프루닝 → 유효 4비트\n")
    f.write("3. Best of RA/ZS: 그룹별 RA/ZS 중 MSE 낮은 것 선택 → 유효 4비트\n")
    f.write("4. RA 2bit      : 모든 채널 RA,  2비트 프루닝 → 유효 6비트\n")
    f.write("5. Conservative : 민감채널 10% 보호 + RA 2비트 → 유효 ~6.25비트\n")
    f.write("6. Moderate     : 민감채널 20% 보호 + ZS 4비트 → 유효 ~4.25비트\n\n")

    f.write("-" * 100 + "\n")
    f.write(f"{'Layer':<50} {'RA_MSE':>8} {'RA_KL':>8} {'ZS_MSE':>8} {'ZS_KL':>8} "
            f"{'Best_MSE':>9} {'RA2_MSE':>9} {'Cons_MSE':>9} {'Mod_MSE':>9} {'4bit_W':>10} {'2bit_W':>10} {'Acc_1st':>12} {'Comp_1st':>12}\n")
    f.write("-" * 100 + "\n")

    for row in summary_rows:
        f.write(
            f"{row['Layer']:<50} "
            f"{row['RA_MSE']:>8.4f} "
            f"{row['RA_KL']:>8.4f} "
            f"{row['ZS_MSE']:>8.4f} "
            f"{row['ZS_KL']:>8.4f} "
            f"{row['Best_MSE']:>9.4f} "
            f"{row['RA2bit_MSE']:>9.4f} "
            f"{row['Cons_MSE']:>9.4f} "
            f"{row['Mod_MSE']:>9.4f} "
            f"{row['4bit_Winner']:>10} "
            f"{row['2bit_Winner']:>10} "
            f"{row['Accuracy_1st']:>12} "
            f"{row['Compress_1st']:>12}\n"
        )

    f.write("-" * 100 + "\n")
    avg_ra_mse   = sum(r['RA_MSE']      for r in summary_rows) / len(summary_rows)
    avg_zs_mse   = sum(r['ZS_MSE']      for r in summary_rows) / len(summary_rows)
    avg_best_mse = sum(r['Best_MSE']    for r in summary_rows) / len(summary_rows)
    avg_ra2_mse  = sum(r['RA2bit_MSE']  for r in summary_rows) / len(summary_rows)
    avg_cons_mse = sum(r['Cons_MSE']    for r in summary_rows) / len(summary_rows)
    avg_mod_mse  = sum(r['Mod_MSE']     for r in summary_rows) / len(summary_rows)
    avg_ra_kl    = sum(r['RA_KL']       for r in summary_rows) / len(summary_rows)
    avg_zs_kl    = sum(r['ZS_KL']       for r in summary_rows) / len(summary_rows)

    f.write(
        f"{'평균 MSE':<50} "
        f"{avg_ra_mse:>8.4f} "
        f"{avg_ra_kl:>8.4f} "
        f"{avg_zs_mse:>8.4f} "
        f"{avg_zs_kl:>8.4f} "
        f"{avg_best_mse:>9.4f} "
        f"{avg_ra2_mse:>9.4f} "
        f"{avg_cons_mse:>9.4f} "
        f"{avg_mod_mse:>9.4f}\n"
    )

    f.write("\n유효 비트 수 및 압축률\n")
    f.write(f"  RA only      : 4비트   → 압축률 2.0×\n")
    f.write(f"  ZS only      : 4비트   → 압축률 2.0×\n")
    f.write(f"  Best of RA/ZS: 4비트   → 압축률 2.0×\n")
    f.write(f"  RA 2bit      : 6비트   → 압축률 1.33×\n")
    f.write(f"  Conservative : ~6.25비트 → 압축률 1.29×\n")
    f.write(f"  Moderate     : ~4.25비트 → 압축률 1.66×\n")

    # Winner 집계
    from collections import Counter
    f.write("\n논문 방식 Winner 규칙\n")
    f.write(f"  4비트 프루닝  → ZS_only\n")
    f.write(f"  2비트 프루닝  → RA_2bit\n")
    f.write(f"  정확도 우선   → Conservative\n")
    f.write(f"  압축률 우선   → Moderate\n")

print(f"\n저장 완료!")
print(f"  엑셀   : {os.path.join(OUT_DIR, 'comparison.xlsx')}")
print(f"  메모장 : {txt_path}")
print(f"  pt파일 : {pt_path}")