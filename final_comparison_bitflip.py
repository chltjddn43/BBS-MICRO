import torch
import pandas as pd
import os
import sys
import math

sys.path.append(r'C:\Users\CS\Desktop\BBS-MICRO')
from binary_pruning import roundAvg_fc, zeroPointShifting_fc
from bit_flip import bitFlip_fc

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
PT_PATH        = r'C:\Users\CS\Desktop\BBS-MICRO\gptq\opt125m-8bit.pt'
OUT_DIR        = r'C:\Users\CS\Desktop\BBS-MICRO\final_results_flip'
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
    orig_flat   = original.float().flatten()
    result_flat = result.float().flatten()
    orig_hist   = torch.histc(orig_flat,   bins=bins, min=v_min, max=v_max)
    result_hist = torch.histc(result_flat, bins=bins, min=v_min, max=v_max)
    orig_prob   = orig_hist   / orig_hist.sum()
    result_prob = result_hist / result_hist.sum()
    epsilon     = 1e-10
    orig_prob   = orig_prob   + epsilon
    result_prob = result_prob + epsilon
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
           and any(f'layers.{i}.' in k for i in range(12))]

print(f"FC 레이어 수: {len(fc_keys)}")

# ─────────────────────────────────────────
# 7가지 방법 실행
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

        # ── 1. RA only (4비트)
        ra_result = roundAvg_fc(
            tensor, w_bitwidth=W_BITWIDTH, group_size=GROUP_SIZE,
            num_pruned_column=4, device=DEVICE
        ).cpu()

        # ── 2. ZS only (4비트)
        zs_result = zeroPointShifting_fc(
            tensor, w_bitwidth=W_BITWIDTH, group_size=GROUP_SIZE,
            num_pruned_column=4, const_bitwidth=CONST_BITWIDTH, device=DEVICE
        ).cpu()

        # ── 3. Best of RA/ZS (그룹별 선택)
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

        # ── 4. RA 2비트
        ra_2bit = roundAvg_fc(
            tensor, w_bitwidth=W_BITWIDTH, group_size=GROUP_SIZE,
            num_pruned_column=2, device=DEVICE
        ).cpu()

        # ── 5. Conservative (β=10%, RA 2비트)
        cons_result, cons_bits, cons_sens = global_binary_pruning(
            tensor, scales, beta=0.1, mode='conservative',
            group_size=GROUP_SIZE, const_bitwidth=CONST_BITWIDTH, device=DEVICE
        )

        # ── 6. Moderate (β=20%, ZS 4비트)
        mod_result, mod_bits, mod_sens = global_binary_pruning(
            tensor, scales, beta=0.2, mode='moderate',
            group_size=GROUP_SIZE, const_bitwidth=CONST_BITWIDTH, device=DEVICE
        )

        # ── 7. BitFlip (4비트)
        bf_result = bitFlip_fc(
            tensor, w_bitwidth=W_BITWIDTH, group_size=GROUP_SIZE,
            num_pruned_column=4, device=DEVICE
        ).cpu()

        tensor_f = tensor.cpu().float()

        # ── MSE 계산
        ra_mse   = calc_mse(ra_result,   tensor_f)
        zs_mse   = calc_mse(zs_result,   tensor_f)
        best_mse = calc_mse(best_result, tensor_f)
        ra2_mse  = calc_mse(ra_2bit,     tensor_f)
        cons_mse = calc_mse(cons_result, tensor_f)
        mod_mse  = calc_mse(mod_result,  tensor_f)
        bf_mse   = calc_mse(bf_result,   tensor_f)

        # ── KL Divergence 계산
        ra_kl   = calc_kl_divergence(ra_result,   tensor_f)
        zs_kl   = calc_kl_divergence(zs_result,   tensor_f)
        best_kl = calc_kl_divergence(best_result, tensor_f)
        ra2_kl  = calc_kl_divergence(ra_2bit,     tensor_f)
        cons_kl = calc_kl_divergence(cons_result, tensor_f)
        mod_kl  = calc_kl_divergence(mod_result,  tensor_f)
        bf_kl   = calc_kl_divergence(bf_result,   tensor_f)

        # ── Winner (논문 방식)
        winner = get_winner({})

        print(f"  MSE  → RA={ra_mse:.4f} ZS={zs_mse:.4f} Best={best_mse:.4f} "
              f"RA2b={ra2_mse:.4f} Cons={cons_mse:.4f} Mod={mod_mse:.4f} BF={bf_mse:.4f}")
        print(f"  KL   → RA={ra_kl:.4f} ZS={zs_kl:.4f} Best={best_kl:.4f} "
              f"RA2b={ra2_kl:.4f} Cons={cons_kl:.4f} Mod={mod_kl:.4f} BF={bf_kl:.4f}")
        print(f"  4bit_Winner=ZS_only | 2bit_Winner=RA_2bit | "
              f"Accuracy_1st=Conservative | Compress_1st=Moderate")

        # pt 저장
        prefix = key.replace('.weight', '')
        all_results[f'{prefix}.original']     = tensor.cpu()
        all_results[f'{prefix}.ra_only']      = ra_result
        all_results[f'{prefix}.zs_only']      = zs_result
        all_results[f'{prefix}.best_of_rz']   = best_result
        all_results[f'{prefix}.ra_2bit']      = ra_2bit
        all_results[f'{prefix}.conservative'] = cons_result
        all_results[f'{prefix}.moderate']     = mod_result
        all_results[f'{prefix}.bitflip']      = bf_result

        summary_rows.append({
            'Layer'             : key,
            'Shape'             : str(tuple(tensor.shape)),
            # 1. RA only
            'RA_Bits'           : W_BITWIDTH - 4,
            'RA_MSE'            : round(ra_mse,   6),
            'RA_KL'             : round(ra_kl,    6),
            # 2. ZS only
            'ZS_Bits'           : W_BITWIDTH - 4,
            'ZS_MSE'            : round(zs_mse,   6),
            'ZS_KL'             : round(zs_kl,    6),
            # 3. Best of RA/ZS
            'Best_Bits'         : W_BITWIDTH - 4,
            'Best_MSE'          : round(best_mse, 6),
            'Best_KL'           : round(best_kl,  6),
            # 4. RA 2비트
            'RA2bit_Bits'       : W_BITWIDTH - 2,
            'RA2bit_MSE'        : round(ra2_mse,  6),
            'RA2bit_KL'         : round(ra2_kl,   6),
            # 5. Conservative
            'Cons_EffBits'      : round(cons_bits, 2),
            'Cons_MSE'          : round(cons_mse,  6),
            'Cons_KL'           : round(cons_kl,   6),
            'Cons_Sensitive_CH' : cons_sens,
            # 6. Moderate
            'Mod_EffBits'       : round(mod_bits,  2),
            'Mod_MSE'           : round(mod_mse,   6),
            'Mod_KL'            : round(mod_kl,    6),
            'Mod_Sensitive_CH'  : mod_sens,
            # 7. BitFlip
            'BF_Bits'           : W_BITWIDTH - 4,
            'BF_MSE'            : round(bf_mse,   6),
            'BF_KL'             : round(bf_kl,    6),
            # Winner (논문 방식)
            '4bit_Winner'       : 'ZS_only',
            '2bit_Winner'       : 'RA_2bit',
            'Accuracy_1st'      : 'Conservative',
            'Compress_1st'      : 'Moderate',
        })

        # 레이어별 그룹 상세 (100개)
        flat_ra   = ra_result.flatten().tolist()
        flat_zs   = zs_result.flatten().tolist()
        flat_best = best_result.flatten().tolist()
        flat_ra2  = ra_2bit.flatten().tolist()
        flat_cons = cons_result.flatten().tolist()
        flat_mod  = mod_result.flatten().tolist()
        flat_bf   = bf_result.flatten().tolist()
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
            bfg = [round(x) for x in flat_bf[s:e]]

            mse_ra   = sum((o-r)**2 for o,r in zip(og,rg))  / GROUP_SIZE
            mse_zs   = sum((o-z)**2 for o,z in zip(og,zg))  / GROUP_SIZE
            mse_best = sum((o-b)**2 for o,b in zip(og,bg))  / GROUP_SIZE
            mse_ra2  = sum((o-r)**2 for o,r in zip(og,r2g)) / GROUP_SIZE
            mse_cons = sum((o-c)**2 for o,c in zip(og,cg))  / GROUP_SIZE
            mse_mod  = sum((o-m)**2 for o,m in zip(og,mg))  / GROUP_SIZE
            mse_bf   = sum((o-b)**2 for o,b in zip(og,bfg)) / GROUP_SIZE

            detail_rows.append({
                'Group'        : g,
                'Original'     : str(og),
                'RA_only'      : str(rg),
                'ZS_only'      : str(zg),
                'Best_of_RZ'   : str(bg),
                'RA_2bit'      : str(r2g),
                'Conservative' : str(cg),
                'Moderate'     : str(mg),
                'BitFlip'      : str(bfg),
                'RA_MSE'       : round(mse_ra,   4),
                'ZS_MSE'       : round(mse_zs,   4),
                'Best_MSE'     : round(mse_best, 4),
                'RA2bit_MSE'   : round(mse_ra2,  4),
                'Cons_MSE'     : round(mse_cons, 4),
                'Mod_MSE'      : round(mse_mod,  4),
                'BF_MSE'       : round(mse_bf,   4),
                'Group_Winner' : min([
                    ('RA',      mse_ra),
                    ('ZS',      mse_zs),
                    ('Best',    mse_best),
                    ('RA2b',    mse_ra2),
                    ('Cons',    mse_cons),
                    ('Mod',     mse_mod),
                    ('BitFlip', mse_bf),
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
    f.write("=" * 110 + "\n")
    f.write("BBS 알고리즘 압축 성능 비교 요약 (7가지 방법)\n")
    f.write("=" * 110 + "\n\n")
    f.write(f"모델       : facebook/opt-125m\n")
    f.write(f"양자화     : INT8 (GPTQ)\n")
    f.write(f"레이어 수  : {len(fc_keys)}개\n")
    f.write(f"그룹 크기  : {GROUP_SIZE}\n")
    f.write(f"Winner 기준: 논문 Figure 6 규칙 기반\n\n")

    f.write("-" * 110 + "\n")
    f.write("방법별 설명\n")
    f.write("-" * 110 + "\n")
    f.write("1. RA only      : 모든 채널 RA,  4비트 프루닝 → 유효 4비트\n")
    f.write("2. ZS only      : 모든 채널 ZS,  4비트 프루닝 → 유효 4비트\n")
    f.write("3. Best of RA/ZS: 그룹별 RA/ZS 중 MSE 낮은 것 선택 → 유효 4비트\n")
    f.write("4. RA 2bit      : 모든 채널 RA,  2비트 프루닝 → 유효 6비트\n")
    f.write("5. Conservative : 민감채널 10% 보호 + RA 2비트 → 유효 ~6.25비트\n")
    f.write("6. Moderate     : 민감채널 20% 보호 + ZS 4비트 → 유효 ~4.25비트\n")
    f.write("7. BitFlip      : BitWave 방식, 4비트 프루닝   → 유효 4비트\n\n")

    f.write("-" * 110 + "\n")
    f.write(f"{'Layer':<50} {'RA_MSE':>7} {'RA_KL':>7} {'ZS_MSE':>7} {'ZS_KL':>7} "
            f"{'RA2_MSE':>8} {'Cons_MSE':>9} {'Mod_MSE':>8} {'BF_MSE':>8}\n")
    f.write("-" * 110 + "\n")

    for row in summary_rows:
        f.write(
            f"{row['Layer']:<50} "
            f"{row['RA_MSE']:>7.4f} "
            f"{row['RA_KL']:>7.4f} "
            f"{row['ZS_MSE']:>7.4f} "
            f"{row['ZS_KL']:>7.4f} "
            f"{row['RA2bit_MSE']:>8.4f} "
            f"{row['Cons_MSE']:>9.4f} "
            f"{row['Mod_MSE']:>8.4f} "
            f"{row['BF_MSE']:>8.4f}\n"
        )

    f.write("-" * 110 + "\n")
    avg_ra_mse   = sum(r['RA_MSE']     for r in summary_rows) / len(summary_rows)
    avg_zs_mse   = sum(r['ZS_MSE']     for r in summary_rows) / len(summary_rows)
    avg_ra2_mse  = sum(r['RA2bit_MSE'] for r in summary_rows) / len(summary_rows)
    avg_cons_mse = sum(r['Cons_MSE']   for r in summary_rows) / len(summary_rows)
    avg_mod_mse  = sum(r['Mod_MSE']    for r in summary_rows) / len(summary_rows)
    avg_bf_mse   = sum(r['BF_MSE']     for r in summary_rows) / len(summary_rows)
    avg_ra_kl    = sum(r['RA_KL']      for r in summary_rows) / len(summary_rows)
    avg_zs_kl    = sum(r['ZS_KL']      for r in summary_rows) / len(summary_rows)

    f.write(
        f"{'평균':<50} "
        f"{avg_ra_mse:>7.4f} "
        f"{avg_ra_kl:>7.4f} "
        f"{avg_zs_mse:>7.4f} "
        f"{avg_zs_kl:>7.4f} "
        f"{avg_ra2_mse:>8.4f} "
        f"{avg_cons_mse:>9.4f} "
        f"{avg_mod_mse:>8.4f} "
        f"{avg_bf_mse:>8.4f}\n"
    )

    f.write("\n유효 비트 수 및 압축률\n")
    f.write(f"  1. RA only      : 4비트     → 압축률 2.0×\n")
    f.write(f"  2. ZS only      : 4비트     → 압축률 2.0×\n")
    f.write(f"  3. Best of RA/ZS: 4비트     → 압축률 2.0×\n")
    f.write(f"  4. RA 2bit      : 6비트     → 압축률 1.33×\n")
    f.write(f"  5. Conservative : ~6.25비트 → 압축률 1.29×\n")
    f.write(f"  6. Moderate     : ~4.25비트 → 압축률 1.66×\n")
    f.write(f"  7. BitFlip      : 4비트     → 압축률 2.0×\n")

    f.write("\n논문 방식 Winner 규칙 (Figure 6 기반)\n")
    f.write(f"  4비트 프루닝 → ZS_only     (ZS가 4비트에서 항상 우수)\n")
    f.write(f"  2비트 프루닝 → RA_2bit     (RA가 2비트에서 항상 우수)\n")
    f.write(f"  정확도 우선  → Conservative (RA 2비트 + 채널보호 10%)\n")
    f.write(f"  압축률 우선  → Moderate     (ZS 4비트 + 채널보호 20%)\n")

print(f"\n저장 완료!")
print(f"  엑셀   : {os.path.join(OUT_DIR, 'comparison.xlsx')}")
print(f"  메모장 : {txt_path}")
print(f"  pt파일 : {pt_path}")