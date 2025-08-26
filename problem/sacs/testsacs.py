import random
import numpy as np


def _parse_and_modify_line_v3(line):
    """这是我们上次尝试修复的版本 (V3)，我们将用它来复现问题。"""
    try:
        if line.startswith("GRUP"):
            part1 = line[0:18]
            od_val = float(line[18:24])
            separator = line[24]
            wt_val = float(line[25:30])
            rest_of_line = line[30:]

            if random.choice([True, False]):
                od_val *= random.uniform(0.95, 1.05)
                od_val = np.clip(od_val, 10.0, 48.0)
            else:
                wt_val *= random.uniform(0.95, 1.05)
                wt_val = np.clip(wt_val, 0.5, 2.5)

            return f"{part1}{od_val: >6.3f}{separator}{wt_val: >5.3f}{rest_of_line}"

        # 为了完整性，也包含PGRUP的逻辑
        elif line.startswith("PGRUP"):
            part1 = line[0:11]
            thick_str = line[11:17]
            rest_of_line = line[17:]
            thick_val = float(thick_str)
            thick_val *= random.uniform(0.95, 1.05)
            thick_val = np.clip(thick_val, 0.250, 0.750)
            return f"{part1}{thick_val:<6.4f}{rest_of_line}"

    except Exception as e:
        print(f"解析或重构时发生错误: {e}")
        return line
    return line


def _parse_and_modify_line_v4(line):
    """
    这是最终修复版 (V4)。此版本极其小心地处理每个部分，
    特别是通过重新构建`rest_of_line`来避免任何潜在的切片错误。
    """
    try:
        # 统一处理，无论是GRUP还是PGRUP
        keyword = line[0:5].strip()  # 'GRUP' or 'PGRUP'

        if keyword == "GRUP":
            # --- 极其严格和安全的解析 ---
            key_part = line[0:4]  # 'GRUP'
            group_part = line[5:18]  # ' LG4         '
            od_str = line[18:24]  # ' 42.000'
            wt_str = line[25:30]  # ' 1.375'
            # 将剩余部分定义为两个独立的块，避免中间的空格丢失
            middle_part = line[30:58]  # ' 29.0011.6050.00 1    1.001.00'
            end_part = line[58:]  # '     0.500N490.00'

            od_val = float(od_str)
            wt_val = float(wt_str)

            # --- 修改 ---
            if random.choice([True, False]):
                od_val *= random.uniform(0.95, 1.05)
                od_val = np.clip(od_val, 10.0, 48.0)
            else:
                wt_val *= random.uniform(0.95, 1.05)
                wt_val = np.clip(wt_val, 0.5, 2.5)

            # --- 极其严格和安全的重构 ---
            # 使用一个空格作为OD和WT之间的分隔符，而不是从原始字符串中提取
            new_line = f"{key_part} {group_part}{od_val:>6.3f} {wt_val:>5.3f}{middle_part}{end_part}"
            return new_line

        elif keyword == "PGRUP":
            # PGRUP的逻辑可以保持简单，因为它没出过问题
            part1 = line[0:11]
            thick_str = line[11:17]
            rest_of_line = line[17:]
            thick_val = float(thick_str)
            thick_val *= random.uniform(0.95, 1.05)
            thick_val = np.clip(thick_val, 0.250, 0.750)
            return f"{part1}{thick_val:<6.4f}{rest_of_line}"

    except Exception as e:
        print(f"解析或重构时发生错误: {e}")
        return line

    return line


# --- 测试 ---
print("--- 开始测试有问题的函数 (V3) ---")
# 这个是日志中出错的行
original_line_grup = "GRUP LG4         42.000 1.375 29.0011.6050.00 1    1.001.00     0.500N490.00"
for i in range(20):  # 运行20次看是否能复现
    modified = _parse_and_modify_line_v3(original_line_grup)
    print(f"V3 -原始: [{original_line_grup}] (len={len(original_line_grup)})")
    print(f"V3 -修改后: [{modified}] (len={len(modified)})")
    if len(modified) != len(original_line_grup):
        print("\n!!!!!! V3 发现长度不匹配错误 !!!!!!\n")

print("\n\n--- 开始测试最终修复版函数 (V4) ---")
for i in range(100):  # 用新函数运行更多次以确保其健壮性
    modified = _parse_and_modify_line_v4(original_line_grup)
    # print(f"V4 -原始: [{original_line_grup}] (len={len(original_line_grup)})")
    # print(f"V4 -修改后: [{modified}] (len={len(modified)})")
    if len(modified) != len(original_line_grup):
        print(f"\n!!!!!! V4 在第 {i + 1} 次迭代时发现长度不匹配错误 !!!!!!\n")
        break
else:
    print("V4 函数在100次测试中全部通过，长度始终保持一致。")

