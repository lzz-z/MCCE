import json
import os
import sys
import argparse
import subprocess
import shlex
import pickle
import numpy as np
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 确保可以 import 到 MCCE 的模块（algorithm, model 等），
# 这样在 unpickle 时不会因为找不到 algorithm 而报错。
MCCE_ROOT = "/home/lzz/MCCE"
if MCCE_ROOT not in sys.path:
    sys.path.append(MCCE_ROOT)

# 全局缓存（进程内生效）
_embedding_cache = {}

try:
    from algorithm.base import Item  # noqa: F401
except (ImportError, ModuleNotFoundError):
    class Item:
        def __init__(self, value, property_dict):
            self.value = value
            self.property = property_dict
            self.total = 0.0


def _parse_circle_solution(sol_str):
    """
    将 circle_packing 的解字符串解析为 (centers, radii)，并展平成向量作为 embedding。
    这里不依赖任何外部模型，仅使用几何参数作为“embedding”。
    """
    if not isinstance(sol_str, str) or "centers" not in sol_str or "radii" not in sol_str:
        return None

    local_vars = {}
    try:
        # 执行由 MCCE 生成的 centers/radii 代码字符串
        exec(sol_str, {"np": np}, local_vars)
        centers = local_vars.get("centers", None)
        radii = local_vars.get("radii", None)
        if centers is None or radii is None:
            return None
        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float)
        # 展平并拼接为一个向量
        emb = np.concatenate([centers.flatten(), radii.flatten()]).astype(np.float32)
        # L2 归一化，避免尺度影响
        norm = np.linalg.norm(emb)
        if norm == 0 or not np.isfinite(norm):
            return None
        emb = emb / norm
        return emb
    except Exception as e:
        print(f"[WARN] 解析 circle_packing 解失败，返回 None: {e}")
        return None


def get_embedding(sol_str):
    """
    获取 circle_packing 解的 embedding（带简单缓存）。
    """
    global _embedding_cache
    if sol_str in _embedding_cache:
        return _embedding_cache[sol_str]
    emb = _parse_circle_solution(sol_str)
    _embedding_cache[sol_str] = emb
    return emb


def calculate_similarity(sol1, sol2):
    """
    基于 embedding 的相似度（余弦相似度）。
    sol1/sol2 为 circle_packing 的解字符串（包含 centers/radii）。
    """
    emb1 = get_embedding(sol1)
    emb2 = get_embedding(sol2)
    if emb1 is None or emb2 is None:
        return 0.0
    # 由于已做过 L2 归一化，内积即为余弦相似度
    sim = float(np.dot(emb1, emb2))
    # 数值安全：限制在 [-1, 1] 区间
    if not np.isfinite(sim):
        return 0.0
    sim = max(min(sim, 1.0), -1.0)
    # 将 [-1,1] 映射到 [0,1]，避免负值
    return (sim + 1.0) / 2.0


def process_one_prompt(idx, query, high_score_pool_solutions, high_score_extended_pool_solutions,
                       low_score_pool_solutions, solution_to_total):
    """
    子进程执行：为单个 prompt 选择相似解并计算相似度（基于 embedding）。
    """
    prompt_text = query.get('prompt', '')
    parents = query.get('parents', [])

    chosen_solutions = find_similar_solutions(
        parents, high_score_pool_solutions,
        similarity_thresholds=[0.7, 0.6, 0.5],
        min_threshold=0.3,
        max_similarity=0.95,
        show_progress=False
    )
    if len(chosen_solutions) < 2:
        chosen_solutions = find_similar_solutions(
            parents, high_score_extended_pool_solutions,
            similarity_thresholds=[0.7, 0.6, 0.5],
            min_threshold=0.3,
            max_similarity=0.95,
            show_progress=False
        )

    rejected_solutions = find_similar_solutions(
        parents, low_score_pool_solutions,
        similarity_thresholds=[0.7, 0.6, 0.5],
        min_threshold=0.0,
        max_similarity=0.95,
        show_progress=False
    )

    # 补足数量
    while len(chosen_solutions) < 2 and len(high_score_pool_solutions) > 0:
        chosen_solutions.append(high_score_pool_solutions[np.random.randint(len(high_score_pool_solutions))][0])
    while len(rejected_solutions) < 2 and len(low_score_pool_solutions) > 0:
        rejected_solutions.append(low_score_pool_solutions[np.random.randint(len(low_score_pool_solutions))][0])

    # 与统一格式对齐：使用 <candidate> 标签包裹候选解
    chosen_response = f"<candidate>{chosen_solutions[0]}</candidate>\n<candidate>{chosen_solutions[1]}</candidate>"
    rejected_response = f"<candidate>{rejected_solutions[0]}</candidate>\n<candidate>{rejected_solutions[1]}</candidate>"

    dpo_item = {
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response
    }

    # 详细数据：记录与前两个父代之间的相似度
    parent_values = [parent.get('value', '') for parent in parents if parent.get('value')]
    similarities = {}
    for p_idx, parent_val in enumerate(parent_values[:2]):
        if parent_val:
            for c_idx, chosen in enumerate(chosen_solutions[:2]):
                try:
                    sim = calculate_similarity(parent_val, chosen)
                    similarities[f"similar_{p_idx+1}_c{c_idx+1}"] = sim
                except Exception:
                    similarities[f"similar_{p_idx+1}_c{c_idx+1}"] = 0.0
            for r_idx, rejected in enumerate(rejected_solutions[:2]):
                try:
                    sim = calculate_similarity(parent_val, rejected)
                    similarities[f"similar_{p_idx+1}_r{r_idx+1}"] = sim
                except Exception:
                    similarities[f"similar_{p_idx+1}_r{r_idx+1}"] = 0.0

    # 填充缺失字段
    for p in range(1, 3):
        for t in ['c', 'r']:
            for m in range(1, 2 + 1):
                key = f"similar_{p}_{t}{m}"
                if key not in similarities:
                    similarities[key] = 0.0

    parent_data = []
    for idx_p in range(min(2, len(parents))):
        parent = parents[idx_p]
        parent_data.append({
            "solution": parent.get('value', ''),
            "total": parent.get('total', 0.0)
        })
    while len(parent_data) < 2:
        parent_data.append({"solution": "", "total": 0.0})

    chosen_totals = [solution_to_total.get(sol, 0.0) for sol in chosen_solutions]
    rejected_totals = [solution_to_total.get(sol, 0.0) for sol in rejected_solutions]

    full_item = {
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "parents": parent_data,
        "chosen_solutions": [
            {
                "solution": chosen_solutions[idx_c] if idx_c < len(chosen_solutions) else '',
                "total": chosen_totals[idx_c] if idx_c < len(chosen_totals) else 0.0
            }
            for idx_c in range(2)
        ],
        "rejected_solutions": [
            {
                "solution": rejected_solutions[idx_r] if idx_r < len(rejected_solutions) else '',
                "total": rejected_totals[idx_r] if idx_r < len(rejected_totals) else 0.0
            }
            for idx_r in range(2)
        ]
    }
    full_item.update(similarities)
    return idx, dpo_item, full_item


def load_prompt_history(pkl_path):
    """从 pkl 文件中加载 prompt 历史记录"""
    pkl_dir = os.path.dirname(pkl_path)
    pkl_basename = os.path.basename(pkl_path).replace('.pkl', '')

    parent_dir = os.path.dirname(pkl_dir)
    prompt_dir = os.path.join(parent_dir, 'prompt')

    if os.path.exists(prompt_dir):
        for filename in os.listdir(prompt_dir):
            if filename.endswith('_prompt.json') and pkl_basename in filename:
                prompt_file_path = os.path.join(prompt_dir, filename)
                try:
                    with open(prompt_file_path, 'r', encoding='utf-8') as f:
                        prompt_data = json.load(f)
                    return prompt_data.get('queries', [])
                except Exception as e:
                    print(f"Failed to load prompt file {prompt_file_path}: {e}")
                    break

    print("No prompt history found, using default empty list")
    return []


def find_similar_solutions(target_parents, sol_pool,
                           similarity_thresholds=[0.7, 0.6, 0.5],
                           min_threshold=0.3,
                           max_similarity=0.95,
                           show_progress=True):
    """
    在解空间池中寻找与 target_parents 相似的解（基于 embedding 相似度）。

    Args:
        target_parents: 目标父代解列表（list of dict, 含 'value' 字段）
        sol_pool: 候选解池 [(solution_str, index), ...]
        similarity_thresholds: 相似度阈值列表，从高到低
        min_threshold: 最低相似度阈值，如果都找不到就选择相似度最高的
        max_similarity: 最高相似度阈值，超过此值的解将被排除

    Returns:
        选中的两个解的字符串列表
    """
    if not target_parents or not sol_pool:
        # 如果没有父代信息或候选池为空，随机选择两个解
        selected = np.random.choice(len(sol_pool), min(2, len(sol_pool)), replace=False)
        sols = [sol_pool[i][0] for i in selected]
        return sols

    parent_values = [parent.get('value', '') for parent in target_parents if parent.get('value')]
    if not parent_values:
        selected = np.random.choice(len(sol_pool), min(2, len(sol_pool)), replace=False)
        sols = [sol_pool[i][0] for i in selected]
        return sols

    selected_solutions = []
    used_indices = set()

    # 尝试不同的相似度阈值
    for threshold in similarity_thresholds:
        if len(selected_solutions) >= 2:
            break

        iterator = tqdm(sol_pool, desc=f"相似度筛选(阈值={threshold:.1f})", leave=False) if show_progress else sol_pool
        for sol_str, sol_idx in iterator:
            if sol_idx in used_indices or len(selected_solutions) >= 2:
                continue

            max_similarity_score = 0.0
            for parent_val in parent_values:
                try:
                    similarity = calculate_similarity(parent_val, sol_str)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue

            # 添加相似度上限限制：相似度必须在 threshold 和 max_similarity 之间
            if threshold <= max_similarity_score <= max_similarity:
                selected_solutions.append(sol_str)
                used_indices.add(sol_idx)

        if len(selected_solutions) >= 2:
            break

    # 如果仍然找不到足够的解，选择相似度最高但不超过上限的解
    if len(selected_solutions) < 2:
        similarities = []
        iterator = tqdm(sol_pool, desc="寻找最高相似度备选", leave=False) if show_progress else sol_pool
        for sol_str, sol_idx in iterator:
            if sol_idx in used_indices:
                continue

            max_similarity_score = 0.0
            for parent_val in parent_values:
                try:
                    similarity = calculate_similarity(parent_val, sol_str)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception:
                    continue

            if max_similarity_score <= max_similarity:
                similarities.append((max_similarity_score, sol_str, sol_idx))

        similarities.sort(key=lambda x: x[0], reverse=True)
        needed = 2 - len(selected_solutions)
        for i in range(min(needed, len(similarities))):
            selected_solutions.append(similarities[i][1])

    # 如果还是不够，随机补充（但仍要满足相似度上限）
    while len(selected_solutions) < 2 and len(sol_pool) > len(selected_solutions):
        available_candidates = []
        iterator = tqdm(sol_pool, desc="补充候选筛选", leave=False) if show_progress else enumerate(sol_pool)
        if show_progress:
            iterable = enumerate(iterator)
        else:
            iterable = iterator
        for i, (sol_str, sol_idx) in iterable:
            if sol_idx in used_indices:
                continue

            max_similarity_score = 0.0
            for parent_val in parent_values:
                try:
                    similarity = calculate_similarity(parent_val, sol_str)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception:
                    continue

            if max_similarity_score <= max_similarity:
                available_candidates.append(i)

        if available_candidates:
            selected_idx = np.random.choice(available_candidates)
            sol_str = sol_pool[selected_idx][0]
            selected_solutions.append(sol_str)
            used_indices.add(sol_pool[selected_idx][1])
        else:
            break

    return selected_solutions[:2]


def create_dpo_data_from_pkl_v2(pkl_path, output_json_path, num_pairs=None):
    """
    从 pkl 文件创建 DPO 数据集，使用基于 embedding 的相似度进行解的选择。
    该脚本专为非分子任务（如 circle_packing）设计。
    """
    print(f"Loading optimization data from {pkl_path}")
    with open(pkl_path, "rb") as fin:
        data = pickle.load(fin)

    all_items = data.get("all_mols", [])
    if not all_items:
        print("No candidates found in the pkl file.")
        return 0

    # 按照总分排序（从高到低）
    sorted_items = sorted(all_items, key=lambda x: x[0].total, reverse=True)
    print(f"Found {len(sorted_items)} candidates, sorted by total score")

    # 加载 prompt 历史记录
    prompt_history = load_prompt_history(pkl_path)
    print(f"Loaded {len(prompt_history)} historical prompts")

    # 限制 prompt 数量，取最新的 128 条
    if len(prompt_history) > 128:
        prompt_history = prompt_history[-128:]
        print(f"Limited to latest {len(prompt_history)} prompts")

    # 计算采样数量
    if num_pairs is None:
        num_pairs = min(len(prompt_history), data['evaluation'][-1]['all_unique_moles'] // 2)

    if num_pairs == 0:
        print("No prompts available for generating DPO data.")
        return 0

    # 定义候选解池范围：前 30% 作为高分区间，后 30% 作为低分区间
    total_items = len(sorted_items)
    high_score_pool_size = int(total_items * 0.3)
    low_score_pool_size = int(total_items * 0.3)

    high_score_pool = [(sorted_items[i][0], i) for i in range(high_score_pool_size)]
    low_score_pool = [(sorted_items[i][0], i) for i in range(total_items - low_score_pool_size, total_items)]

    # 扩展候选池（如果需要的话）：前 50%
    high_score_extended_pool = [(sorted_items[i][0], i) for i in range(int(total_items * 0.5))]

    print(f"High score pool: {len(high_score_pool)} candidates")
    print(f"Low score pool: {len(low_score_pool)} candidates")

    # 预处理：将候选池转换为仅含字符串表示的轻量结构
    high_score_pool_solutions = [(item.value, idx) for item, idx in high_score_pool]
    low_score_pool_solutions = [(item.value, idx) for item, idx in low_score_pool]
    high_score_extended_pool_solutions = [(item.value, idx) for item, idx in high_score_extended_pool]

    # 构建 solution -> total 的映射，以便在子进程查询
    solution_to_total = {}
    for item, _ in sorted_items:
        solution_to_total[item.value] = item.total

    # 并行处理每个 prompt
    dpo_data = [None] * min(num_pairs, len(prompt_history))
    full_data = [None] * min(num_pairs, len(prompt_history))

    total_pairs = min(num_pairs, len(prompt_history))
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [
            executor.submit(
                process_one_prompt,
                i,
                prompt_history[i],
                high_score_pool_solutions,
                high_score_extended_pool_solutions,
                low_score_pool_solutions,
                solution_to_total,
            )
            for i in range(total_pairs)
        ]
        for fut in tqdm(as_completed(futures), total=total_pairs, desc="并行生成 DPO 数据对（embedding 相似度）", unit="pair"):
            idx, dpo_item, full_item = fut.result()
            dpo_data[idx] = dpo_item
            full_data[idx] = full_item

    # 保存 DPO 数据结果
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as fout:
        json.dump(dpo_data, fout, indent=2, ensure_ascii=False)

    # 保存完整数据
    # 使用项目内的相对路径
    MCCE_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fulldata_dir = os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_training", "fulldata")
    os.makedirs(fulldata_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(output_json_path))[0]
    fulldata_path = os.path.join(fulldata_dir, f"{base_filename}_full.json")

    with open(fulldata_path, "w") as fout:
        json.dump(full_data, fout, indent=2, ensure_ascii=False)

    print(f"DPO 数据集构造完成，共生成 {len(dpo_data)} 条数据对，已保存到 {output_json_path}")
    print(f"完整数据已保存到 {fulldata_path}")
    print("Chosen 解来自高分区间，Rejected 解来自低分区间，基于与 parents 的 embedding 相似度选择")
    return len(dpo_data)


def main():
    # 获取MCCE项目根目录
    MCCE_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(
        description="Process circle_packing (or other non-molecular) data for DPO training using embedding-based similarity."
    )
    parser.add_argument("--exp", required=True, help="Experiment name, used for output files and directories.")
    parser.add_argument("--pkl_path", required=True, help="Path to the input pkl file containing optimization data.")
    parser.add_argument("--data_dir", default=os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_training"), 
                       help="Directory to save training data (JSON).")
    parser.add_argument("--model_dir", default=os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_models"), 
                       help="Directory to save trained models.")
    parser.add_argument("--prev_exp", help="Experiment name of the previous run to use as a base model.")
    parser.add_argument("--num_pairs", type=int, help="Number of data pairs to generate (optional).")
    parser.add_argument(
        "--ref_model_path",
        default="/home/lzz/models/Qwen/Qwen2.5-7B-Instruct",
        help="Reference model path (always the original base model)",
    )

    args = parser.parse_args()

    output_json_path = os.path.join(args.data_dir, f"{args.exp}.json")
    model_output_path = os.path.join(args.model_dir, args.exp)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Creating DPO dataset (embedding similarity) for experiment: {args.exp}")
    num_data_pairs = create_dpo_data_from_pkl_v2(args.pkl_path, output_json_path, args.num_pairs)

    if num_data_pairs == 0:
        print("No data pairs generated. Exiting.")
        return

    # 2. 运行 DPO 训练脚本（使用项目内的训练脚本）
    training_script = os.path.join(MCCE_PROJECT_ROOT, "training", "train_dpo.py")
    print(f"Starting DPO training for experiment: {args.exp}")

    command_list = [
        "python",
        training_script,
        "--train_data_path",
        output_json_path,
        "--output_dir",
        model_output_path,
        "--exp_name",
        args.exp,
        "--ref_model_path",
        args.ref_model_path,
    ]

    if args.prev_exp:
        prev_model_path = os.path.join(args.model_dir, args.prev_exp)
        if os.path.exists(prev_model_path):
            command_list.extend(["--model_name_or_path", prev_model_path])
            print(f"Using policy model from previous experiment: {prev_model_path}")
            print(f"Using reference model (fixed): {args.ref_model_path}")
        else:
            print(f"Warning: Previous model path {prev_model_path} does not exist. Using default base model.")
            print(f"Using reference model (fixed): {args.ref_model_path}")
    else:
        print("Using default base model for policy model")
        print(f"Using reference model (fixed): {args.ref_model_path}")

    command_str = " ".join(shlex.quote(c) for c in command_list)
    # 使用完整的conda初始化路径以支持subprocess调用
    final_command = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate verl && {command_str}"

    print("Running command:")
    print(f"cd {MCCE_PROJECT_ROOT} && {final_command}")

    start_time = time.time()

    try:
        subprocess.run(
            final_command,
            check=True,
            cwd=MCCE_PROJECT_ROOT,
            shell=True,
            executable="/bin/bash",
        )

        end_time = time.time()
        duration = (end_time - start_time) / 60
        print(f"DPO training script finished successfully in {duration:.2f} minutes.")
        print(f"Trained model saved to: {model_output_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error running DPO training script: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during DPO training: {e}")
        raise


if __name__ == "__main__":
    main()


