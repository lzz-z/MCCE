import json
import pandas as pd
import os
import sys
import argparse
import subprocess
import shlex
import pickle
import numpy as np
import torch
import time
from tdc import Oracle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 确保可以 import 到 MCCE 的模块（algorithm, model 等），
# 这样在 unpickle 时不会因为找不到 algorithm 而报错。
MCCE_ROOT = "/home/lzz/MCCE"
if MCCE_ROOT not in sys.path:
    sys.path.append(MCCE_ROOT)

# 全局缓存（进程内生效）
_similarity_cache = {}
_oracle_cache = {}

try:
    from algorithm.base import Item
except (ImportError, ModuleNotFoundError):
    class Item:
        def __init__(self, value, property_dict):
            self.value = value
            self.property = property_dict
            self.total = 0.0

def calculate_similarity(smiles1, smiles2):
    """计算两个分子的相似度（带简单缓存，进程内）"""
    global _similarity_cache, _oracle_cache
    key = (smiles1, smiles2)
    if key in _similarity_cache:
        return _similarity_cache[key]
    oracle = _oracle_cache.get(smiles1)
    if oracle is None:
        oracle = Oracle(name='Similarity_Meta', target_smiles=smiles1)
        _oracle_cache[smiles1] = oracle
    try:
        similarity = oracle([smiles2])[0]
    except Exception:
        similarity = 0.0
    _similarity_cache[key] = similarity
    return similarity

def process_one_prompt(idx, query, high_score_pool_smiles, high_score_extended_pool_smiles, low_score_pool_smiles, smiles_to_total):
    """子进程执行：为单个prompt选择相似分子并计算相似度"""
    prompt_text = query.get('prompt', '')
    parents = query.get('parents', [])

    chosen_molecules = find_similar_molecules(
        parents, high_score_pool_smiles,
        similarity_thresholds=[0.7, 0.6, 0.5],
        min_threshold=0.3,
        max_similarity=0.8,
        show_progress=False
    )
    if len(chosen_molecules) < 2:
        chosen_molecules = find_similar_molecules(
            parents, high_score_extended_pool_smiles,
            similarity_thresholds=[0.7, 0.6, 0.5],
            min_threshold=0.3,
            max_similarity=0.8,
            show_progress=False
        )

    rejected_molecules = find_similar_molecules(
        parents, low_score_pool_smiles,
        similarity_thresholds=[0.7, 0.6, 0.5],
        min_threshold=0.0,
        max_similarity=0.8,
        show_progress=False
    )

    # 补足数量
    while len(chosen_molecules) < 2 and len(high_score_pool_smiles) > 0:
        chosen_molecules.append(high_score_pool_smiles[np.random.randint(len(high_score_pool_smiles))][0])
    while len(rejected_molecules) < 2 and len(low_score_pool_smiles) > 0:
        rejected_molecules.append(low_score_pool_smiles[np.random.randint(len(low_score_pool_smiles))][0])

    # 与统一格式对齐：使用 <candidate> 标签包裹分子或代码候选
    chosen_response = f"<candidate>{chosen_molecules[0]}</candidate>\n<candidate>{chosen_molecules[1]}</candidate>"
    rejected_response = f"<candidate>{rejected_molecules[0]}</candidate>\n<candidate>{rejected_molecules[1]}</candidate>"

    dpo_item = {
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response
    }

    # 详细数据
    parent_smiles = [parent.get('value', '') for parent in parents if parent.get('value')]
    similarities = {}
    for p_idx, parent_smiles_str in enumerate(parent_smiles[:2]):
        if parent_smiles_str:
            for c_idx, chosen_mol in enumerate(chosen_molecules[:2]):
                try:
                    sim = calculate_similarity(parent_smiles_str, chosen_mol)
                    similarities[f"similar_{p_idx+1}_c{c_idx+1}"] = sim
                except Exception:
                    similarities[f"similar_{p_idx+1}_c{c_idx+1}"] = 0.0
            for r_idx, rejected_mol in enumerate(rejected_molecules[:2]):
                try:
                    sim = calculate_similarity(parent_smiles_str, rejected_mol)
                    similarities[f"similar_{p_idx+1}_r{r_idx+1}"] = sim
                except Exception:
                    similarities[f"similar_{p_idx+1}_r{r_idx+1}"] = 0.0

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
            "smiles": parent.get('value', ''),
            "total": parent.get('total', 0.0)
        })
    while len(parent_data) < 2:
        parent_data.append({"smiles": "", "total": 0.0})

    chosen_totals = [smiles_to_total.get(sm, 0.0) for sm in chosen_molecules]
    rejected_totals = [smiles_to_total.get(sm, 0.0) for sm in rejected_molecules]

    full_item = {
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "parents": parent_data,
        "chosen_molecules": [
            {"smiles": chosen_molecules[idx_c] if idx_c < len(chosen_molecules) else '',
             "total": chosen_totals[idx_c] if idx_c < len(chosen_totals) else 0.0}
            for idx_c in range(2)
        ],
        "rejected_molecules": [
            {"smiles": rejected_molecules[idx_r] if idx_r < len(rejected_molecules) else '',
             "total": rejected_totals[idx_r] if idx_r < len(rejected_totals) else 0.0}
            for idx_r in range(2)
        ]
    }
    full_item.update(similarities)
    return idx, dpo_item, full_item

def load_prompt_history(pkl_path):
    """从pkl文件中加载prompt历史记录"""
    # 首先尝试从pkl文件相同目录下找prompt json文件
    pkl_dir = os.path.dirname(pkl_path)
    pkl_basename = os.path.basename(pkl_path).replace('.pkl', '')
    
    # 在prompt目录下寻找对应的prompt json文件
    parent_dir = os.path.dirname(pkl_dir)
    prompt_dir = os.path.join(parent_dir, 'prompt')
    
    if os.path.exists(prompt_dir):
        # 寻找匹配的prompt文件
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

def find_similar_molecules(target_parents, mol_pool, similarity_thresholds=[0.7, 0.6, 0.5], min_threshold=0.3, max_similarity=0.8, show_progress=True):
    """
    在分子池中寻找与target_parents相似的分子
    
    Args:
        target_parents: 目标父代分子列表
        mol_pool: 候选分子池 [(mol_item, index), ...]
        similarity_thresholds: 相似度阈值列表，从高到低
        min_threshold: 最低相似度阈值，如果都找不到就选择相似度最高的
        max_similarity: 最高相似度阈值，超过此值的分子将被排除
    
    Returns:
        选中的两个分子的SMILES列表
    """
    if not target_parents or not mol_pool:
        # 如果没有父代信息或分子池为空，随机选择两个分子
        selected = np.random.choice(len(mol_pool), min(2, len(mol_pool)), replace=False)
        smiles_list = []
        for i in selected:
            mol_item = mol_pool[i][0]
            mol_smiles = mol_item.value if hasattr(mol_item, "value") else mol_item
            smiles_list.append(mol_smiles)
        return smiles_list
    
    # 提取父代分子的SMILES
    parent_smiles = [parent.get('value', '') for parent in target_parents if parent.get('value')]
    if not parent_smiles:
        # 如果没有有效的父代SMILES，随机选择
        selected = np.random.choice(len(mol_pool), min(2, len(mol_pool)), replace=False)
        smiles_list = []
        for i in selected:
            mol_item = mol_pool[i][0]
            mol_smiles = mol_item.value if hasattr(mol_item, "value") else mol_item
            smiles_list.append(mol_smiles)
        return smiles_list
    
    selected_molecules = []
    used_indices = set()
    
    # 尝试不同的相似度阈值
    for threshold in similarity_thresholds:
        if len(selected_molecules) >= 2:
            break
            
        iterator = tqdm(mol_pool, desc=f"相似度筛选(阈值={threshold:.1f})", leave=False) if show_progress else mol_pool
        for mol_item, mol_idx in iterator:
            if mol_idx in used_indices or len(selected_molecules) >= 2:
                continue
                
            mol_smiles = mol_item.value if hasattr(mol_item, 'value') else mol_item
            # 计算与任意父代分子的最大相似度
            max_similarity_score = 0.0
            for parent_smiles_str in parent_smiles:
                try:
                    similarity = calculate_similarity(parent_smiles_str, mol_smiles)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue
            
            # 添加相似度上限限制：相似度必须在threshold和max_similarity之间
            if threshold <= max_similarity_score <= max_similarity:
                selected_molecules.append(mol_smiles)
                used_indices.add(mol_idx)
        
        if len(selected_molecules) >= 2:
            break
    
    # 如果仍然找不到足够的分子，选择相似度最高但不超过上限的分子
    if len(selected_molecules) < 2:
        similarities = []
        iterator = tqdm(mol_pool, desc="寻找最高相似度备选", leave=False) if show_progress else mol_pool
        for mol_item, mol_idx in iterator:
            if mol_idx in used_indices:
                continue
                
            mol_smiles = mol_item.value if hasattr(mol_item, 'value') else mol_item
            max_similarity_score = 0.0
            for parent_smiles_str in parent_smiles:
                try:
                    similarity = calculate_similarity(parent_smiles_str, mol_smiles)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception as e:
                    continue
            
            # 只考虑相似度不超过上限的分子
            if max_similarity_score <= max_similarity:
                similarities.append((max_similarity_score, mol_smiles, mol_idx))
        
        # 按相似度排序，选择最高的
        similarities.sort(key=lambda x: x[0], reverse=True)
        needed = 2 - len(selected_molecules)
        for i in range(min(needed, len(similarities))):
            selected_molecules.append(similarities[i][1])
    
    # 如果还是不够，随机补充（但仍要满足相似度上限）
    while len(selected_molecules) < 2 and len(mol_pool) > len(selected_molecules):
        available_candidates = []
        iterator = tqdm(mol_pool, desc="补充候选筛选", leave=False) if show_progress else enumerate(mol_pool)
        # 由于上面用到了 enumerate，这里在非进度条时也需要enumerate
        if show_progress:
            iterable = enumerate(iterator)
        else:
            iterable = iterator
        for i, (mol_item, mol_idx) in iterable:
            if mol_idx in used_indices:
                continue
            
            mol_smiles = mol_item.value if hasattr(mol_item, 'value') else mol_item
            max_similarity_score = 0.0
            for parent_smiles_str in parent_smiles:
                try:
                    similarity = calculate_similarity(parent_smiles_str, mol_smiles)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception as e:
                    continue
            
            # 只添加相似度不超过上限的候选分子
            if max_similarity_score <= max_similarity:
                available_candidates.append(i)
        
        if available_candidates:
            selected_idx = np.random.choice(available_candidates)
            mol_item = mol_pool[selected_idx][0]
            mol_smiles = mol_item.value if hasattr(mol_item, "value") else mol_item
            selected_molecules.append(mol_smiles)
            used_indices.add(mol_pool[selected_idx][1])
        else:
            # 如果没有满足条件的分子，随机选择
            available_indices = [i for i, (_, mol_idx) in enumerate(mol_pool) if mol_idx not in used_indices]
            if available_indices:
                selected_idx = np.random.choice(available_indices)
                mol_item = mol_pool[selected_idx][0]
                mol_smiles = mol_item.value if hasattr(mol_item, "value") else mol_item
                selected_molecules.append(mol_smiles)
                used_indices.add(mol_pool[selected_idx][1])
            else:
                break
    
    return selected_molecules[:2]

def find_molecule_total_score(smiles, sorted_mols):
    """从sorted_mols中找到分子的total得分"""
    for mol_item, _ in sorted_mols:
        if mol_item.value == smiles:
            return mol_item.total
    return 0.0

def create_dpo_data_from_pkl_v2(pkl_path, output_json_path, num_pairs=None):
    """
    从pkl文件创建DPO数据集，使用基于相似度的分子选择方法
    
    Args:
        pkl_path: 包含分子数据的pkl文件路径
        output_json_path: 输出DPO JSON文件路径
        num_pairs: 生成的数据对数量，如果为None则自动计算
    """
    print(f"Loading molecular data from {pkl_path}")
    with open(pkl_path, "rb") as fin:
        data = pickle.load(fin)
    
    all_mols = data.get("all_mols", [])
    if not all_mols:
        print("No molecules found in the pkl file.")
        return 0
        
    # 按照总分排序（从高到低）
    sorted_mols = sorted(all_mols, key=lambda x: x[0].total, reverse=True)
    print(f"Found {len(sorted_mols)} molecules, sorted by total score")
    
    # 加载prompt历史记录
    prompt_history = load_prompt_history(pkl_path)
    print(f"Loaded {len(prompt_history)} historical prompts")
    
    # 限制prompt数量，取最新的128条
    if len(prompt_history) > 128:
        prompt_history = prompt_history[-128:]
        print(f"Limited to latest {len(prompt_history)} prompts")
    
    # 计算采样数量
    if num_pairs is None:
        num_pairs = min(len(prompt_history), data['evaluation'][-1]['all_unique_moles'] // 2)
    
    if num_pairs == 0:
        print("No prompts available for generating DPO data.")
        return 0
    
    # 定义分子池范围
    total_mols = len(sorted_mols)
    high_score_pool_size = int(total_mols * 0.3)  # 前30%
    low_score_pool_size = int(total_mols * 0.3)   # 后30%
    
    high_score_pool = [(sorted_mols[i][0], i) for i in range(high_score_pool_size)]
    low_score_pool = [(sorted_mols[i][0], i) for i in range(total_mols - low_score_pool_size, total_mols)]
    
    # 扩展候选池（如果需要的话）
    high_score_extended_pool = [(sorted_mols[i][0], i) for i in range(int(total_mols * 0.5))]
    
    print(f"High score pool: {len(high_score_pool)} molecules")
    print(f"Low score pool: {len(low_score_pool)} molecules")
    
    # 预处理：将分子池转换为仅含SMILES的轻量结构，构建查询列表
    high_score_pool_smiles = [(mol_item.value, idx) for mol_item, idx in high_score_pool]
    low_score_pool_smiles = [(mol_item.value, idx) for mol_item, idx in low_score_pool]
    high_score_extended_pool_smiles = [(mol_item.value, idx) for mol_item, idx in high_score_extended_pool]

    # 构建 SMILES -> total 的映射以便在子进程查询
    smiles_to_total = {}
    for mol_item, _ in sorted_mols:
        smiles_to_total[mol_item.value] = mol_item.total

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
                high_score_pool_smiles,
                high_score_extended_pool_smiles,
                low_score_pool_smiles,
                smiles_to_total,
            )
            for i in range(total_pairs)
        ]
        for fut in tqdm(as_completed(futures), total=total_pairs, desc="并行生成DPO数据对", unit="pair"):
            idx, dpo_item, full_item = fut.result()
            dpo_data[idx] = dpo_item
            full_data[idx] = full_item
    
    # 保存DPO数据结果
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as fout:
        json.dump(dpo_data, fout, indent=2, ensure_ascii=False)
    
    # 保存完整数据
    # 使用项目内的相对路径
    MCCE_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fulldata_dir = os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_training", "fulldata")
    os.makedirs(fulldata_dir, exist_ok=True)
    
    # 从output_json_path提取文件名
    base_filename = os.path.splitext(os.path.basename(output_json_path))[0]
    fulldata_path = os.path.join(fulldata_dir, f"{base_filename}_full.json")
    
    with open(fulldata_path, "w") as fout:
        json.dump(full_data, fout, indent=2, ensure_ascii=False)
    
    print(f"DPO数据集构造完成，共生成 {len(dpo_data)} 条数据对，已保存到 {output_json_path}")
    print(f"完整数据已保存到 {fulldata_path}")
    print(f"Chosen分子来自高分区间，Rejected分子来自低分区间，基于与parents的相似度选择")
    return len(dpo_data)

def main():
    # 获取MCCE项目根目录
    MCCE_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description="Process molecular data for DPO training and run the training script (v2 with similarity-based selection).")
    parser.add_argument("--exp", required=True, help="Experiment name, used for output files and directories.")
    parser.add_argument("--pkl_path", required=True, help="Path to the input pkl file containing molecular data.")
    parser.add_argument("--data_dir", default=os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_training"), 
                       help="Directory to save training data (JSON).")
    parser.add_argument("--model_dir", default=os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_models"), 
                       help="Directory to save trained models.")
    parser.add_argument("--prev_exp", help="Experiment name of the previous run to use as a base model.")
    parser.add_argument("--num_pairs", type=int, help="Number of data pairs to generate (optional).")
    parser.add_argument("--ref_model_path", default="/home/lzz/models/Qwen/Qwen2.5-7B-Instruct", 
                       help="Reference model path (always the original base model)")
    
    args = parser.parse_args()

    # Define output paths
    output_json_path = os.path.join(args.data_dir, f"{args.exp}.json")
    model_output_path = os.path.join(args.model_dir, args.exp)
    
    # Ensure directories exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # 1. Create DPO data from pkl file
    print(f"Creating DPO dataset for experiment: {args.exp}")
    num_data_pairs = create_dpo_data_from_pkl_v2(args.pkl_path, output_json_path, args.num_pairs)
    
    if num_data_pairs == 0:
        print("No data pairs generated. Exiting.")
        return

    # 2. Run DPO training script（使用项目内的训练脚本）
    training_script = os.path.join(MCCE_PROJECT_ROOT, "training", "train_dpo.py")
    print(f"Starting DPO training for experiment: {args.exp}")

    # Prepare training command
    command_list = [
        "python", 
        training_script,
        "--train_data_path", output_json_path,
        "--output_dir", model_output_path,
        "--exp_name", args.exp,
        "--ref_model_path", args.ref_model_path  # 始终指定参考模型路径
    ]
    
    # If there's a previous experiment, use it as the base model for policy model
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
        print(f"Using default base model for policy model")
        print(f"Using reference model (fixed): {args.ref_model_path}")
    
    # Execute training command
    command_str = " ".join(shlex.quote(c) for c in command_list)
    # 使用完整的conda初始化路径以支持subprocess调用
    final_command = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate verl && {command_str}"

    print("Running command:")
    print(f"cd {MCCE_PROJECT_ROOT} && {final_command}")

    # Record start time
    start_time = time.time()
    
    try:
        subprocess.run(
            final_command,
            check=True, 
            cwd=MCCE_PROJECT_ROOT,
            shell=True,
            executable="/bin/bash"
        )
        
        # Record end time and print duration
        end_time = time.time()
        duration = (end_time - start_time) / 60  # Convert to minutes
        print(f"DPO training script finished successfully in {duration:.2f} minutes.")
        print(f"Trained model saved to: {model_output_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running DPO training script: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during DPO training: {e}")
        raise

if __name__ == '__main__':
    main()
