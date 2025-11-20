import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import re
import copy
from tqdm import tqdm
# Set your OpenAI API key
import random
import torch
from functools import partial
import os
from algorithm.base import ItemFactory,HistoryBuffer
from openai import AzureOpenAI
from rdkit import Chem
import json
from eval import get_evaluation
import time
from model.util import *
from algorithm import PromptTemplate
import importlib
import pickle
import shutil
from model.LLM import LLM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MOO:
    def __init__(self, reward_system, llm, property_list, config, seed, llm2=None):
        self.reward_system = reward_system
        self.config = config
        self.seed = seed
        # 主模型与辅模型（保持与 MOLLM_1 协同接口一致）
        self.llm = llm
        self.llm1 = llm
        self.llm2 = llm2
        self.current_llm_flag = 0  # 0: 主模型, 1: 辅模型
        self.history = HistoryBuffer()
        self.item_factory = ItemFactory(property_list)
        self.property_list = property_list
        self.pop_size = self.config.get('optimization.pop_size')
        self.budget = self.config.get('optimization.eval_budget')
        # genetic_gfn (au) 已移除，保留 model_collaboration（MCCE Qwen协同）
        # self.use_au = self.config.get('use_au')
        self.save_dir = os.path.join(self.config.get('save_dir'),self.config.get('model.name'))
        self.prompt_module = getattr(PromptTemplate ,self.config.get('model.prompt_module',default='Prompt'))
        self.history_moles = []
        self.mol_buffer = []  # same as all_mols but with orders for computing auc
        # genetic_gfn buffers 已移除
        # self.main_mol_buffer = []
        # self.au_mol_buffer = []
        self.results_dict = {'results': []}
        self.history_experience = []
        self.repeat_num = 0
        self.failed_num = 0
        self.generated_num = 0
        self.llm_calls = 0
        self.start = True
        self.patience = 0
        self.old_score = 0
        self.early_stopping = False
        self.time_step = 0
        self.start_time = time.time()
        self.num_offspring = self.config.get('num_offspring', default=2)

        # 记录统计信息
        self.record_dict = {}
        # genetic_gfn 'au' 统计已移除，仅保留主模型统计
        for j in ['all_num', 'failed_num', 'repeat_num']:
            self.record_dict['main_' + j] = 0
        self.record_dict['main_history_smiles'] = []

        # ---------------- DPO / 日志 相关：对齐 MOLLM_1/algorithm/MOO.py ----------------
        # 只在需要协同 / DPO 时使用，不会影响原有单模型流程
        self._now_str = time.strftime('%m%d%H%M')
        # 统一实验标识：用于 results / prompt 等日志文件名
        self.experiment_tag = (
            f"{'_'.join(self.property_list)}_"
            f"{self.config.get('save_suffix')}_"
            f"{self.seed}_{self._now_str}"
        )
        self.prompt_records = {
            "queries": [],
            "init_pops": [],
            "final_pops": []
        }
        # DPO 训练状态
        self.previous_dpo_exp_name = None
        self.current_dpo_model_path = None
        self.original_ref_model_path = "/home/lzz/models/Qwen/Qwen2.5-7B-Instruct"
        # 模型轮次计数（与 MOLLM_1 一致）
        self.model_round_counter = 0
        self.llm1_rounds = 0
        self.llm2_rounds = 0

    def generate_initial_population(self, n):
        module_path = self.config.get('evalutor_path')  # e.g., "molecules"
        module = importlib.import_module(module_path)
        _generate = getattr(module, "generate_initial_population")
        strings = _generate(self.config,self.seed)
        if len(strings) == 0:
            print('generate init population return empty list, use llm to generate init pops')
            return []
        if isinstance(strings[0],str):
            return [self.item_factory.create(i) for i in strings]
        else:
            self.store_history_moles(strings)
            return strings
    def llm_init(self,generated):
        #a = 'start_time,end_time,combination_id\n2023-11-01T00:00:00,2023-11-01T08:00:00,1#&3#\n2023-11-01T08:00:00,2023-11-01T16:56:55.633423,3#\n'
        #generated = [self.item_factory.create(a)]
        
        
        
        prompt = self.prompt_generator.get_prompt('empty',None,None)
        log_gap = 10
        i=0
        while len(generated) <= self.pop_size:
            response = self.llm.chat(prompt)
            new_smiles = extract_smiles_from_string(response)
            new_smiles = [self.item_factory.create(smile) for smile in new_smiles]
            generated.extend(new_smiles)
            i+= 1
            if i%log_gap==0:
                print(f'{i} llm calls, init pop size [{len(generated)}/{self.pop_size}]')
        return generated
            
        
    def mutation(self, parent_list):
        prompt = self.prompt_generator.get_prompt('mutation',parent_list,self.history_moles)
        #print('mutation prompt:',prompt)
        #assert False
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def crossover(self, parent_list):
        prompt = self.prompt_generator.get_prompt('crossover',parent_list,self.history_moles)
        #print('crossover prompt:',prompt)
        #assert False
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        
        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def evaluate(self,pops_origin):
        pops = [i for i in pops_origin if i.total is None]
        already_evaluated = [i for i in pops_origin if i.total is not None]
        if len(pops) == 0:
            return pops_origin
        pops, log_dict = self.reward_system.evaluate(pops,self.mol_buffer)
        self.failed_num += log_dict['invalid_num']
        self.repeat_num += log_dict['repeated_num']
        pops.extend(already_evaluated)
        pops = self.store_history_moles(pops)
        return pops


    
    def store_history_moles(self,pops):
        unique_pop = []
        for i in pops:
            if i.value not in self.history_moles:
                self.history_moles.append(i.value)
                self.mol_buffer.append([i, len(self.mol_buffer)+1])
                unique_pop.append(i)
            else:
                self.repeat_num += 1
        return unique_pop
        
    def explore(self):
        pass 

    def log_results(self, 
                mol_buffer: list = None, 
                buffer_type: str = "default", 
                finish: bool = False) -> None:
        """
        Logs performance metrics (AUCs, top scores, diversity, hypervolume, etc.) 
        for the given molecular buffer, and saves them to JSON.

        Parameters:
        - mol_buffer (list): The molecular buffer to evaluate, list of (Molecule, Info) tuples.
        - buffer_type (str): The name of the buffer type ('main', 'au', or 'default' for mol_buffer).
        - finish (bool): Whether this is the final evaluation.
        """
        if mol_buffer is None:
            mol_buffer = self.mol_buffer
        auc1 = top_auc(mol_buffer, 1, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc10 = top_auc(mol_buffer, 10, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc100 = top_auc(mol_buffer, 100, finish=finish, freq_log=100, max_oracle_calls=self.budget)

        top100 = sorted(mol_buffer, key=lambda item: item[0].total, reverse=True)[:100]
        top100_mols = [i[0] for i in top100]
        top10 = top100_mols[:10]

        avg_top10 = np.mean([i.total for i in top10])
        avg_top100 = np.mean([i.total for i in top100_mols])

        if self.config.get('cal_div',default=False):
            from tdc import Evaluator
            div_evaluator = Evaluator(name = 'Diversity')
            diversity_top100 = div_evaluator([i.value for i in top100_mols])
        else:
            diversity_top100 = 0
        
        ### 
        if 'l_delta_b' in top10[0].property and 'aspect_ratio' in top10[0].property:
            all_mols = [item[0] for item in mol_buffer]
            all_mols = [i for i in all_mols if i.constraints['feasibility']<0.01]
            if len(all_mols)>0:
                scores = np.array([[-i.property['l_delta_b'],i.property['aspect_ratio']] for i in all_mols])
                volume = cal_fusion_hv(scores)
            else:
                volume = 0
        else:
            scores = np.array([i.scores for i in top100_mols])
            volume = cal_hv(scores) ###

        if buffer_type == "default":
            uniqueness = 1 - self.repeat_num / (self.generated_num + 1e-6)
            validity = 1 - self.failed_num / (self.generated_num + 1e-6)
        else:
            uniqueness = 1 - self.record_dict[f'{buffer_type}_repeat_num'] / (self.record_dict[f'{buffer_type}_all_num'] + 1e-6)
            validity = 1 - self.record_dict[f'{buffer_type}_failed_num'] / (self.record_dict[f'{buffer_type}_all_num'] + 1e-6)

        # Handle early stopping for default logging only
        if buffer_type == "default":
            new_score = avg_top100
            if new_score - self.old_score < 1e-4 and self.old_score>0.05:
                self.patience += 1
                if self.config.get('early_stopping',default=True) and self.patience >= 6000:
                    print('convergence criteria met, abort ...... ')
                    self.early_stopping = True
            else:
                self.patience = 0
            self.old_score = new_score

        # Select results_dict and path
        if buffer_type == "default":
            results_dict = self.results_dict
            save_dir = os.path.join(self.save_dir, "results")
        elif buffer_type == "main":
            results_dict = self.main_results_dict
            save_dir = os.path.join(self.save_dir, "results_main")
        elif buffer_type == "au":
            results_dict = self.au_results_dict
            save_dir = os.path.join(self.save_dir, "results_au")
        else:
            raise ValueError(f"Unknown buffer_type: {buffer_type}")

        # 日志文件统一带时间戳，便于区分多次运行的结果
        json_path = os.path.join(save_dir, f"{self.experiment_tag}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        results_dict['results'].append({
            'all_unique_moles': len(self.history_moles),
            'llm_calls': self.llm_calls,
            'Uniqueness': uniqueness,
            'Validity': validity,
            'Training_step': self.time_step,
            'avg_top1': top10[0].total,
            'avg_top10': avg_top10,
            'avg_top100': avg_top100,
            'top1_auc': auc1,
            'top10_auc': auc10,
            'top100_auc': auc100,
            'hypervolume': volume,
            'div': diversity_top100,
            'input_tokens': self.llm.input_tokens,
            'output_tokens': self.llm.output_tokens,
            'generated_num': self.generated_num,
            'running_time[s]': time.time()-self.start_time
        })

        # Only include config and history in default log
        if buffer_type == "default":
            print('================================================================')
            results_dict['params'] = self.config.to_string()
            if len(self.history_experience) > 1:
                results_dict['history_experience'] = [self.history_experience[0], self.history_experience[-1]]

        with open(json_path, 'w',encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False,indent=4)
        
        # 在协同模式下，每次 log_results 都保存 prompt.json（像 motsp 一样），确保中断后也能保留记录
        if buffer_type == "default" and getattr(self.config, "get", None) and self.config.get('model_collaboration', default=False):
            self._save_prompt_records_json()
        
        print(f'{buffer_type}: {len(self.history_moles)}/{self.budget} generated: {self.generated_num} | '
            f'mol_buffer: {len(mol_buffer)} | '
            f'Uniqueness: {uniqueness:.4f} | '
            f'Validity: {validity:.4f} | '
            f'llm_calls: {self.llm_calls} | '
            f'Training step: {self.time_step} | '
            f'avg_top1: {top10[0].total:.6f} | '
            f'avg_top10: {avg_top10:.6f} | '
            f'avg_top100: {avg_top100:.6f} | '
            f'top1_auc: {auc1:.4f} | '
            f'top10_auc: {auc10:.4f} | '
            f'top100_auc: {auc100:.4f} | '
            f'hv: {volume:.4f} | '
            f'unique top 100:{len(np.unique([i.value for i in top100_mols]))} | '
            f'input_tokens: {self.llm.input_tokens} | '
            f'output_tokens: {self.llm.output_tokens} | '
            f'running_time: {(time.time()-self.start_time)/3600:.3f} h | '
            f'div: {diversity_top100:.4f}'
            )


    def update_experience(self):
        prompt,best_moles_prompt,bad_moles_prompt = self.prompt_generator.make_experience_prompt(self.mol_buffer)
        response = self.llm.chat(prompt)
        
        #self.prompt_generator.experience = (f"I already have some experience, take advantage of them :{response}"
        #                                    )
        self.prompt_generator.pure_experience = response
        self.prompt_generator.experience = (f"I already have some experience: <experience> {response} </experience>" 
                                            f"You can take advantage of these experience and try to propose better molecules according to the objectives.\n")
        #self.prompt_generator.experience = (f"I already have some experience and some good and bad moleculles, the experience is: <experience> {response}" 
        #                                    f"good example molecules are: {best_moles_prompt},\n"
        #                                    f"and bad example molecules that you need to avoid molecules like them: {bad_moles_prompt}. </experience>\n"
        #                                    f"You can take advantage of them and try to propose better molecules according to the objectives.\n"
        #                                    )
        
        self.history_experience.append(self.prompt_generator.experience) 
        print('length exp:',len(self.prompt_generator.experience))

    
    def run(self):
        store_path = os.path.join(self.save_dir,'mols','_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}' +'.pkl')
        self.prompt_generator = self.prompt_module(self.config)
        
        if not os.path.exists(os.path.dirname(store_path)):
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
            
        # genetic_gfn (au_model) 初始化已移除

        """High level logic"""
        print('exper_name',self.config.get('exper_name'))
        set_seed(self.seed)
        start_time = time.time()
        
        #initialization 
        if self.config.get('inject_per_generation'):
            module_path = self.config.get('evalutor_path')  # e.g., "molecules"
            module = importlib.import_module(module_path)
            _get = getattr(module, "get_database")
            database = _get(self.config, n_sample=200)

        if self.config.get('resume'):
            population, init_pops = self.load_ckpt(store_path)
        else:
            population = self.generate_initial_population(n=self.pop_size)
            if len(population) < self.pop_size:
                population = self.llm_init(population)
            population = self.evaluate(population)  # including removing invalid and repeated candidates
            self.log_results()
            init_pops = copy.deepcopy(population)

        # 初始化 init_pops 到 prompt_records（与 MOLLM_1 一致）
        self.prompt_records["init_pops"] = [
            {
                "value": item.value,
                "property": getattr(item, "property", None),
                "total": float(getattr(item, "total", 0.0))
            }
            for item in init_pops
        ]

        data = {
            'history': self.history,
            'init_pops': init_pops,
            'final_pops': population,
            'all_mols': self.mol_buffer,
            'properties': self.property_list,
            'evaluation': self.results_dict['results'],
            'running_time': f'{(time.time() - start_time) / 3600:.2f} hours',
            'experience': self.history_experience,
        }
        with open(store_path, 'wb') as f:
            pickle.dump(data, f)

        self.num_gen = 0

        # DPO 触发辅助变量
        last_triggered_multiple = 0

        while True:
            if self.config.get('inject_per_generation'):
                print('inject!')
                population.extend(random.sample(database, self.config.get('inject_per_generation')))

            # 协同模式下，每一代先切换当前 LLM，与 MOLLM_1 对齐
            if getattr(self.config, "get", None) and self.config.get('model_collaboration', default=False):
                self.switch_llm()

            offspring_times = max(
                min(self.pop_size // self.num_offspring,
                    (self.budget - len(self.mol_buffer)) // self.num_offspring),
                1
            )
            offspring = self.generate_offspring(population, offspring_times)

            # 若存在新个体，基于历史缓冲与前代个体进行 Pareto 选择
            if offspring:
                if len(self.mol_buffer) >= self.pop_size:
                    population = self.select_next_population(self.pop_size)
                else:
                    population = offspring
            else:
                # 若本轮未能生成有效个体，回退到历史最优种群，确保后续 LLM 仍有父代
                selected = self.select_next_population(self.pop_size)
                population = selected if selected else population
            self.log_results()

            # 与 MOLLM_1 一致：按 experience_prob 更新经验
            if self.config.get('model.experience_prob') > 0 and len(self.mol_buffer) > 100:
                self.update_experience()

            # ---------------- DPO 触发逻辑：大小模型交替一次就触发 ----------------
            if getattr(self.config, "get", None) and self.config.get('model_collaboration', default=False) and self.llm2 is not None:
                min_rounds = min(self.llm1_rounds, self.llm2_rounds)
                # 修改为：每次大小模型都至少执行1轮后就触发DPO训练
                should_trigger_dpo = (
                    self.llm1_rounds >= 1 and
                    self.llm2_rounds >= 1 and
                    min_rounds - last_triggered_multiple >= 1
                )

                if should_trigger_dpo and len(self.mol_buffer) < self.budget:
                    last_triggered_multiple = min_rounds
                    print(
                        f"Triggering DPO training: LLM1 rounds={self.llm1_rounds}, "
                        f"LLM2 rounds={self.llm2_rounds}, total molecules={len(self.mol_buffer)}"
                    )

                    # 保存最新数据，以供 DPO 数据脚本构造训练数据
                    data = {
                        'history': self.history,
                        'init_pops': init_pops,
                        'final_pops': population,
                        'all_mols': self.mol_buffer,
                        'properties': self.property_list,
                        'evaluation': self.results_dict['results'],
                        'running_time': f'{(time.time() - start_time) / 3600:.2f} hours'
                    }
                    with open(store_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"Data saved to {store_path} before calling DPO training script.")

                    # 同步将 prompt_records 写入 JSON，供 makerldata_* 读取（统一记录格式）
                    self._save_prompt_records_json()

                    # 1) 先卸载本地模型，释放显存（与 MOLLM_1 一致）
                    if hasattr(self.llm2, 'reset_model'):
                        self.llm2.reset_model()

                    # 2) 调用 DPO 数据&训练脚本（支持通过配置覆盖路径，默认使用分子任务脚本）
                    train_exp_name = (
                        f"{'_'.join(self.property_list)}_{self.config.get('save_suffix')}_"
                        f"{self.seed}_{self._now_str}_round{min(self.llm1_rounds, self.llm2_rounds)}"
                    )
                    # 使用项目内的相对路径
                    MCCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    dpo_data_dir = os.path.join(MCCE_ROOT, "data", "dpo_training")
                    dpo_model_dir = os.path.join(MCCE_ROOT, "data", "dpo_models")

                    import subprocess
                    # 使用绝对路径调用 MCCE 下的 DPO 数据&训练脚本，避免找不到文件；
                    # 若配置中提供 dpo_script_path，则优先使用（方便为不同任务定制，如 circle_packing 使用 embedding 相似度脚本）
                    dpo_script_path = self.config.get(
                        'dpo_script_path',
                        default="/home/lzz/MCCE/tools/makerldata_dpov3.py"
                    )
                    dpo_command = [
                        "python", dpo_script_path,
                        "--exp", train_exp_name,
                        "--pkl_path", store_path,
                        "--data_dir", dpo_data_dir,
                        "--model_dir", dpo_model_dir,
                        "--ref_model_path", self.original_ref_model_path
                    ]
                    if self.previous_dpo_exp_name:
                        dpo_command.extend(["--prev_exp", self.previous_dpo_exp_name])

                    print(f"Starting DPO training with command: {' '.join(dpo_command)}")
                    subprocess.run(dpo_command)

                    # 更新上一次实验名称
                    self.previous_dpo_exp_name = train_exp_name

                    # 3) 训练完成后加载新模型，并删除旧的 DPO 模型目录（与 MOLLM_1 保持一致）
                    new_model_path = os.path.join(dpo_model_dir, train_exp_name)
                    if os.path.exists(new_model_path) and hasattr(self.llm2, 'load_model_from_path'):
                        previous_model_path = self.current_dpo_model_path

                        print(f"Loading new DPO model from: {new_model_path}")
                        self.llm2.load_model_from_path(new_model_path)
                        self.current_dpo_model_path = new_model_path

                        if previous_model_path and os.path.isdir(previous_model_path) and previous_model_path != new_model_path:
                            print(f"New DPO model loaded. Deleting old model directory: {previous_model_path}")
                            try:
                                shutil.rmtree(previous_model_path)
                                print("Successfully deleted old DPO model directory.")
                            except OSError as e:
                                print(f"Error deleting old DPO model directory {previous_model_path}: {e}")
                    else:
                        print(f"Warning: Trained model path {new_model_path} does not exist. Continuing with current model.")

            # ---------------- 终止条件与常规保存 ----------------
            if len(self.mol_buffer) >= self.budget or self.early_stopping:
                # 结束前记录 final_pops
                self.prompt_records["final_pops"] = [
                    {
                        "value": item.value,
                        "property": getattr(item, "property", None),
                        "total": float(getattr(item, "total", 0.0))
                    }
                    for item in population
                ]

                self.log_results(finish=True)
                # genetic_gfn log_results 已移除
                break

            self.num_gen += 1
            data = {
                'history': self.history,
                'init_pops': [obj[0] for obj in self.mol_buffer[:self.pop_size]],
                'final_pops': population,
                'all_mols': self.mol_buffer,
                'properties': self.property_list,
                'evaluation': self.results_dict['results'],
                'running_time': f'{(time.time() - start_time) / 3600:.2f} hours',
                'experience': self.history_experience,
            }
            with open(store_path, 'wb') as f:
                pickle.dump(data, f)
            if self.num_gen % 10 == 0:
                print(f"Data saved to {store_path}")

        # 协同 + DPO 模式下，结束时也写一次 prompt_records，确保与 pkl 对应
        if getattr(self.config, "get", None) and self.config.get('model_collaboration', default=False):
            self._save_prompt_records_json()

        print(f'=======> total running time { (time.time() - start_time) / 3600 :.2f} hours <=======')

        return init_pops, population  # 计算效率

    def _save_prompt_records_json(self):
        """
        将当前的 prompt_records 统一写入 JSON，供各类 DPO 数据构建脚本读取。
        文件命名规则与 MOLLM_1 保持一致：
        <save_dir>/prompt/<properties>_<save_suffix>_<seed>_prompt.json
        """
        if not (getattr(self.config, "get", None) and self.config.get('model_collaboration', default=False)):
            return

        prompt_dir = os.path.join(self.save_dir, "prompt")
        os.makedirs(prompt_dir, exist_ok=True)
        # 与 results 一样，使用统一的 experiment_tag，并在文件名中加入 _prompt 后缀
        prompt_json_path = os.path.join(prompt_dir, self.experiment_tag + '_prompt.json')
        with open(prompt_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.prompt_records, f, ensure_ascii=False, indent=2)
        print(f"Prompt records saved to {prompt_json_path}.")

    def record(self, tmp_offspring: list, buffer_type: str) -> list:
        """
        Records valid and unique molecules from the given temporary offspring list.

        Parameters:
        - tmp_offspring (list): List of generated candidate molecules (Item objects).
        - buffer_type (str): Type identifier for tracking stats (e.g., 'main', 'au').

        Returns:
        - offspring (list): Filtered list of valid and unique Item objects.
        """
        offspring = []
        smiles_this_gen = []
        self.record_dict[buffer_type + '_all_num'] += len(tmp_offspring)

        for child in tmp_offspring:
            mol = Chem.MolFromSmiles(child.value)
            if mol is None:
                self.record_dict[buffer_type + '_failed_num'] += 1
            else:
                child.value = Chem.MolToSmiles(mol)
                if (child.value in self.record_dict[buffer_type + '_history_smiles'] or
                    child.value in smiles_this_gen):
                    self.record_dict[buffer_type + '_repeat_num'] += 1
                else:
                    self.record_dict[buffer_type + '_history_smiles'].append(child.value)
                    smiles_this_gen.append(child.value)
                    offspring.append(child)
        return offspring

    def mating(self, parent_list: list, au: bool = False) -> tuple:
        """
        Applies genetic operators (crossover, mutation, or exploration) to parent list.

        Parameters:
        - parent_list (list): A list of parent Item objects.
        - au (bool): Whether this is for auxiliary model (default: False).

        Returns:
        - tuple: (list of offspring Items, str prompt, str response)
        """
        crossover_prob = self.config.get('model.crossover_prob')
        mutation_prob = self.config.get('model.mutation_prob')
        explore_prob = self.config.get('model.explore_prob')
        function = np.random.choice(
            [self.crossover, self.mutation, self.explore],
            p=[crossover_prob, mutation_prob, explore_prob]
        )
        items, prompt, response = function(parent_list)
        return items, prompt, response
    
    def generate_offspring(self, population: list, offspring_times: int) -> list:
        """
        Generates new offspring from the population using LLM-driven operations, and evaluates them.

        Parameters:
        - population (list): List of Item objects representing the current population.
        - offspring_times (int): Number of offspring pairs to generate.

        Returns:
        - list: Evaluated and recorded offspring.
        """
        # 防御性检查：如果当前种群过小或为空，避免 random.sample 报错
        if not population:
            print("Warning: population is empty, skip offspring generation for this round.")
            return []

        # 当种群数量 < num_offspring 时，按可用个体数采样，避免 ValueError
        sample_k = min(self.num_offspring, len(population))
        
        # 根据当前使用的模型类型调整 offspring_times（方便调试/节省 API 成本）
        LOCAL_LLM_MODELS = [
            'qwen2.5-0.5b-instruct',
            'qwen2.5-3b-instruct',
            'qwen2.5-7b-instruct',
            'Qwen3-32B',
            'DeepSeek-R1-0528-Qwen3-8B',
            'Qwen3-8B',
            'merged-sft-qwen2.5-7b-instruct',
            'customdata-sft-qwen2.5-7b-instruct',
            'customdata-sft-qwen2.5-7b-instruct-v2'
        ]
        
        # 删除本地模型推理次数限制，统一使用默认的offspring_times
        
        parents = [random.sample(population, sample_k) for i in range(offspring_times)]

        # 与 MOLLM_1 保持一致：本地大模型串行，API 模型并行
        LOCAL_LLM_MODELS = [
            'qwen2.5-0.5b-instruct',
            'qwen2.5-3b-instruct',
            'qwen2.5-7b-instruct',
            'Qwen3-32B',
            'DeepSeek-R1-0528-Qwen3-8B',
            'Qwen3-8B',
            'merged-sft-qwen2.5-7b-instruct',
            'customdata-sft-qwen2.5-7b-instruct',
            'customdata-sft-qwen2.5-7b-instruct-v2'
        ]

        if hasattr(self.llm, 'model_choice') and self.llm.model_choice in LOCAL_LLM_MODELS:
            # 本地模型：串行推理，避免多线程占用显存，并显示进度条
            children, prompts, responses, parent_lists = [], [], [], []
            for parent_list in tqdm(parents, desc="LLM (local) generating offspring", leave=False):
                child, prompt, response = self.mating(parent_list)
                children.append(child)
                prompts.append(prompt)
                responses.append(response)
                parent_lists.append(parent_list)
                self.llm_calls += 1
        else:
            # API 模型：并行加速，同时在终端显示进度条
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.mating, parent_list=parent_list) for parent_list in parents]
                results = []
                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(futures),
                                   desc="LLM (API) generating offspring",
                                   leave=False):
                    try:
                        result = future.result(timeout=900)
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        print("Warning: A task timed out after 900 seconds.")
                        continue
                if results:
                    children, prompts, responses = zip(*results)
                    parent_lists = parents
                    self.llm_calls += len(results)
                else:
                    print("No results collected due to timeouts.")
                    return []

        # 先记录本轮所有 LLM 调用的 prompt/response（无论是否产生有效 offspring）
        self.history.push(prompts, children, responses)

        # 在协同 + DPO 场景下，统一记录每一步的 prompt / parents / children / response
        if getattr(self.config, "get", None) and self.config.get('model_collaboration', default=False):
            def _safe_total(item):
                val = getattr(item, "total", 0.0)
                if val is None:
                    return 0.0
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0

            record_len = len(prompts)
            for idx in range(record_len):
                parent_list = parent_lists[idx] if idx < len(parent_lists) else []
                child_list = children[idx] if idx < len(children) else []
                prompt = prompts[idx]
                response = responses[idx]

                self.prompt_records["queries"].append({
                    "prompt": prompt,
                    "response": response,
                    "parents": [
                        {
                            "value": p.value,
                            "property": getattr(p, "property", None),
                            "total": _safe_total(p)
                        }
                        for p in parent_list
                    ],
                    "children": [
                        {
                            "value": c.value,
                            "property": getattr(c, "property", None),
                            "total": _safe_total(c)
                        }
                        for c in child_list
                    ]
                })

        tmp_offspring = []
        if self.start:
            self.start = False
        for child_pair in children:
            self.generated_num += len(child_pair)
            tmp_offspring.extend(child_pair)

        # genetic_gfn 流程已移除

        offspring = tmp_offspring
        if len(offspring) == 0:
            return []

        offspring = self.evaluate(offspring)

        return offspring

    # save_log_mols 函数已移除（genetic_gfn 专用）

    # ---------------- 协同模式下主/辅 LLM 切换（对齐 MOLLM_1） ----------------
    def switch_llm(self):
        """
        两个 LLM 轮流出手：
        - flag=0: 使用主模型 llm1
        - flag=1: 使用辅模型 llm2（本地 Qwen，用于 DPO 微调）
        """
        if self.llm2 is not None:
            self.current_llm_flag = 1 - self.current_llm_flag
            self.llm = self.llm1 if self.current_llm_flag == 0 else self.llm2
            # 更新轮次计数器（与 MOLLM_1 一致）
            if self.current_llm_flag == 0:
                self.llm1_rounds += 1
            else:
                self.llm2_rounds += 1
            self.model_round_counter += 1

    def mol_buffer_store(self, mol_buffer: list, mols: list) -> list:
        """
        Stores non-duplicate molecules into a specified buffer.

        Parameters:
        - mol_buffer (list): List buffer to store Item objects.
        - mols (list): List of Item objects to attempt storing.

        Returns:
        - list: Updated mol_buffer with new unique molecules.
        """
        all_smiles = [i[0].value for i in mol_buffer]
        for child in mols:
            mol = Chem.MolFromSmiles(child.value) 
            if mol is None: # check if valid
                pass
            else:
                child.value = Chem.MolToSmiles(mol,canonical=True)
                # check if repeated
                if child.value in all_smiles:
                    pass
                else:
                    all_smiles.append(child.value)
                    mol_buffer.append([child,len(mol_buffer)+1])
        return mol_buffer


    def select_next_population(self,pop_size):
        whole_population = [i[0] for i in self.mol_buffer]
        if len(self.property_list)>1:
            return nsga2_so_selection(whole_population, pop_size)
        else:
            return so_selection(whole_population,pop_size)

    def generate_offspring_au(self, population, offspring_times=20):
        parents = [random.sample(population, 2) for i in range(offspring_times)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.mating, parent_list=parent_list,au=True) for parent_list in parents]
            results = [future.result() for future in futures]
            children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']
            self.llm_calls += len(results)
        tmp_offspring = []
        for child_pair in children:
            self.generated_num += len(child_pair)
            tmp_offspring.extend(child_pair)
        return tmp_offspring
    
    def load_ckpt(self,store_path):
        print('resume training')
        save_dir = os.path.join(self.save_dir, "results")
        json_path = os.path.join(save_dir, '_'.join(self.property_list) + '_' +
                            self.config.get('save_suffix') + f'_{self.seed}.json')
        with open(store_path,'rb') as f:
            ckpt = pickle.load(f)
        with open(json_path,'rb') as f:
            result_ckpt = json.load(f)
        self.mol_buffer = ckpt['all_mols']
        population = self.select_next_population(self.pop_size) 
        init_pops = [i[0] for i in self.mol_buffer[:self.pop_size]]#ckpt['init_pops']
        self.history = ckpt['history']
        
        self.history_moles = [i[0].value for i in self.mol_buffer]
        self.results_dict['results'] = ckpt['evaluation']
        self.generated_num = result_ckpt['results'][-1]['generated_num']
        self.llm_calls = result_ckpt['results'][-1]['llm_calls']
        self.repeat_num = int((1-result_ckpt['results'][-1]['Uniqueness']) * self.generated_num)
        self.failed_num = int((1-result_ckpt['results'][-1]['Validity']) * self.generated_num)
        return population, init_pops

    
    
 