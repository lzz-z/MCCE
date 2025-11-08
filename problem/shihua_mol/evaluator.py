
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from tdc import Oracle
import os
from rdkit.Chem import AllChem
from pyscf import gto, scf, dft
import pandas as pd
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
import time
def check_validity(smi):
    # 1. SMILES 是否可解析
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    # 2. 化学合理性检查
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False

    # 3. 环系统合理性
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) < 3 or len(ring) > 8:
            return False

    return True


def calc_gap_and_dipole_pyscf(smiles: str, method="B3LYP", basis="def2-svp", charge=0, spin=0):
    """
    输入: SMILES 字符串
    输出: dict {'homo_lumo_gap_au': float, 'dipole_moment_debye': float}
    - 采用 PySCF DFT 计算 (默认 B3LYP/def2-SVP)
    - 单位:
        * gap: Hartree (au)
        * dipole: Debye
    """
    # 生成3D分子
    mol_rd = Chem.MolFromSmiles(smiles)
    mol_rd = Chem.AddHs(mol_rd)
    AllChem.EmbedMolecule(mol_rd, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol_rd, maxIters=500)
    conf = mol_rd.GetConformer()
    atoms = [(a.GetSymbol(), (conf.GetAtomPosition(i).x,
                              conf.GetAtomPosition(i).y,
                              conf.GetAtomPosition(i).z))
             for i, a in enumerate(mol_rd.GetAtoms())]

    # 建立PySCF分子
    mol = gto.Mole()
    mol.build(atom=atoms, basis=basis, charge=charge, spin=spin, unit='Angstrom')

    # DFT计算
    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = method
    mf.conv_tol = 1e-8
    mf.kernel()

    # 取HOMO/LUMO
    e = np.array(mf.mo_energy)
    occ = np.array(mf.mo_occ)
    homo = e[occ > 0].max()
    lumo = e[occ == 0].min()
    gap_au = float(lumo - homo)

    # 偶极矩（Debye）
    dm = mf.make_rdm1()
    dip = mf.dip_moment(mol, dm=dm, unit='Debye')
    dip_debye = float(np.linalg.norm(dip))

    return {"homo_lumo_gap_au": gap_au, "dipole_moment_debye": dip_debye}

sa_scorer = Oracle(name='SA')

constraints = [
    {"constraint_id": "test_001", "molecular_weight": [200, 350], "sa_score": [1, 4], "homo_lumo_gap": [0.25, 0.35], "dipole_moment": [1.0, 3.0]},
{"constraint_id": "test_002", "molecular_weight": [350, 500], "sa_score": [3, 6], "homo_lumo_gap": [0.20, 0.30], "dipole_moment": [2.0, 5.0]},
{"constraint_id": "test_003", "molecular_weight": [150, 250], "sa_score": [1, 3.5], "homo_lumo_gap": [0.30, 0.40], "dipole_moment": [0.0, 1.5]},
{"constraint_id": "test_004", "molecular_weight": [250, 400], "sa_score": [2.5, 5.5], "homo_lumo_gap": [0.22, 0.32], "dipole_moment": [4.0, 7.0]},
{"constraint_id": "test_005", "molecular_weight": [280, 320], "sa_score": [2.0, 3.5], "homo_lumo_gap": [0.29, 0.31], "dipole_moment": [1.5, 2.5]}
]
constraint = constraints[4]
print('constraint:', constraint)
def generate_initial_population(config, seed=42):
    n_sample = config.get('optimization.pop_size')
    df = pd.read_csv('/root/nian/MOLLM/problem/shihua_mol/start_mols.csv')
    #filtered_df = df[
    #    (df["sa_score"].between(*constraint["sa_score"])) &
    #    (df["HOMO_LUMO_gap_au"].between(*constraint["homo_lumo_gap"])) &
    #    (df["Dipole_debye"].between(*constraint["dipole_moment"]))
    #]
    filtered_df = df
    #filtered_df = filtered_df.sample(n=n_sample,random_state=seed)
    strings = filtered_df.SMILES.values.tolist()
    return strings


from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import time, os

class RewardingSystem:
    def __init__(self, config):
        self.df = pd.read_csv('/root/nian/MOLLM/problem/shihua_mol/start_mols.csv')
        self.config = config
        self.qm9_smiles = self.df.SMILES.values.tolist()
        print('qm9_smiles:',len(self.qm9_smiles))

    def _evaluate_single(self, smi):
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not check_validity(smi):
            return smi, None, "invalid molecule"
        if smi in self.qm9_smiles:
            weight = self.df[self.df.SMILES == smi].iloc[0]["molecular_weight"]
            sa = self.df[self.df.SMILES == smi].iloc[0]["sa_score"]
            gap = self.df[self.df.SMILES == smi].iloc[0]["homo_lumo_gap_au"]
            dipole = self.df[self.df.SMILES == smi].iloc[0]["dipole_moment_debye"]
        else:
            weight = Descriptors.MolWt(mol)
            sa = float(sa_scorer(smi))
            gap_info = calc_gap_and_dipole_pyscf(smi)
            gap = gap_info["homo_lumo_gap_au"]
            dipole = gap_info["dipole_moment_debye"]

        result = {
            "original_results": {
                "molecular_weight": weight,
                "sa_score": sa,
                "homo_lumo_gap_au": gap,
                "dipole_moment_debye": dipole,
            }}
            #"transformed_results": {
            #    "molecular_weight": 1 - int(constraint["molecular_weight"][0] <= weight <= constraint["molecular_weight"][1]),
            #    "sa_score": 1 - int(constraint["sa_score"][0] <= sa <= constraint["sa_score"][1]),
            #    "homo_lumo_gap_au": 1 - int(constraint["homo_lumo_gap"][0] <= gap <= constraint["homo_lumo_gap"][1]),
            #    "dipole_moment_debye": 1 - int(constraint["dipole_moment"][0] <= dipole <= constraint["dipole_moment"][1]),
            #},
        result["transformed_results"] = result['constraint_results'] ={
                "molecular_weight": np.abs(weight - (constraint['molecular_weight'][0] + constraint['molecular_weight'][1]) / 2) / 100,
                "sa_score": np.abs(sa - (constraint['sa_score'][0] + constraint['sa_score'][1]) / 2) / 10,
                "homo_lumo_gap_au": np.abs(gap - (constraint['homo_lumo_gap'][0] + constraint['homo_lumo_gap'][1]) / 2) ,
                "dipole_moment_debye": np.abs(dipole - (constraint['dipole_moment'][0] + constraint['dipole_moment'][1]) / 2) / 10,
            }
        
        result["overall_score"] = 4 - sum(result["transformed_results"].values())
        #if result["overall_score"] >= 3.7:
        print(f"smi: {smi}, result:",result['constraint_results'],result['transformed_results'],result['overall_score'])
        return smi, result, None

   

    def evaluate(self, items, mol_buffer, timeout_sec=240):
        invalid_num = 0
        repeated_num = 0
        evaluated = []
        smiles_seen = set()
        new_items = []
        history_moles = [i.value for i, _ in mol_buffer]

        # === Step 1: 去重与有效性检查 ===
        for i in items:
            mol = Chem.MolFromSmiles(i.value)
            if mol is None or not check_validity(i.value):
                invalid_num += 1
                continue
            i.value = Chem.MolToSmiles(mol)
            if i.value not in smiles_seen and i.value not in history_moles:
                smiles_seen.add(i.value)
                new_items.append(i)
            else:
                repeated_num += 1

        items = new_items

        # === Step 2: 多线程并行计算，单任务最大等待 5 分钟 ===
        max_workers = min(5, os.cpu_count())
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._evaluate_single, item.value): item for item in items}

            for future in as_completed(futures):
                item = futures[future]
                try:
                    smi, result, error = future.result(timeout=timeout_sec)
                    if result:
                        item.assign_results(result)
                        evaluated.append(item)
                    else:
                        invalid_num += 1
                        print(f"⚠️ {smi}: {error}")

                except TimeoutError:
                    invalid_num += 1
                    print(f"⏰ Timeout (> {timeout_sec}s): {item.value} 被跳过")

                except Exception as e:
                    invalid_num += 1
                    print(f"⚠️ {item.value}: {str(e)}")

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        log_dict = {
            "invalid_num": invalid_num,
            "repeated_num": repeated_num,
        }

        return evaluated, log_dict

