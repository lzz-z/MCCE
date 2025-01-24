import pickle
import json
import uuid
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# 禁用RDKit的日志输出
RDLogger.DisableLog('rdApp.*')

import selfies as sf
import re
import os
import pickle 
import torch
import copy
from bio_t5.nanoT5.utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
    setup_basics_without_acc
)
from accelerate import Accelerator

import yaml
from argparse import Namespace
import argparse
def load_yaml_to_args(yaml_file):
    # 加载 YAML 文件
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # 将字典转换为命名空间
    def dict_to_namespace(d):
        namespace = Namespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace
    
    args = dict_to_namespace(config)
    return args



def rm_map_number(smiles):
    t = re.sub(':\d*', '', smiles)
    return t

def canonicalize(smiles):
    try:
        smiles_to_selfies(smiles)
        smiles = rm_map_number(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        else:
            return Chem.MolToSmiles(mol)
    except:
        return None

def smiles_to_selfies(smiles):
    return sf.encoder(smiles, strict=False)

def selfies_to_smiles(selfies):
    return sf.decoder(selfies)

def read_inference_output(output_file, col_name, col_idx):
    with open(output_file, 'rb') as f:
        data_pick = f.readlines()
    data_pick = [x.decode('utf-8') for x in data_pick]
    return data_pick

def get_t5_input(pred_sfi, task):
    if task == 'bace':
        t5_input = f'Definition: Molecule property prediction task (a binary classification task) for the BACE dataset. The BACE dataset provides qualitative (binary label) binding results for a set of inhibitors of human beta-secretase 1 (BACE-1). If the given molecule can inhibit BACE-1, indicate via "Yes". Otherwise, response via "No".\n\nNow complete the following example -\nInput: Molecule: <bom>{pred_sfi}<eom>\nOutput: '
    elif task == 'bbbp':
        t5_input = f'Definition: Molecule property prediction task (a binary classification task) for the BBBP dataset. The blood-brain barrier penetration (BBBP) dataset is designed for the modeling and prediction of barrier permeability. If the given molecule can penetrate the blood-brain barrier, indicate via "Yes". Otherwise, response via "No".\n\nNow complete the following example -\nInput: Molecule: <bom>{pred_sfi}<eom>\nOutput: '
        
    return t5_input

def decode(tokenizer, preds):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    preds = [pred.strip() for pred in preds]
    return preds

def get_checkpoint_path(task, ckpt_blob_root):
    if task == 'bbbp':
        return os.path.join(ckpt_blob_root, 'v-qizhipei/checkpoints/biot5/bbbp/240508_gpu1_bsz4_acc1_ts50000_eps100_ws100_cosine_lr3e-4_dp0.1_seed256/checkpoint-ft-12400/pytorch_model.bin')
    elif task == 'bace':
        return os.path.join(ckpt_blob_root, 'v-qizhipei/checkpoints/biot5/bace/240508_gpu1_bsz16_acc2_ts50000_eps100_ws100_cosine_lr3e-4_dp0.1_seed102/checkpoint-ft-3200/pytorch_model.bin')

def split_list_by_length_indices(lst, length=128):
    # 使用列表切片方法，根据指定的长度计算索引范围
    return [(i, min(i + length - 1, len(lst) - 1)) for i in range(0, len(lst), length)]

def get_all_prediction(input_smiles_all, task, tokenizer, model, batch_size=1280, args=None):

    all_res = {}
    all_res['input_smiles'] = copy.deepcopy(input_smiles_all)
    

    pred_smi_can_all = [canonicalize(s) for s in input_smiles_all]
    all_res['input_smi_can'] = copy.deepcopy(pred_smi_can_all)

    mask = [True if x is not None else False for x in pred_smi_can_all]

    valid_input_smiles_can = [pred_smi_can_all[i] for i in range(len(input_smiles_all)) if mask[i]]
 
    pred_sfi = [smiles_to_selfies(vs) for vs in valid_input_smiles_can]


    #print(f'valid ratio: {sum(mask)/len(mask)}, num of valid: {sum(mask)}')

    result = []
    pred_sfi_index = 0

    for i in range(len(input_smiles_all)):
        if mask[i]:
            result.append(pred_sfi[pred_sfi_index])
            pred_sfi_index += 1
        else:
            result.append(None)
    all_res['input_sfi'] = result

  
    t5_input = [ get_t5_input(s, task) for s in pred_sfi]

    sources = t5_input

    # sources = [t5_input,t5_input]
    model_inputs = tokenizer(
        sources,
        max_length=1024,
        padding='longest',
        return_tensors='pt',
        truncation=True,
        pad_to_multiple_of=8,
    )


    split_idx = split_list_by_length_indices(list(range(len(valid_input_smiles_can))), batch_size)
    all_prediction = []
 

    for i, (start, end) in enumerate(split_idx):
        model_inputs_batch = {k: v[start:end+1].to(model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            # print(model.device, model_inputs_batch['input_ids'].device)
            predictions = model.generate(
                input_ids=model_inputs_batch['input_ids'],
                attention_mask=model_inputs_batch['attention_mask'],
                max_length=8,
                generation_config=model.generation_config,
            )
            predict = decode(tokenizer, predictions)
            all_prediction.extend(predict)

    
    # all_res['predictions'] = [all_prediction[i] if mask[i] else None for i in range(len(pred_smi_can_all)) ]

    result = []
    all_prediction_index = 0

    for i in range(len(input_smiles_all)):
        if mask[i]:
            result.append(all_prediction[all_prediction_index])
            all_prediction_index += 1
        else:
            result.append(None)
    all_res['predictions'] = result

   

    return all_res
 


# input_smiles = 'CCCN(CCCC)C(=O)c1cc(C)cc(C)c1'

def evaluate_one_file(this_file_path, task, tokenizer, model, batch_size, args):
    all_inference_res = pd.read_csv(this_file_path)
    col_name_list = all_inference_res.columns.tolist()
    all_evaluation_res = {}
    for col_name in col_name_list:
        if col_name in args.skip_col_name:
            continue
        this_col_smiles = all_inference_res[col_name].tolist()
        if '<mol>' in this_col_smiles[0]:
            this_col_smiles = [x.split('<mol>')[-1].split('</mol>')[0] for x in this_col_smiles]
        this_res = get_all_prediction(this_col_smiles, task, tokenizer, model, batch_size, args)
        all_evaluation_res[col_name] = this_res
    return all_evaluation_res

def evaluate_one_folder(folder_path, task, tokenizer, model, batch_size, args):
    all_files = os.listdir(folder_path)
    all_res = {}
    for file_name in all_files:
        print(file_name)
        this_file_path = os.path.join(folder_path, file_name)
        this_res = evaluate_one_file(this_file_path, task, tokenizer, model, batch_size, args)
        all_res[file_name] = this_res
    return all_res

def evaluate_one_task(task_folder, task, hparams):


    # this_file_path = os.path.join(inference_blob_root, 'Data/202407/total_training_data_1/inference/bbbp/4_pret_sft_dpo/bbbp_sft_epoch_1_dpo_epoch_1.csv')
    # task = 'bbbp'
    batch_size = 1280
    args = torch.load(hparams.args_save_path)
    # args.device = hparams.device
    # args.ckpt_blob_root = hparams.ckpt_blob_root
    # args.inference_blob_root = hparams.inference_blob_root
    # args.inferece_folder = hparams.inference_folder

    args.model.checkpoint_path = get_checkpoint_path(task, hparams.ckpt_blob_root)
    # args.device = args.device
    config = get_config(args)
    tokenizer = get_tokenizer(args)
    logger = setup_basics_without_acc(args)
    model = get_model(args, config, tokenizer, logger)
    model = model.to(hparams.device)
    all_class_folders = os.listdir(task_folder)
    all_res = {}
    for class_folder in all_class_folders:
            
        this_class_folder = os.path.join(task_folder, class_folder)
        this_res = evaluate_one_folder(this_class_folder, task, tokenizer, model, batch_size, hparams)
        all_res[class_folder] = this_res
    return all_res

def load_bbbp_model():
    batch_size = 1280
    args = torch.load('/home/msrai4srl4s/nian/MOLLM/bio_t5/args.save')
    args.model.checkpoint_path = '/home/msrai4srl4s/nian/rl4s/Checkpoint/biot5/bbbp/240508_gpu1_bsz4_acc1_ts50000_eps100_ws100_cosine_lr3e-4_dp0.1_seed256/checkpoint-ft-12400/pytorch_model.bin'
    config = get_config(args)
    tokenizer = get_tokenizer(args)
    logger = setup_basics_without_acc(args)
    model = get_model(args, config, tokenizer, logger)
    model = model.to('cuda')
    return tokenizer,model

def get_bbbp(input_smiles_all, tokenizer, model):
    results = get_all_prediction(input_smiles_all, 'bbbp', tokenizer, model, batch_size=1280, args=None)['predictions']
    results = [1. if i=='Yes.' else 0. for i in results]
    return results

if __name__ == '__main__':


    # ckpt_blob_root = '/blob/blob/'
    # inference_blob_root = '/blob/rl4s/'
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--task', default='bace', help='task name')
    Parser.add_argument('--ckpt_blob_root', default='/blob/blob/', help='blob root for checkpoint')
    Parser.add_argument('--inference_blob_root', default='/blob/rl4s/', help='blob root for inference')
    Parser.add_argument('--inference_folder', default='Data/202407/total_training_data_1/inference_for_llama3_9', help='inference folder')
    Parser.add_argument('--device', default='cuda:0', help='device')
    Parser.add_argument('--args_save_path', default='/home/aiscuser/Evaluation/bio_t5/args.save', help='args save path')

    hparams = Parser.parse_args()
    
    input_smiles_all = ['CCO','CCC']
    tokenizer,model = load_bbbp_model()
    get_bbbp(input_smiles_all,tokenizer,model)



 