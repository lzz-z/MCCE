# -*- coding: utf-8 -*-
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
import torch
import os
from tools.rl4s.evaluation_client import get_evaluation
import argparse
#nano rl4s/Data/202407/qed/dpo_qed_test.csv
# validation = False

def load_scigpt():
    tokenizer_home = '/home/v-nianran/rl4s/Data/Llama-2-7b-hf'
    tokenizer = SFMDecTokenizer.from_pretrained(
        tokenizer_home,
        prot_spm_path='/home/v-nianran/blob/shufxi/data/scigpt/ur50bpe/bpe',
        dna_spm_path='/home/v-nianran/blob/shufxi/data/scigpt/dnabpe/bpe',
        rna_spm_path='/home/v-nianran/blob/shufxi/data/scigpt/rnabpe/bpe',
    )


    from transformers import AutoTokenizer, AutoModelForCausalLM

    exp_name =  'eval_dpo_qed_epoch3'  #'inference_test'
    validation = False #args.validation
    # step = 29999
    # bsz = 96
    ckpt_home = '/home/v-nianran/rl4s/users/yuwang5/logs/test/total/20240712154953/global_step4818'  #f'/hai1/shufxi/scigpt/7bv3/stageB/global_step29999/'
    save_dir = '/home/v-nianran/rl4s/users/nian/evaluation_res'  #'/blob/yuwang5/'
    setting = ['similarity','logp']  #['similarity', 'logp', 'qed', 'drd2']
    print( ['similarity']  + setting)

    def show_ckpt(name, ckpt):
        for k, v in ckpt.items():
            if 'dummy' not in k:
                print(name, k, v.shape)

    model = AutoModelForCausalLM.from_pretrained(tokenizer_home)

    model_dict = model.state_dict()
    ckpt_dict = {}
    layer0 = torch.load(os.path.join(ckpt_home, "layer_00-model_states.pt"), map_location=torch.device("cpu"))
    del layer0['embed_tokens_ref.weight']
    ckpt_dict['model.embed_tokens.weight'] = layer0['embed_tokens.weight']
    show_ckpt('layer0', layer0)

    for l in range(0, 32):
        l_index = str(l + 1).zfill(2)
        layer = torch.load(os.path.join(ckpt_home, f"layer_{l_index}-model_states.pt"), map_location=torch.device("cpu"))
        del layer['self_attn_ref.q_proj.weight']
        del layer['self_attn_ref.k_proj.weight']
        del layer['self_attn_ref.v_proj.weight']
        del layer['self_attn_ref.o_proj.weight']
        del layer['mlp_ref.gate_proj.weight']
        del layer['mlp_ref.up_proj.weight']
        del layer['mlp_ref.down_proj.weight']
        del layer['input_layernorm_ref.weight']
        del layer['post_attention_layernorm_ref.weight']
        show_ckpt(l_index, layer)
        for k in layer:
            if "dummy" in k or 'rotary_emb' in k:
                continue
            ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]
    layer = torch.load(os.path.join(ckpt_home, "layer_33-model_states.pt"), map_location=torch.device("cpu"))
    # del layer['norm_ref.weight']
    show_ckpt(33, layer)
    ckpt_dict["model.norm.weight"] = layer["norm.weight"]

    layer = torch.load(os.path.join(ckpt_home, "layer_34-model_states.pt"), map_location=torch.device("cpu"))
    del layer['lm_head_ref.weight']
    del layer['num_head_ref.fc1.weight']
    del layer['num_head_ref.fc1.bias']
    del layer['num_head_ref.fc2.weight']
    del layer['num_head_ref.fc2.bias']
    show_ckpt(34, layer)
    ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
    model_dict.update(ckpt_dict)

    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(model_dict)
    model = model.cuda()

    print('model built')
    return model,tokenizer