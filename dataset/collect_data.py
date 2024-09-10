import pandas as pd
import re

def extract_smiles_from_string(text):
    pattern = r"<mol>(.*?)</mol>"
    smiles_list = re.findall(pattern, text)
    return smiles_list

def read_csv():
    donor_df = pd.read_csv('/home/v-nianran/src/data/dpo_donor_test.csv')
    qed_df = pd.read_csv('/home/v-nianran/src/data/dpo_qed_test.csv')
    logp_df = pd.read_csv('/home/v-nianran/src/data/dpo_logp_test.csv')
    return qed_df,logp_df,donor_df

def make_dataset():
    qed_df,logp_df,donor_df = read_csv()

    # uniformly sample molecules from three test files
    df = pd.concat([logp_df.iloc[:100],qed_df.iloc[100:200],donor_df.iloc[200:300]])
    moles = [extract_smiles_from_string(p)[0] for p in df.prompt]
    #moles

    combine_prompt = []
    for index,mol in enumerate(moles):
        new_p = re.sub(r'<mol>.*?</mol>', '', logp_df.prompt[index]) + ' and still for this molecule ' +  \
            re.sub(r'<mol>.*?</mol>', '', qed_df.prompt[index]) + ' and still for this molecule  ' +\
            re.sub(r'<mol>.*?</mol>', '', donor_df.prompt[index]) + f' This molecule is <mol>{mol}</mol>'
        combine_prompt.append(new_p)
    len(combine_prompt),combine_prompt[1]
    return combine_prompt,moles

def read_meta():
    import json
    with open('/home/v-nianran/src/data/test_qed.metadata.json','r') as f:
        qed_meta = json.load(f)
    with open('/home/v-nianran/src/data/test_logp.metadata.json','r') as f:
        logp_meta = json.load(f)
    with open('/home/v-nianran/src/data/test_donor.metadata.json','r') as f:
        donor_meta = json.load(f)
    return qed_meta,logp_meta,donor_meta