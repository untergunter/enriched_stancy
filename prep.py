import json
import pandas as pd
from glob import glob
import os
import numpy as np

def get_raw_files():
    """ findes combines and splits to train dev test """
    all_jsons = glob('./**/*.json',recursive=True)
    for file_name in all_jsons:
        name = file_name.split(os.sep)[-1]
        if name == 'perspective_pool_v1.0.json':
            perspective = file_name
        elif name == 'evidence_pool_v1.0.json':
            evidence = file_name
        elif name == 'dataset_split_v1.0.json':
            split = file_name
        elif name == 'perspectrum_with_answers_v1.0.json':
            merger = file_name
            
    perspective = pd.read_json(perspective)
    perspective.columns= ['pId','perspective','source']
    evidence = pd.read_json(evidence)
    split = pd.read_json(split,typ='series').reset_index()
    split.columns = ['id','split']
    merger = pd.read_json(merger)
    return perspective,evidence,split,merger

def unwind_merger(merger_df):
    merger_df = merger_df.explode('perspectives')
    row_per_cluster = merger_df['perspectives'].apply(pd.Series)[['pids','stance_label_3']]
    row_per_cluster = row_per_cluster.explode('pids')
    row_per_cluster = row_per_cluster.drop_duplicates()
    row_per_pid_cid = pd.merge(merger_df,row_per_cluster,how='inner',left_index=True,right_index=True)
    row_per_pid_cid = row_per_pid_cid[['cId','text','pids','stance_label_3','split']]
    row_per_pid_cid = row_per_pid_cid.drop_duplicates()
    return row_per_pid_cid

def get_train_test_dev():
    """ main fubction """
    perspective,evidence,split,merger = get_raw_files()
    merger = pd.merge(merger,split,left_on='cId',right_on='id')
    merger = unwind_merger(merger)
    claim_and_pres = pd.merge(merger,perspective,left_on='pids',right_on='pId',how='inner')
    claim_pres_split = claim_and_pres[['text','perspective','stance_label_3','split']].drop_duplicates()
    train = claim_pres_split[claim_pres_split['split']=='train']
    test = claim_pres_split[claim_pres_split['split']=='test']
    dev = claim_pres_split[claim_pres_split['split']=='dev']
    return train,dev,test