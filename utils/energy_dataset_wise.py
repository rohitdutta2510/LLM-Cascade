import json
import pandas as pd
import numpy as np
import os
import math
from transformers import AutoTokenizer
from carbontracker import parser
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, classification_report
import sys


DATADIR = '../datasets/'
RESDIR = '../temp_outs/'
OUTDIR = '../energy_results/'
BATCH_SIZE = 32

datasets = ['boolq_test_1k']


model_map = {
    "Llama": "meta-llama",
    "Mistral": "mistralai",
    "TinyLlama": "TinyLlama",
    "flan": "google"
}

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)



for datasetname in datasets:
    golddata = pd.read_csv(os.path.join(DATADIR, '%s.csv'%(datasetname)))
    golddata = golddata.sort_values('prompt_text', key = lambda col: col.apply(len))


    combinedout = []

    outdataall = []
    for dn in sorted([dn for dn in next(os.walk(RESDIR))[1] if datasetname in dn]):

        outdata = []
        modelname = dn[dn.rfind('_') + 1:]

        try:
            df = pd.read_csv(os.path.join(RESDIR, dn, 'emissions.csv'))
            energyCC = {row['project_name'] : row['energy_consumed'] for i, row in df.iterrows()}
        
        except Exception as e:
            print(e)
            continue

        for i in range(len(energyCC.keys())):
            outdata.append([modelname + "_%d"%(i)])
            
            try:
                outdata[-1].append(round(energyCC[dn + "_%d"%(i)] * 10**3, 3))
                
            except Exception as e:
                print(e)
                outdata.pop(-1)
                continue



        outdataall.extend(outdata)

        vals = np.array([x[1:] for x in outdata])
        combinedout.append([modelname] + np.mean(vals, axis = 0).tolist() + np.sum(vals, axis = 0).tolist())
        # print(datasetname, combinedout[-1])


    df = pd.DataFrame(outdataall, columns = ["name", "energy_CC(Wh)"])
    df.to_csv(os.path.join(OUTDIR, "%s.csv"%(datasetname)), index = False)


    df2 = pd.DataFrame(combinedout, columns = ["name", "Avg. energy_CC(Wh)", "Total energy_CC(Wh)"])
    df2.to_csv(os.path.join(OUTDIR, "%s_average.csv"%(datasetname)), index = False)


