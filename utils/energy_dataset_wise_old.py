import json
import pandas as pd
import numpy as np
import os
import math
from transformers import AutoTokenizer
from carbontracker import parser
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, classification_report
import sys


# DATADIR = '../../datasets/datasets_train_512/'
# RESDIR = '../../temp_outs/temp_outs_4/'
# OUTDIR = '../../results/results_dataset_wise_pc_4/'
# BATCH_SIZE = 4

DATADIR = '../datasets/'
RESDIR = '../temp_outs/'
OUTDIR = '../energy_results/'
BATCH_SIZE = 32


# DATADIR = '../../datasets/datasets_train_4k/'
# RESDIR = '../../temp_outs/temp_outs_16_energy_eff/'
# OUTDIR = '../../results/results_dataset_wise_energy_prompt/energy_prompt_16_energy_eff'
# BATCH_SIZE = 16


# BATCH_SIZE = int(sys.argv[1])

# DATADIR = '../../datasets/datasets_train_4k'
# RESDIR = '../../temp_outs/temp_outs_bs_%d'%(BATCH_SIZE)
# OUTDIR = '../../results/results_bs/dataset_wise_%d'%(BATCH_SIZE)




# datasets = ['boolq', 'copa', 'qnli', 'cola', 'stsb', 'sst2', 'mnli', 'mrpc', 'wnli', 'cnn_dm']
# datasets = ['dolly-hhrlhf', 'samsum']
# datasets = ['boolq', 'sst2']
# datasets = ['caves', 'vax']
# datasets = ['cnn_dm_complex', 'cnn_dm_easy']
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
        # if 'Llama-2' in modelname:
        #     continue
         
        # modelpath = model_map[modelname[:modelname.find("-")]] + '/' + modelname
        # tokenizer = AutoTokenizer.from_pretrained(modelpath)

        try:
            df = pd.read_csv(os.path.join(RESDIR, dn, 'emissions.csv'))
            energyCC = {row['project_name'] : row['energy_consumed'] for i, row in df.iterrows()}


            # ctdata = parser.parse_all_logs(os.path.join(RESDIR, dn, "carbon_tracker/"))
            # energyCT = [x['actual']['energy (kWh)'] for x in ctdata]
            # kmcarCT = [x['actual']['equivalents']['km travelled by car'] for x in ctdata]
        
            # with open(os.path.join(RESDIR, dn, "output.json")) as fp:
            #     preddata, timestamps = json.load(fp)
        
        except Exception as e:
            print(e)
            continue




        # if not (len(golddata) == len(preddata) and math.ceil(len(golddata) / BATCH_SIZE) == len(timestamps)):
        #     print("IO len mismatch %d and %d and %d %s"%(len(golddata), len(preddata),  len(timestamps), dn))
        #     continue

        for i in range(len(energyCC.keys())):
            outdata.append([modelname + "_%d"%(i)])
            
            try:
                # outdata[-1].append(round(timestamps[i][1] - timestamps[i][0], 3))

                # avginlen = np.mean([len(tokenizer.encode(x)) for x in golddata.prompt_text[i * BATCH_SIZE: (i+1) * BATCH_SIZE]])
                # avgoutlen = np.mean([len(tokenizer.encode(x)) for x in preddata[i * BATCH_SIZE: (i+1) * BATCH_SIZE]])
                # outdata[-1].append(avginlen)
                # outdata[-1].append(avgoutlen)

                # outdata[-1].append(round(energyCT[i] * 10**3, 3))
                outdata[-1].append(round(energyCC[dn + "_%d"%(i)] * 10**3, 3))
                # outdata[-1].append(round(kmcarCT[i] * 10**3, 3))
                
            except Exception as e:
                print(e)
                outdata.pop(-1)
                continue



        outdataall.extend(outdata)

        vals = np.array([x[1:] for x in outdata])
        combinedout.append([modelname] + np.mean(vals, axis = 0).tolist())
        # print(datasetname, combinedout[-1])

    
    # df = pd.DataFrame(outdataall, columns = ["name", "response_time(s)", "avg_input_len", "avg_output_len", "energy_CT(Wh)", "energy_CC(Wh)", "m_car"])
    # df.to_csv(os.path.join(OUTDIR, "%s.csv"%(datasetname)), index = False)


    # df2 = pd.DataFrame(combinedout, columns = ["name", "response_time(s)", "avg_input_len", "avg_output_len", "energy_CT(Wh)", "energy_CC(Wh)", "m_car"])
    # df2.to_csv(os.path.join(OUTDIR, "%s_average.csv"%(datasetname)), index = False)

    df = pd.DataFrame(outdataall, columns = ["name", "energy_CC(Wh)"])
    df.to_csv(os.path.join(OUTDIR, "%s.csv"%(datasetname)), index = False)


    df2 = pd.DataFrame(combinedout, columns = ["name", "energy_CC(Wh)"])
    df2.to_csv(os.path.join(OUTDIR, "%s_average.csv"%(datasetname)), index = False)


