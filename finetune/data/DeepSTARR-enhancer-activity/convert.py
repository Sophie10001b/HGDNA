import os
import re
import glob
import pandas as pd
import numpy as np

curDir = os.path.dirname(os.path.abspath(__file__))
dataPaths = glob.glob(os.path.join(curDir, "*.parquet"))

tgtMetric = ["Developmental", "HouseKeeping"]
bins = [-10, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10]

for metric in tgtMetric:
    for dataPath in dataPaths:
        dataSplit = os.path.basename(dataPath).split(".")[0]
        datas = pd.read_parquet(dataPath)

        realKey = "Dev_log2_enrichment_scaled" if metric == "Developmental" else "Hk_log2_enrichment_scaled"
        minVal, maxVal =datas[realKey].min(), datas[realKey].max()
        print(f"{dataPath}-{metric}: min: {minVal}-{np.digitize(minVal, bins)-1}, max: {maxVal}-{np.digitize(maxVal, bins)-1}, total num: {len(datas)}")

        cache_regression, cache_generation = [], []
        label_list = []
        for data in datas.itertuples(index=False):
            if metric == "Developmental": activity = data.Dev_log2_enrichment_scaled
            elif metric == "HouseKeeping": activity = data.Hk_log2_enrichment_scaled

            label = np.digitize(activity, bins) - 1
            label_list.append(label)
            cache_regression.append(tuple([data.sequence, activity]))
            cache_generation.append(tuple([data.sequence, label]))
        
        counts = np.bincount(label_list)
        for i in range(len(counts)):
            print(f"{dataPath}-{metric}: label: {i}-{counts[i]}-{counts[i] / sum(counts)}")
        print("------------------------")
        
        df = pd.DataFrame(cache_regression, columns=["sequence", "activity"])
        savePath = os.path.join(curDir, "enhancer_activity_regression", metric)

        if not os.path.exists(savePath): os.makedirs(savePath)
        df.to_parquet(os.path.join(savePath, f"{dataSplit}.parquet"))

        df = pd.DataFrame(cache_generation, columns=["sequence", "label"])
        savePath = os.path.join(curDir, "enhancer_activity_generation", metric)

        if not os.path.exists(savePath): os.makedirs(savePath)
        df.to_parquet(os.path.join(savePath, f"{dataSplit}.parquet"))