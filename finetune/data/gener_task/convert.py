import os
import random
import glob
import pandas as pd
import numpy as np

RATE = 0.1

if __name__ == '__main__':
    random.seed(17)
    curDir = os.path.dirname(os.path.abspath(__file__))

    fileList = glob.glob(curDir + "/*/*.parquet")

    for _file in fileList:
        baseDir = os.path.dirname(_file)
        df = pd.read_parquet(_file)
        if "test" in _file:
            print(f"{baseDir}|test: {len(df)}, seqlen: {sum(len(_) for _ in df['sequence']) / len(df)}")
            continue

        dev = df.sample(frac=RATE, random_state=17)
        train = df.drop(dev.index)

        print(f"{baseDir}|train: {len(train)}, seqlen: {sum(len(_) for _ in train['sequence']) / len(train)}")
        print(f"{baseDir}|dev: {len(dev)}, seqlen: {sum(len(_) for _ in dev['sequence']) / len(dev)}")

        train.to_parquet(os.path.join(baseDir, "train_new.parquet"), index=False)
        dev.to_parquet(os.path.join(baseDir, "dev_new.parquet"), index=False)