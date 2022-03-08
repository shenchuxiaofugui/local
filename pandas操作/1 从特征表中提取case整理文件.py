import os
from pathlib import Path
import pandas as pd
import shutil
formpath = r'\\mega\syli\dataset\EC_all\yuchuli\T1CE\test_numeric_feature.csv'
filepath = r'\\mega\syli\dataset\EC_all\EC_all_process_data'
storepath = r'\\mega\syli\python\nnunet\pre_chai/test'
df = pd.read_csv(formpath).iloc[:,0]
cases = df.values.tolist()
dirs = [i for i in Path(filepath).iterdir()]
for dir in dirs:
    if dir.name in cases:
        shutil.copytree(dir, os.path.join(storepath, dir.name))
