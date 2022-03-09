#给表格额外加数据
import pandas as pd
import os
import shutil
def split_dirs(sourcepath, storepath, formpath):
    df = pd.read_excel(formpath).iloc[:, :2]
    df = df.astype(dtype='str')
    if not os.path.exists(os.path.join(storepath, '0')):
        os.makedirs(os.path.join(storepath, '0'))
        os.makedirs(os.path.join(storepath, '1'))
    for index,row in df.iterrows():
        dir = os.path.join(sourcepath, row['ID'])
        storedir = os.path.join(storepath, row['淋巴结转移阳性'], row['ID'])
        shutil.copytree(dir, storedir)

source = r'\\mega\syli\dataset\EC_all\EC_all_process_data'
store = r'\\mega\syli\dataset\resnet\EC_all'
form = r'\\mega\syli\dataset\EC_all\all_frame/fin_df.xlsx'
split_dirs(source, store, form)








