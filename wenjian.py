import os
import pandas as pd
from xpinyin import Pinyin


def copy(path, target):
    file = os.path.basename(path)
    with open(path,'rb') as rstream:
        container=rstream.read()
        path1=os.path.join(target,file)
        with open(path1,'wb') as wstream:
            wstream.write(container)


def guina(filepath, new_path):
    file_list = os.listdir(filepath)
    file_list.sort()
    case_name = file_list[0]
    case_name = case_name[:case_name.index('.')]
    new_file = os.path.join(new_path, case_name)
    os.mkdir(new_file)
    for img in file_list:
        name = img[:img.index('.')]
        if name == case_name:
            copy(os.path.join(filepath, img), new_file)
        else:
            case_name = img[:img.index('.')]
            new_file = os.path.join(new_path, case_name)
            os.mkdir(new_file)
            copy(os.path.join(filepath, img), new_file)


def rename(filepath, modals):
    # name = name[name.rfind('.', -1) + 1:]
    filename = os.path.basename(filepath)
    filename = filename.upper()
    #filename = filename.replace('TI', 'T1')
    filename = filename.replace('T1C+', 'T1+')
    if filename[-3:] == 'NII':
        houzhui = '.nii'
    else:
        houzhui = '_roi.nii.gz'
    new_path = os.path.dirname(filepath)
    for modal in modals:
        if modal in filename:
            if modal == 'T1':
                if 'T1+' not in filename:
                    new_name = os.path.join(new_path, 'T1') + houzhui
                    try:
                        os.rename(filepath, new_name)
                    except (Exception, BaseException) as e:
                        print(filepath)
                        print(e)
            else:
                new_name = os.path.join(new_path, modal) + houzhui
                try:
                    os.rename(filepath, new_name)
                except (Exception, BaseException) as e:
                    print(filepath)
                    print(e)


def batch_rename(filepath, modals):
    files = os.listdir(filepath)
    for file1 in files:
        path = os.path.join(filepath, file1)
        dirs = os.listdir(path)
        for file in dirs:
            rename(os.path.join(path, file), modals)


def check(filepath, number, length=25):
    files = os.listdir(filepath)
    for file1 in files:
        path = os.path.join(filepath, file1)
        dirs = os.listdir(path)
        if len(dirs) != number:
            print(path, len(dirs))
        for file in dirs:
            if len(file) > length:
                print(file)


def change_case_name(filepath, line='Name', save=False):
    df = pd.read_excel(filepath)
    p = Pinyin()
    case_name = []
    for i in df[line]:
        case_name.append(p.get_pinyin(i, '_').upper())
    df.insert(df.columns.get_loc('Name') + 1, 'pinyin', case_name)
    if save:
        df.to_csv(filepath, index=False, encoding="utf_8_sig")
    ids = df.set_index('pinyin').iloc[:, 0]
    return ids


def change_file_name(filepath, df_path):
    files = os.listdir(filepath)
    ids = change_case_name(df_path)
    for file in files:
        try:
            new_name = os.path.join(filepath, ids[file])
            os.rename(os.path.join(filepath, file), new_name)
        except (Exception, BaseException) as e:
            print(file)
            print(e)




path = r'\\mega\syli\dataset\neimo\eaoc-seg'
new_path = r'\\mega\syli\dataset\neimo\eaoc'
dfpath = r'\\mega\syli\dataset\neimo\eaoc+final+reading+data2(1).xlsx'
#guina(filepath, new_path)
#batch_rename(new_path, ['T1', 'T2', 'T1+', 'DWI', 'ADC'])
check(new_path, 10)
#(new_path, dfpath)
