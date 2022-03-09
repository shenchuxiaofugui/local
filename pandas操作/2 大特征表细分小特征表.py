import os
import pandas as pd

filepath = r'\\mega\syli\dataset\EC_all\yuchuli'
modal_list = ['DWI', 'T1CE', 'T2']
feature_list = ['log-sigma', 'wave', 'original']

for modal in modal_list:
    modalpath = os.path.join(filepath, modal)
    zong = pd.read_csv(modalpath + f'\\{modal}.csv')
    df3 = zong.iloc[:, :2]
    for feature in feature_list:
        #os.makedirs(modalpath + f"\\{feature}\\firstorder\\result")
        os.makedirs(modalpath + f"\\{feature}\\shape\\result")
        #os.makedirs(modalpath + f"\\{feature}\\texture\\result")
        feature_df = zong.filter(regex=f'{feature}')
        df2 = feature_df.filter(regex='^(?!.*?firstorder).*$')  # ^不匹配，?!不包含,.*?非贪婪，.*所有内容
        df2 = df2.filter(regex='^(?!.*?shape).*$')
        texture_df = pd.merge(df3, df2, left_index=True, right_index=True)
        texture_df.to_csv(modalpath + f'\\{feature}/texture\\{modal}_{feature}_Texture.csv', index=False)
        df4 = feature_df.filter(regex='firstorder')
        firstorder_df = pd.merge(df3, df4, left_index=True, right_index=True)
        firstorder_df.to_csv(modalpath + f'\\{feature}/firstorder\\{modal}_{feature}_firstorder.csv', index=False)
        if feature == 'original':
            df5 = feature_df.filter(regex='shape')
            shape_df = pd.merge(df3, df5, left_index=True, right_index=True)
            shape_df.to_csv(modalpath + f'\\{feature}/shape\\{modal}_{feature}_shape.csv', index=False)