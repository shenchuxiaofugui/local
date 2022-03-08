#给表格额外加数据
import pandas as pd
import numpy as np
import os
os.makedirs(r'\\mega\syli\dataset\EC_all\all_frame\hahahah')
# old_df = pd.read_excel(r'\\mega\syli\dataset\EC_all\all_frame\fin_df.xlsx')
# new_df_1 = pd.read_excel(r'\\mega\syli\dataset\EC_all\all_frame\ori\biao4.xlsx')[['ID', '脉管内癌栓']]
# new_df_2 = pd.read_excel(r'\\mega\syli\dataset\EC_all\all_frame\ori\biao6.xlsx')[['ID', '脉管内癌栓']]
# # for index, row in new_df_2.iterrows():
# #     #new_df_2.loc[index, 'ID'] = row['ID'][1:]
# #     print(row['ID'])
# old_df['ID'] = old_df['ID'].astype(dtype='str')
# new_df_1['ID'] = new_df_1['ID'].astype(dtype='str')
# new_df_2['ID'] = new_df_2['ID'].astype(dtype='str')
#
# # print(new_df_2.head())
# all_df_1 = pd.merge(old_df, new_df_1)  #, on='ID'
# all_df_2 = pd.merge(old_df, new_df_2)
# all_df = pd.concat([all_df_1, all_df_2])
# a = old_df['ID'].values.tolist()
# b = all_df['ID'].values.tolist()
# print(len(list(set(a).difference(set(b)))))
# df = pd.read_excel(r'\\mega\syli\dataset\all_frame\fin_df_1.xlsx')
# for index, row in df.iterrows():
#     if row['病理类型'] != 1:
#         df.loc[index, '病理类型'] = 0
# df.to_excel(r'\\mega\syli\dataset\all_frame/fin_df_2.xlsx', index=False)








