import os
import pandas as pd
number = 3
feature_path=['']*number
df_path=['']*number
select_df=['']*number
feature_path[0]=r'\\mega\syli\dataset\EC_all\yuchuli\T1CE\original\firstorder\result\MinMax\PCC\RFE_2\RFE_train_feature.csv'
df_path[0]=r'\\mega\syli\dataset\EC_all\yuchuli\T1CE\T1CE.csv'
feature_path[1]=r'\\mega\syli\dataset\EC_all\yuchuli\T1CE\original\shape\result\MinMax\PCC\Relief_3\Relief_train_feature.csv'
df_path[1]=r'\\mega\syli\dataset\EC_all\yuchuli\T1CE\T1CE.csv'
feature_path[2]=r'\\mega\syli\dataset\EC_all\yuchuli\T1CE\original\texture\result\Mean\PCC\KW_12\KW_train_feature.csv'
df_path[2]=r'\\mega\syli\dataset\EC_all\yuchuli\T1CE\T1CE.csv'
# feature_path[3]=r'\\mega\syli\dataset\neimo\data\T1+_result\Zscore\PCC\KW_15\_train_feature.csv'
# df_path[3]=r'\\mega\syli\dataset\neimo\data\T1+.csv'
# feature_path[4]=r'\\mega\syli\dataset\neimo\data\T2_result\Zscore\PCC\Relief_3\_train_feature.csv'
# df_path[4]=r'\\mega\syli\dataset\neimo\data\T2.csv'
for i in range(number):
    print(i)
    feature_df=pd.read_csv(feature_path[i])
    feature=feature_df.columns[2:].values.tolist()   #索引行并转换为列表 list(feature_df.index[0])+
    data_df=pd.read_csv(df_path[i])
    feature.insert(0, 'CaseName')
    feature.insert(1, 'label')
    select_df[i]=data_df[feature]
    if i == 0:
        zong_df = select_df[i]
    else:
        zong_df = pd.merge(zong_df,select_df[i],on=('CaseName','label'))
# zong_df=pd.merge(select_df[0],select_df[1],on=('CaseName','label'))  #on可以是两列
# zong_df=pd.merge(zong_df,select_df[2],on=('CaseName','label'))
zong_df.to_csv(r'\\mega\syli\dataset\EC_all\yuchuli\T1CE\zong\zong.csv',index=False)