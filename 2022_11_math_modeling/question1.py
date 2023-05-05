import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
pd.set_option('display.float_format',lambda x:'%.2f'%x) #仅显示数据小数点后两位


data=pd.DataFrame()

data1_1=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/国民经济核算/国内生产总值GDP(年).xlsx')
temp=data1_1
temp['年份']=temp['指标名称'].apply(lambda x:x.year)#新建一列【年份】并只显示年份
temp=temp.groupby(temp['年份']).mean()#按年份分组
data=temp.reset_index().copy()

data1_2=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/国民经济核算/中国宏观杠杆率(季).xlsx')
temp=data1_2
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')#与前面数据取并集处理，缺失值填充为NaN

data2_1=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/工业/工业增加值(年).xlsx')
temp=data2_1
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data2_2=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/工业/工业企业分行业_利润总额_累计值.xlsx')
temp=data2_2
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data2_3=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/工业/工业企业分行业_营业利润_累计值.xlsx')
temp=data2_3
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data2_4=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/工业/工业企业分行业_营业收入_累计值.xlsx')
temp=data2_4
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data2_5=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/工业/工业企业分行业_主营业务收入_累计值.xlsx')
temp=data2_5
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data2_6=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/工业/工业企业经济效益指标(月).xlsx')
temp=data2_6
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data2_7=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/工业/工业增加值(年).xlsx')
temp=data2_7
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data3_1=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/价格指数/价格指数(PPI,PPIRM,RPI,CGPI).xlsx')
temp=data3_1
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data3_2=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/价格指数/中国大宗商品价格指数_总指数.xlsx')
temp=data3_2
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data3_3=pd.read_excel('./材料/附件1：宏观经济指标数据（1、2问）/价格指数/CPI.xlsx')
temp=data3_3
temp['年份']=temp['指标名称'].apply(lambda x:x.year)
temp=temp.groupby(temp['年份']).mean()
data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

file_path = './材料/附件1：宏观经济指标数据（1、2问）/银行与货币/'

# tupple(dirpath, dirnames, filenames)<=os.walk(file_path)
# dirpath，string，目录的路径,
# dirnames，list，dirpath下所有子目录的名字,
# filenames，list，非目录文件的名字.
for i in os.walk(file_path):
    for k in i[2]:
        print(file_path + k)
        temp = pd.read_excel(file_path + k)
        temp['年份'] = temp['指标名称'].apply(lambda x: x.year)
        temp = temp.groupby(temp['年份']).mean()

        data = pd.merge(data, temp.reset_index(), on='年份', how='outer')

file_path = './材料/附件1：宏观经济指标数据（1、2问）/利率汇率/'
for i in os.walk(file_path):
    for k in i[2]:
        print(file_path + k)
        temp = pd.read_excel(file_path + k)
        temp['年份'] = temp['指标名称'].apply(lambda x: x.year)
        temp = temp.groupby(temp['年份']).mean()

        data = pd.merge(data, temp.reset_index(), on='年份', how='outer')

file_path = './材料/附件1：宏观经济指标数据（1、2问）/财政/'
for i in os.walk(file_path):
    for k in i[2]:
        print(file_path + k)
        temp = pd.read_excel(file_path + k)
        temp['年份'] = temp['指标名称'].apply(lambda x: x.year)
        temp = temp.groupby(temp['年份']).mean()

        data = pd.merge(data, temp.reset_index(), on='年份', how='outer')

file_path = './材料/附件1：宏观经济指标数据（1、2问）/就业与工资/'
for i in os.walk(file_path):
    for k in i[2]:
        print(file_path + k)
        temp = pd.read_excel(file_path + k)
        temp['年份'] = temp['指标名称'].apply(lambda x: x.year)
        temp = temp.groupby(temp['年份']).mean()

        data = pd.merge(data, temp.reset_index(), on='年份', how='outer')

file_path='./材料/附件1：宏观经济指标数据（1、2问）/景气指数/'
for i in os.walk(file_path):
    for k in i[2]:
        print(file_path+k)
        temp=pd.read_excel(file_path+k)
        temp['年份']=temp['指标名称'].apply(lambda x:x.year)
        temp=temp.groupby(temp['年份']).mean()
        data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

file_path='./材料/附件1：宏观经济指标数据（1、2问）/人口/'
for i in os.walk(file_path):
    for k in i[2]:
        print(file_path+k)
        temp=pd.read_excel(file_path+k)
        temp['年份']=temp['指标名称'].apply(lambda x:int(x))
        del temp['指标名称']
        temp=temp.groupby(temp['年份']).mean()
        data=pd.merge(data,temp.reset_index(),on='年份',how='outer')

data.drop(index=[71,72,73,74],inplace=True)
temp=data[data['年份']>2000]
temp.reset_index(inplace=True,drop=True)
tt=pd.DataFrame(temp.isnull().sum(axis=0)/72 ) #统计每个指标的缺失值占比
temp=temp[tt[tt[0]<0.15].index] #筛选出缺失值占比小于0.15的指标的数据

temp.fillna(method='bfill',inplace=True) #后一个数据填充缺失值
temp.fillna(method='ffill',inplace=True) #前一个数据填充缺失值
for i in temp.columns: #打印并删除每个指标中的缺失值
    if temp[i].nunique()==1:
        print(i)
        del temp[i]

temp.to_excel('汇总数据_填补_2001~2020.xlsx',index=None)

newtemp=pd.DataFrame()
newtemp['年份']=temp['年份']

zhibiao=['GDP:现价', 'GDP:支出法']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['国内生产总值GDP(年)']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['实体经济部门杠杆率', '居民部门杠杆率', '非金融企业部门杠杆率', '政府部门杠杆率', '地方政府杠杆率',
       '金融部门杠杆率(资产方)', '金融部门杠杆率(负债方)']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['中国宏观杠杆率(季)']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['全部工业增加值_x', '全部工业增加值:同比_x', '规模以上工业增加值:同比_x']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['工业增加值（年）']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['煤炭开采和洗选业:利润总额:累计值', '石油和天然气开采业:利润总额:累计值', '黑色金属矿采选业:利润总额:累计值',
       '有色金属矿采选业:利润总额:累计值', '非金属矿采选业:利润总额:累计值', '其他采矿业:利润总额:累计值',
       '农副食品加工业:利润总额:累计值', '食品制造业:利润总额:累计值', '酒、饮料和精制茶制造业:利润总额:累计值',
       '烟草制品业:利润总额:累计值', '纺织业:利润总额:累计值', '纺织服装、服饰业:利润总额:累计值',
       '皮革、毛皮、羽毛及其制品和制鞋业:利润总额:累计值', '木材加工及木、竹、藤、棕、草制品业:利润总额:累计值',
       '家具制造业:利润总额:累计值', '造纸及纸制品业:利润总额:累计值', '印刷业和记录媒介的复制:利润总额:累计值',
       '文教、工美、体育和娱乐用品制造业:利润总额:累计值', '石油、煤炭及其他燃料加工业:利润总额:累计值',
       '化学原料及化学制品制造业:利润总额:累计值', '医药制造业:利润总额:累计值', '化学纤维制造业:利润总额:累计值',
       '非金属矿物制品业:利润总额:累计值', '黑色金属冶炼及压延加工业:利润总额:累计值', '有色金属冶炼及压延加工业:利润总额:累计值',
       '金属制品业:利润总额:累计值', '通用设备制造业:利润总额:累计值', '专用设备制造业:利润总额:累计值',
       '铁路、船舶、航空航天和其他运输设备制造业:利润总额:累计值', '汽车制造:利润总额:累计值', '电气机械及器材制造业:利润总额:累计值',
       '计算机、通信和其他电子设备制造业:利润总额:累计值', '仪器仪表制造业:利润总额:累计值', '其他制造业:利润总额:累计值',
       '废弃资源综合利用业:利润总额:累计值', '电力、热力的生产和供应业:利润总额:累计值', '燃气生产和供应业:利润总额:累计值',
       '水的生产和供应业:利润总额:累计值', '电力、热力、燃气及水的生产和供应业:利润总额:累计值']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['分行业_利润总额']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['非金属矿物制品业:主营业务收入:累计值', '煤炭开采和洗选业:主营业务收入:累计值', '黑色金属矿采选业:主营业务收入:累计值',
       '其他采矿业:主营业务收入:累计值', '废弃资源综合利用业:主营业务收入:累计值', '仪器仪表制造业:主营业务收入:累计值',
       '计算机、通信和其他电子设备制造业:主营业务收入:累计值', '铁路、船舶、航空航天和其他运输设备制造业:主营业务收入:累计值',
       '石油、煤炭及其他燃料加工业:主营业务收入:累计值', '木材加工及木、竹、藤、棕、草制品业:主营业务收入:累计值',
       '纺织服装、服饰业:主营业务收入:累计值', '酒、饮料和精制茶制造业:主营业务收入:累计值', '食品制造业:主营业务收入:累计值',
       '石油和天然气开采业:主营业务收入:累计值', '电气机械及器材制造业:主营业务收入:累计值', '专用设备制造业:主营业务收入:累计值',
       '通用设备制造业:主营业务收入:累计值', '金属制品业:主营业务收入:累计值', '黑色金属冶炼及压延加工业:主营业务收入:累计值',
       '有色金属冶炼及压延加工业:主营业务收入:累计值', '化学纤维制造业:主营业务收入:累计值',
       '化学原料及化学制品制造业:主营业务收入:累计值', '文教、工美、体育和娱乐用品制造业:主营业务收入:累计值',
       '造纸及纸制品业:主营业务收入:累计值', '皮革、毛皮、羽毛及其制品和制鞋业:主营业务收入:累计值', '纺织业:主营业务收入:累计值',
       '其他制造业:主营业务收入:累计值', '烟草制品业:主营业务收入:累计值', '燃气生产和供应业:主营业务收入:累计值',
       '水的生产和供应业:主营业务收入:累计值', '汽车制造:主营业务收入:累计值', '非金属矿采选业:主营业务收入:累计值',
       '有色金属矿采选业:主营业务收入:累计值', '印刷业和记录媒介的复制:主营业务收入:累计值', '医药制造业:主营业务收入:累计值',
       '家具制造业:主营业务收入:累计值', '农副食品加工业:主营业务收入:累计值', '电力、热力的生产和供应业:主营业务收入:累计值',]
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['分行业_主营业务收入']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['工业企业:主营业务收入:累计值', '工业企业:利润总额:累计值', '工业企业:主营活动利润:累计值',
       '工业企业:主营业务成本:累计值', '工业企业:利息费用:累计值', '工业企业:存货', '工业企业:产成品存货',
       '工业企业:应收账款', '工业企业:资产合计', '工业企业:负债合计']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['工业企业经济效益指标（月）']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['全部工业增加值_y', '全部工业增加值:同比_y', '规模以上工业增加值:同比_y']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['工业产品产量_当月值']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['PPI:全部工业品:当月同比', 'PPI:全部工业品:环比', 'PPI:全部工业品:累计同比']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['工业生产者出厂价格指数（PPI）']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['PPIRM:当月同比', 'PPIRM:环比', 'PPIRM:累计同比']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['工业生产者购进价格指数（PPIRM）']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['RPI:当月同比', 'RPI:环比']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['商品零售价格指数（RPI）']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['CGPI:当月同比', 'CGPI:环比']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['企业商品交易价格指数（CGPI）']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['中国大宗商品价格指数:总指数']
newtemp['中国大宗商品价格指数:总指数']=temp[zhibiao]
for i in zhibiao:
    del temp[i]

zhibiao=['CPI:当月同比', 'CPI:累计同比', 'CPI:环比', 'CPI:累计同比.1']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['居民消费价格指数（CPI）']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['公开市场操作:货币净投放', '公开市场操作:货币投放', '公开市场操作:货币回笼']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['公开市场操作(周)']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['人民币存款准备金率:中小型存款类金融机构(变动日期)', '人民币存款准备金率:大型存款类金融机构(变动日期)',
       '正回购数量:91天', '正回购利率:91天', '逆回购数量:7天', '逆回购数量:14天', '逆回购利率:7天',
       '逆回购利率:14天']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['央行货币工具(日)']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['国库现金管理商业银行定期存款:中标量', '国库现金管理商业银行定期存款:到期量',
       '国库现金管理商业银行定期存款:中标利率:3个月', '央行票据:发行量:3个月(发行日)', '央行票据:发行量:3个月(缴款日)',
       '央行票据:发行利率:3个月', '央行票据:未到期量']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['央行货币政策(日)']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['M0', 'M0:同比', 'M1', 'M1:同比', 'M2', 'M2:同比', '现金净投放:当月值', '货币乘数']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['货币供应量']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['超额存款准备金率(超储率):金融机构']
newtemp['超额存款准备金率(超储率):金融机构']=temp[zhibiao]
for i in zhibiao:
    del temp[i]

zhibiao=['银行间同业拆借:加权平均利率:当月值', '银行间同业拆借:加权平均利率:1天:当月值',
       '银行间同业拆借:加权平均利率:7天:当月值', '银行间同业拆借:加权平均利率:14天:当月值',
       '银行间同业拆借:加权平均利率:21天:当月值', '银行间同业拆借:加权平均利率:30天:当月值',
       '银行间同业拆借:加权平均利率:60天:当月值', '银行间同业拆借:加权平均利率:90天:当月值',
       '银行间同业拆借:加权平均利率:120天:当月值', '银行间同业拆借:加权平均利率:6个月:当月值',
       '银行间同业拆借:加权平均利率:9个月:当月值', '银行间同业拆借:加权平均利率:1年:当月值']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['银行间同业拆借利率']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['短期贷款利率:6个月(含)(月)', '短期贷款利率:6个月至1年(含)(月)', '中长期贷款利率:1至3年(含)(月)',
       '中长期贷款利率:3至5年(含)(月)', '中长期贷款利率:5年以上(月)', '个人住房公积金贷款利率:5年以下(含5年)(月)',
       '个人住房公积金贷款利率:5年以上(月)', '活期存款利率(月)', '定期存款利率:3个月(月)', '定期存款利率:6个月(月)',
       '定期存款利率:1年(整存整取)(月)', '定期存款利率:2年(整存整取)(月)', '定期存款利率:3年(整存整取)(月)',
       '定期存款利率:1年(零存整取、整存零取、存本取息)(月)', '定期存款利率:3年(零存整取、整存零取、存本取息)(月)',
       '协定存款利率(月)', '通知存款利率:1天(月)', '通知存款利率:7天(月)']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['人民币存贷款利率']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['SHIBOR:隔夜', 'SHIBOR:1周', 'SHIBOR:2周', 'SHIBOR:3个月', 'SHIBOR:1个月',
       'SHIBOR:6个月', 'SHIBOR:9个月', 'SHIBOR:1年', '银行间同业拆借加权利率:1年',
       '银行间同业拆借加权利率:6个月', '银行间同业拆借加权利率:9个月', '银行间同业拆借加权利率:4个月',
       '银行间同业拆借加权利率:3个月', '银行间同业拆借加权利率:2个月', '银行间同业拆借加权利率:21天',
       '银行间同业拆借加权利率:14天', '银行间同业拆借加权利率:7天', '银行间同业拆借加权利率:1天',
       '银行间质押式回购加权利率:1天', '银行间质押式回购加权利率:7天', '银行间质押式回购加权利率:14天',
       '银行间质押式回购加权利率:21天', '银行间质押式回购加权利率:1个月', '银行间质押式回购加权利率:2个月',
       '银行间质押式回购加权利率:3个月', '银行间质押式回购加权利率:4个月', '银行间质押式回购加权利率:6个月',
       '银行间质押式回购加权利率:9个月', '银行间质押式回购加权利率:1年', '7天回购利率:加权平均:最近1周(B1W)',
       '7天回购利率:加权平均:最近2周(B2W)', '7天回购利率:加权平均:最近1月(B1M)',
       '7天回购利率:算术平均:最近1周(B_1W)', '7天回购利率:算术平均:最近2周(B_2W)',
       '7天回购利率:算术平均:最近1月(B_1M)']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['拆借回购利率']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['公共财政收入:当月值', '财政收支差额:当月值', '税收收入:当月值', '非税收入:当月值', '公共财政支出:当月值',
       '中央本级财政支出:当月值', '地方财政支出:当月值',]
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['国家财政收支']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['就业人员:合计', '经济活动人口', '国家全员劳动生产率']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['就业']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['平均工资:合计', '私营单位就业人员平均工资']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['工资']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['宏观经济景气指数:预警指数', '宏观经济景气指数:一致指数', '宏观经济景气指数:先行指数',
       '宏观经济景气指数:滞后指数']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['景气指数']=newX
for i in zhibiao:
    del temp[i]

zhibiao=['PMI']
newtemp['采购经理指数']=temp[zhibiao]
for i in zhibiao:
    del temp[i]

zhibiao=['消费者信心指数(月)']
newtemp['信心指数']=temp[zhibiao]
for i in zhibiao:
    del temp[i]

zhibiao=['总人口', '总人口:男性', '总人口:女性']
X = temp[zhibiao]
pca = PCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)

newtemp['人口']=newX
for i in zhibiao:
    del temp[i]

newtemp.to_excel('pca_data.xlsx')

# 使用spsspro中的熵权法得到权重列表
temp=pd.read_excel('权重列表.xlsx')
# 选出前5作为高频指标
newtemp[['年份']+temp.head(5)['项'].tolist()].to_excel('高频指标.xlsx',index=None)



