# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:50:52 2020

@author: FENDI
"""

import numpy as np
import pandas as pd
import jqdatasdk
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from factor_analyzer import FactorAnalyzer, Rotator #做因子分析
from sklearn import preprocessing


risk_free = [3.43, 2.519, 2.3, ]
#3.327, 3.035, 2.564

class fund_analysis:
    
    def __init__(self, fscode):
        self.fscode =fscode
        self.year_list = [2014, 2015, 2016]

    def  get_df(self):    
        df  = get_worthlist(self.fscode)
        #增加date列方便筛选数据
        df['date'] = [pd.to_datetime(i) for i in df.index]
        #获取2014-2019数据
        df = df[(df.date>=pd.to_datetime('20140101')) & (df.date<=pd.to_datetime('20161231'))]
        return df

    def  get_hs300(self):
#        ts.set_token('2b303ca87bfd010bd43137e1ec52485887cb669b3c4bcb0cbebdd017')
#        pro = ts.pro_api()
#        bench = ts.get_hist_data('hs300')
#        bench = bench.sort_index()
        jqdatasdk.auth('17826801367', 'Cye87931798')
        df_300 = jqdatasdk.get_price(security='000300.XSHG', start_date = '2014-01-01', end_date = '2016-12-31',
                                 frequency='daily', fields=['close'])
        df_300['300_return'] = df_300.close/df_300.close.shift(1) - 1
        return df_300
       
        
        
        
        
#获得复权基金净值
    def get_adjnv(self):        
        df_new = self.get_df()
        df2 = df_new.copy()
        day_return = df2['单位净值']/df2['单位净值'].shift(1)  #每日单位净值除以上一日单位净值
        factor =   np.cumprod(day_return)             
        dividend = df2[df2['分红']>0]['分红']  #获得有分红的日期和分红
       
        if len(dividend)>0:
            n = 0  #flag
            #以下循环可以得到每个分红所对应的每日分红再投资值
            for date, value in dividend.items():
                factor_new = factor[date:]
                if n == 0:
                    df3 = pd.DataFrame(factor_new) 
                    df3.columns = ['factor']
                    df3['adj_value %d'%n] = value*factor_new
                else:
                    df3['adj_value %d'%n] = value*factor_new       
                n+= 1
            
            df3.drop('factor', inplace = True, axis = 1)
            div_inv = df3.apply(lambda x: x.sum(), axis=1)  #将每列相加
            df_new['div_inv'] =  div_inv.fillna(0, inplace =True)
            df_new['复权净值'] = df_new['div_inv']+ df_new['单位净值']
        else:
            df_new['复权净值'] = df_new['单位净值']
       
        return df_new
     
 #获得每年的天数  
    def get_yearindex(self):
         #获得每年的天数
         df = self.get_df()
         year_number = []
         for i in self.year_list:
             n = 0
             for j in df.date:
                 if j.year == i:
                     n+= 1
             year_number.append(n-1)
         num = list(np.cumsum(year_number)) #获取每年最后日期的索引
         num.insert(0,0)

         return num


#获取六年的年收益率
    def get_yearreturn(self):
        df = self.get_adjnv()
        num = self.get_yearindex()
        year_return = []
        for i in range(len(num)-1):
            if i==0:
                ret = (df['复权净值'][num[i+1]] - df['复权净值'][num[i]])/df['复权净值'][num[i]]
            else:
                ret = (df['复权净值'][num[i+1]] - df['复权净值'][num[i]+1])/df['复权净值'][num[i]+1]
            year_return.append(ret)
       # year_return = pd.DataFrame(zip(self.year_list, year_return))
        return pd.Series(year_return)
        
#获取收益率的年化波动率
    def get_yearstd(self):
        df = self.get_adjnv()
        num = self.get_yearindex()
        df['日收益率'] = df['复权净值']/df['复权净值'].shift(1) - 1   
        year_std = []
        for i in range(len(num)-1):
            if i==0:
                std = np.std(df['日收益率'][num[i]:num[i+1]])*np.sqrt(245)
            else:
                std = np.std(df['日收益率'][num[i]+1:num[i+1]])*np.sqrt(245)
            year_std.append(std)
#        year_std =  pd.DataFrame(zip(self.year_list, year_std))
        return pd.Series(year_std), df

#获取每年的 sharp ratio   
    def get_sharpratio(self):
        df = self.get_adjnv()
        num = self.get_yearindex()
        df['日收益率'] = df['复权净值']/df['复权净值'].shift(1) - 1   
        sharp_list = []
        for i in range(len(num)-1):
            if i == 0: 
                exReturn = df['日收益率'][num[i]:num[i+1]] - risk_free[i]/100/245
                sharp=np.sqrt(len(exReturn))*exReturn.mean()/exReturn.std()
                
            else:
                exReturn = df['日收益率'][num[i]+1:num[i+1]] - risk_free[i]/100/245
                sharp=np.sqrt(len(exReturn))*exReturn.mean()/exReturn.std()
                
            sharp_list.append(sharp)
        #sharp_list = pd.DataFrame(zip(self.year_list,sharp_list))
        return pd.Series(sharp_list)
        
        
#最大回撤
    def get_MaxDrawdown(self):  
        #numpy:np.maximum.accumulate计算序列累计最大值
        _,df = self.get_yearstd()
        num = self.get_yearindex()
        md_list = []
        for i in range(len(num)-1):
            if i == 0:
                return_list = df['复权净值'][num[i]:num[i+1]]
            else: 
                return_list = df['复权净值'][num[i]+1:num[i+1]]
            start = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
            if start == 0:
                md = 0
            end = np.argmax(return_list[:start])  # 开始位置
            md = (return_list[end] - return_list[start]) / (return_list[end])
            md_list.append(md)
    
        
        #md_list = pd.DataFrame(zip(self.year_list, md_list))
        return pd.Series(md_list)


#alpha: 实际收益和按照Beta系数计算的期望收益之间的差额。
#代表策略多大程度上跑赢了预期的收益率
#beta：相当于业绩评价基准收益的总体波动性
#衡量策略的系统性风险：

    def get_camp(self):
        num = self.get_yearindex()
        _,df = self.get_yearstd()
        y = df['日收益率'].dropna()
        df_300 = self.get_hs300()
        x = df_300['300_return'].dropna()
        merge = pd.merge (y,x, left_index = True, right_index = True, how = 'outer') #因为两者数据长度有不同所以先合并
        merge.fillna(method = 'ffill', inplace = True)
        alpha = []
        beta = []
        #线性回归得到每年的alpha beta
        for i in range(len(num)-1):
            if i == 0:
                merge_new = merge[num[i]:num[i+1]]
                b,a,r_value,p_value,std_err=stats.linregress(merge_new['300_return'], merge_new['日收益率'])
                beta.append(b)
                alpha.append(a)
            else:
                merge_new = merge[num[i]+1:num[i+1]]
                b,a,r_value,p_value,std_err=stats.linregress(merge_new['300_return'], merge_new['日收益率'])
                beta.append(b)
                alpha.append(a)
        #alpha = pd.DataFrame(zip(self.year_list,alpha))
        #beta = pd.DataFrame(zip(self.year_list,beta))
        return  pd.Series(alpha), pd.Series(beta) 
                
        
    def get_informationraio(self):
        df = self.get_adjnv()
        df['日收益率'] = df['复权净值']/df['复权净值'].shift(1) - 1   
        bench = self.get_hs300()
        num = self.get_yearindex()
        merge = pd.merge (df['日收益率'],bench['300_return'], left_index = True, right_index = True, how = 'outer') #因为两者数据长度有不同所以先合并
        merge.fillna(method = 'ffill', inplace = True)
        information_ratio = []
        for i in range(len(num)-1):
            if i == 0: 
                exReturn = merge['日收益率'][num[i]:num[i+1]] - merge['300_return'][num[i]:num[i+1]]
                info = np.sqrt(len(exReturn))*exReturn.mean()/exReturn.std()
                
            else:
                exReturn = merge['日收益率'][num[i]+1:num[i+1]] - merge['300_return'][num[i]+1:num[i+1]]
                info = np.sqrt(len(exReturn))*exReturn.mean()/exReturn.std()
            
            information_ratio.append(info)
        #information_ratio =  pd.DataFrame(zip(self.year_list, information_ratio))
        return pd.Series(information_ratio)
        
        
       
    
        
#c-l模型
#得到选股能力alpha, beta1:熊市beta,beta2:牛市beta， beta2-beta1表示择时能力
    def  get_ClModel(self, size = 0.8, rf = 0.03):
        df = self.get_adjnv()
        df['日收益率'] = df['复权净值']/df['复权净值'].shift(1) - 1   
        bench = self.get_hs300()
        merge = pd.merge (df['日收益率'],bench['300_return'], left_index = True, right_index = True, how = 'outer')
        merge.fillna(method = 'ffill', inplace = True)
        merge.dropna(inplace = True)
        merge['coef1'] =[ min(0, i- rf/245) for i in merge['300_return']]
        merge['coef2'] = [max(0, i- rf/245) for i in merge['300_return']]       
        y = merge['日收益率'] - rf/245
        num = self.get_yearindex()
        xuangu, zeshi = [], [] #选股能力 择时能力
        #线性回归
        for i in range(len(num)-1):
            if i == 0 :
                merge_new = merge[num[i]:num[i+1]]
                y_new = y[num[i] : num[i+1]]
            else:
                merge_new = merge[num[i]+1:num[i+1]]
                y_new = y[num[i]+1 :num[i+1]]
                
            x_train,x_test,y_train,y_test=train_test_split(merge_new[['coef1','coef2']], y_new, train_size = size, random_state=1000)
            model = LinearRegression()
            model.fit(x_train, y_train)
            alpha = model.intercept_ 
            beta= model.coef_
            score = model.score(x_test,y_test) #R方检测
            xuangu.append(alpha)
            zeshi.append(beta[1] - beta[0])
            print(score)  
        
        return pd.Series(xuangu), pd.Series(zeshi)
        
#合并所有指标
    def  get_allindicators(self):
            '''
            indicators: dataframe
            '''
            annual_return = self.get_yearreturn()
            annual_std,_ = self.get_yearstd()
            md = self.get_MaxDrawdown()
            alpha,beta = self.get_camp()
            info_ratio = self.get_informationraio()
            sharp_ratio = self.get_sharpratio()
            xuangu, zeshi = self.get_ClModel()
            indicators=pd.concat([annual_return, annual_std, md, alpha, beta,info_ratio, sharp_ratio, xuangu, zeshi],axis=1, join='outer',
                                  sort='False')
            indicators = indicators.round(5)
            indicators.columns = ['annual_return','annual_std','max drawdown','alpha','beta','infomation ratio','sharp ratio','选股','择时']
            indicators.index= self.year_list
            mean = pd.DataFrame(indicators.mean().T) #简单选取三年平均值
            
            return mean








        




if __name__ == '__main__':  
    fund_list = ['260101','151001','100022','162201','240001','162703','213001','257020','257020','398001'
'161706','161706','161903','162607','460005']
    indicators = pd.DataFrame()
    for i in fund_list():
        a=  fund_analysis('519674')
        mean = a.get_allindicators()
        indicators.append(mean)
  
        
    
    
    



       

