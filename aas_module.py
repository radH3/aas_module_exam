#!/usr/bin/env python
# coding: utf-8

# In[4]:


#radHE radHE
# lets add libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[191]:


#lets upload files from drive 
app_data_df=pd.read_csv('application_data.csv')
pre_data_df=pd.read_csv('previous_application.csv')


# In[11]:


#lets check metadata of given files
#output shows we have 122 colomns and 307511 rows for application_data.csv
##output shows we have 37 colomns and 1672014 rows for previous_application.csv


print(app_data_df.shape)
print(pre_data_df.shape)


# In[12]:


#detailed metadata for files
app_data_df.info()


# In[13]:


#detailed metadata for files

pre_data_df.info()


# In[14]:


app_data_df.head()


# In[15]:


pre_data_df.head()


# # lets start data cleaning## for application_csv

# In[31]:


#lets check null values

null_df=app_data_df.isnull().sum()
print(null_df)
pd.set_option('display.max_columns',None)


# In[25]:


#total columns having null value
print(len(app_percent))


# In[30]:


#lets calculate the null value percentage
app_percent=round((100*app_data_df.isnull().sum()/len(app_data_df)),3)
pd.set_option('display.max_columns',None)
print(app_percent)


# In[46]:


#lets drop column with 35%of data where data is null
app_data_df=app_data_df.loc[:,app_data_df.isnull().mean()<=.35]


# In[48]:


len(app_data_df.columns)


# In[54]:


#number of columns dropped are 50
dropped_count=122-72
print(dropped_count)


# In[60]:


#lets check tha data again
app_data_df.info()


# In[ ]:





# # # lets start data cleaning## for previous_application_csv

# In[56]:


pre_null_df=pre_data_df.isnull().sum()
print(pre_null_df)
pd.set_option('display.max_columns',None)


# In[57]:


pre_percent=round((100*pre_data_df.isnull().sum()/len(pre_data_df)),3)
pd.set_option('display.max_columns',None)
print(pre_percent)


# In[58]:


#lets frop columns where 25% of data is null
pre_data_df=pre_data_df.loc[:,pre_data_df.isnull().mean()<=.25]


# In[59]:


#remaining columns
len(pre_data_df.columns)


# In[62]:


#lets check the data again
pre_data_df.info()


# # IMPUTING MISSING VALUES FOR application_csv with median , mode,mean

# In[72]:


#Missing Vlues Greater than 25%
misco=app_data_df.isnull().sum()
misco=misco[misco.values>(0.25*len(misco))]
len(misco)


# In[80]:


# repalcing missing values with median 
app_data_df['AMT_ANNUITY'].fillna(app_data_df.AMT_ANNUITY.median(),inplace=True)


# In[63]:


app_data_df['NAME_TYPE_SUITE'].value_counts()


# In[64]:


#checking mode value in the NAME_TYPE_SUITE column
app_data_df['NAME_TYPE_SUITE'].mode()


# In[65]:


#lets fill the 'name_type_suite' column with mode value 'unaccompanied', as it occurs many times in the column.
app_data_df['NAME_TYPE_SUITE'].fillna(value='Unaccompanied',inplace=True)


# In[66]:


#we can see the changes
app_data_df['NAME_TYPE_SUITE'].value_counts()


# In[67]:


#lets check the mean value in AMT_REQ_CREDIT_BUREAU_HOUR column.
app_data_df.AMT_REQ_CREDIT_BUREAU_HOUR.mean()


# In[89]:


#filling AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_DAY , AMT_REQ_CREDIT_BUREAU_WEEK , AMT_REQ_CREDIT_BUREAU_MON,
#AMT_REQ_CREDIT_BUREAU_QRT,AMT_REQ_CREDIT_BUREAU_YEAR with '0' as mean is 0.0064 
app_data_df.loc[:,'AMT_REQ_CREDIT_BUREAU_HOUR':'AMT_REQ_CREDIT_BUREAU_YEAR']=app_data_df.loc[:,'AMT_REQ_CREDIT_BUREAU_HOUR':'AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0)


# In[174]:


#amt_goods_price with mean
app_data_df.AMT_GOODS_PRICE.mean()


# In[175]:


#amt_goods_price with mean

app_data_df['AMT_GOODS_PRICE'].fillna(app_data_df.AMT_GOODS_PRICE.mean(),inplace=True)


# In[85]:


#lets check gender column now
app_data_df.CODE_GENDER.value_counts()


# In[87]:


#replace the'XNA' values with the females as majority are female and xna wont affect our analysis
app_data_df.CODE_GENDER.replace(to_replace='XNA',value='F',inplace=True)


# In[88]:


#checking again the updated gender column
app_data_df['CODE_GENDER'].value_counts()


# In[91]:


#lets check organization column now
app_data_df.ORGANIZATION_TYPE.value_counts()


# In[92]:


#lets drop the rows having 'XNA' values 
app_data_df=app_data_df.drop(app_data_df.loc[app_data_df['ORGANIZATION_TYPE']=='XNA'].index)


# In[93]:


#lets again check updated organization column now
app_data_df.ORGANIZATION_TYPE.value_counts()


# # create bins for the 'AMT_INCOME_TOTAL' and 'AMT_CREDIT'

# In[112]:


# FOR AMT_INCOME_TOTAL
bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

app_data_df['AMT_INCOME_RANGE']=pd.cut(app_data_df['AMT_INCOME_TOTAL'],bins,labels=slot)


# In[113]:


app_data_df['AMT_INCOME_RANGE'].head()


# In[114]:


# FOR AMT_CREDIT
bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
slots = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000 and above']
app_data_df['AMT_CREDIT_RANGE']=pd.cut(app_data_df['AMT_CREDIT'],bins=bins,labels=slots)


# In[97]:


app_data_df['AMT_CREDIT_RANGE'].head()


# In[98]:


app_data_df.head()


# # LETS CHECK IMBALANCE RATIO
# 

# In[99]:


#LETS LOOK INTO TARGET DATA
#0 means non defaulters
#1 means defaulters
app_data_df.TARGET.value_counts()


# In[101]:


#lets plot the TARGET values 
plt.figure(figsize=[12,9])
app_data_df.TARGET.value_counts().plot.barh(color='BLACK')
plt.title('Targets 0 & 1\n', fontsize=20)
plt.xlabel('Count',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.show()


# In[105]:


# Calculating Imbalance percentage
    
# Since the majority is target0 and minority is target1
target_0=app_data_df.loc[app_data_df['TARGET']==0]
target_1=app_data_df.loc[app_data_df['TARGET']==1]

round(len(target_0)/len(target_1),3)


# # UNIVARIATE ANALYSIS FOR TARGET=0 PEOPLE

# In[119]:


#LETS PLOT FOR INCOME RANGE ACROSS VARIOUS GENDER
plt.figure(figsize=[12,7])
sns.set_style('darkgrid')

sns.countplot(data=target_0, x='AMT_INCOME_RANGE',hue='CODE_GENDER',palette='Oranges')
plt.xticks(rotation=90)
plt.title('distribution ofincome range\n', fontsize=15)
plt.xlabel('income range',fontsize=15)
plt.ylabel('number',fontsize=15)
plt.yscale('log')
plt.show()


# In[ ]:


#observations:


#1. Male counts are higher
#2, Income range trom 100000 to 200000 iS having more numoer Of credits,
#3. Less count tor Income range 450000-475000


# In[132]:


#PLotting for the various Income types across various Gender. TARGET 0
plt.figure(figsize=[15, 7])
sns.set_style('whitegrid')
sns.countplot(data=target_0, x='NAME_INCOME_TYPE',hue='CODE_GENDER',palette='mako')
plt.xticks(rotation=45)
plt.title('distribution ofincome range\n', fontsize=15)
plt.xlabel('Income type',fontsize=15)
plt.ylabel('number',fontsize=15)
plt.yscale('log')
plt.show()
    


# In[ ]:


#OBSERVATION FOR TARGET 0 INCOME TYPE

#For income type ‘working’, ’commercial associate’, and ‘State Servant’ the number of credits are higher than others.
#For this Females are having more number of credits than male.
#Less number of credits for income type ‘student’ ,’pensioner’, ‘Businessman’ and ‘Maternity leave’..


# In[122]:


#PLotting for the various Income types across various Gender. TARGET 1
plt.figure(figsize=[15, 7])
sns.set_style('whitegrid')
sns.countplot(data=target_1, x='NAME_INCOME_TYPE',hue='CODE_GENDER',palette='mako')
plt.xticks(rotation=45)
plt.title('distribution ofincome range\n', fontsize=15)
plt.xlabel('Income type',fontsize=15)
plt.ylabel('number',fontsize=15)
plt.yscale('log')
plt.show()
    


# In[ ]:


#Conclusions trom me graph:
#1. For incorne type working,commercial associat and State Servant the number ot credits are higher than other 
#   Maternity leave.
#2. For this Females are having more number or credits tnan male.
#3. Less number of credits income type 'Maternity leave'


# In[126]:


#PLotting for the various various organization types for target 1
plt.figure(figsize=[11, 30])

sns.countplot(data=target_1, y='ORGANIZATION_TYPE',order=target_1['ORGANIZATION_TYPE'].value_counts().index,palette='cool')
plt.xticks(rotation=45,fontsize=15)
plt.title('distribution of various organization types \n', fontsize=15)
plt.xlabel('count',fontsize=15)
plt.ylabel('organization type',fontsize=15)
plt.show()
    


# In[ ]:


#1)Clients which have applied for credits are from most of the organization type ‘Business entity Type 3’ ,
#  ‘Self employed’ ,      ‘Other’ , ‘Medicine’ and ‘Government’.
#2)Less clients are from Industry type 8,type 6, type 10, religion and trade type 5, type 4.
#3)Same as type 0 in distribution of organization type.


# In[130]:


#PLotting for the various various organization types for target 0
plt.figure(figsize=[11, 30])

sns.countplot(data=target_0, y='ORGANIZATION_TYPE',order=target_0['ORGANIZATION_TYPE'].value_counts().index,palette='flare')
plt.xticks(rotation=45,fontsize=15)
plt.title('distribution of various organization types \n', fontsize=15)
plt.xlabel('count',fontsize=15)
plt.ylabel('organization type',fontsize=15)
plt.show()
    


# In[ ]:


#OBSERVATION FOR 
#1)Clients which have applied for credits are from most of the organization type ‘Business entity Type 3’ 
#     ‘Self employed’, ‘Other’ , ‘Medicine’ and ‘Government’.
#2)Less clients are from Industry type 8,type 6, type 10, religion and trade type 5, type 4.


# # correlation between TARGET 0  and 1

# In[142]:


#lets calculate the correlation among the target_0 and target_1 people
target_0_corr=target_0.iloc[0:,2:]
target_1_corr=target_1.iloc[0:,2:]


# In[145]:


target_0=target_0_corr.corr(method='spearman')
target_1=target_1_corr.corr(method='spearman')


# In[147]:


target_0


# In[153]:


#lets plot correlation for target 1
def targets_corr(data,title):
    plt.figure(figsize=(15, 10))
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['axes.titlepad'] = 70

    sns.heatmap(data, cmap="RdYlGn",annot=True)

    plt.title(title)
    plt.yticks(rotation=0)
    plt.show()


# In[155]:


targets_corr(data=target_0,title='Correlation for target 0')


# In[148]:


target_1


# In[157]:


targets_corr(data=target_1,title='Correlation for target 1')


# # find top 10 correlation for target 0 and target 1

# In[158]:


#for target 0
target_0_corr=target_0.iloc[0:,2:].corr()
target_0


# In[161]:


#convert the negative valtues to positive values and sort them
corr_0=target_0_corr.abs().unstack().sort_values(kind='quicksort').dropna()
corr_0=corr_0[corr_0 !=1.0 ]
corr_0


# In[162]:


# TOP 1- CORRELATIONS FOR TARGET 0
corr_0.tail(10)


# In[163]:


#for target 1
target_1_corr=target_0.iloc[0:,2:].corr()
target_1


# In[164]:


#convert the negative valtues to positive values and sort them
corr_1=target_1_corr.abs().unstack().sort_values(kind='quicksort').dropna()
corr_1=corr_1[corr_1 !=1.0 ]
corr_1


# In[165]:


# TOP 10 CORRELATIONS FOR TARGET 1
corr_1.tail(10)


# # BIVARATE ANALYSIS FOR NUM COLUMNS

# In[192]:


plt.figure(figsize=[15,7])
plt.subplot(1,2,1)
sns.scatterplot(target_0.AMT_CREDIT,target_0,AMT_INCOME_RANGE)
plt.title('INCOME VS CREDIT for target=0\n',fontsize=15)
plt.xlabel('\nCredit',fontsize=15)
plt.yscale('log')
plt.ylabel('\nINCOME',fontsize=15)
plt.subplot(1,2,2)
sns.scatterplot(target_1.AMT_CREDIT,target_1,AMT_INCOME_RANGE)
plt.title('INCOME VS CREDIT for target=1\n',fontsize=15)
plt.xlabel('\nCredit',fontsize=15)
plt.ylabel('\nINCOME',fontsize=15)

plt.show()


# In[185]:


plt.figure(figsize=[15,7])
plt.subplot(1,2,1)
sns.scatterplot(x='AMT_CREDIT',y='AMT_GOODS_PRICE',data=target_0)
plt.title('CREDIT VS GOODS PRICE for target=0\n',fontsize=15)
plt.xlabel('\nCredit',fontsize=15)
plt.ylabel('\nGoods Price',fontsize=15)
plt.subplot(1,2,2)
sns.scatterplot(x='AMT_CREDIT',y='AMT_GOODS_PRICE',data=target_1)
plt.title('CREDIT VS GOODS PRICE for target=1\n',fontsize=15)
plt.xlabel('\nCredit',fontsize=15)
plt.ylabel('\nGoods Price',fontsize=15)

plt.show()


# In[ ]:


#with the scatter plot we can determine that AMT CREDIT and AMT GOODS PRICE are highly correlate which means if increase in goods price and the credit increased directly vice versa


# # finding outliners

# # univariate analysis

# In[184]:


# for target 0
#Distribution of income amount
plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data =target_0, y='AMT_INCOME_TOTAL')
plt.title('Distribution of income amount')
plt.show()


# In[188]:


# Disrtibution of credit amount
plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data =target_0, y='AMT_CREDIT')
plt.title('Distribution of credit amount')
plt.show()


# In[ ]:





# In[194]:


# Distribution of anuuity amount
plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data =target_0, y='AMT_ANNUITY')
plt.title('Distribution of Annuity amount')
plt.show()


# In[ ]:




#Few points can be concluded from the graph above.

    #Some outliers are noticed in annuity amount.
    #The first quartile is bigger than third quartile for annuity amount which means 
    #                           most of the annuity clients are from first quartile.


# In[196]:


# Distribution of income amount
plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data =target_1, y='AMT_INCOME_TOTAL')
plt.title('Distribution of income amount')
plt.show()


# # Now merging the Application dataset with previous appliaction dataset
# 

# In[204]:


import warnings
warnings.filterwarnings("ignore")


# In[200]:


new_data_df=pd.merge(left=app_data_df,right=pre_data_df,how='inner',on='SK_ID_CURR',suffixes='_x')


# In[201]:


new_df.head()


# In[202]:


new_df.shape


# In[203]:


new_df.columns


# In[205]:





# In[206]:


merge_df = new_df.rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY',
                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',
                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START','NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV',
                         'AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV',
                         'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',
                         'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'}, axis=1)


# In[207]:


# Removing unwanted columns for analysis

merge_df.drop(['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 
              'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
              'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY','WEEKDAY_APPR_PROCESS_START_PREV',
              'HOUR_APPR_PROCESS_START_PREV', 'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)


# # Performing univariate analysis

# In[208]:


# Distribution of contract status in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of contract status with purposes')
ax = sns.countplot(data = merge_df, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=merge_df['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS',palette='magma') 


# In[ ]:




#Points to be concluded from above plot:

    #Most rejection of loans came from purpose 'repairs'.
    #For education purposes we have equal number of approves and rejection.
    #Payign other loans and buying a new car is having significant higher rejection than approves.


# In[209]:


#Distribution of contract status

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of purposes with target ')
ax = sns.countplot(data = merge_df, y= 'NAME_CASH_LOAN_PURPOSE', 
                  order=merge_df['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'TARGET',palette='magma') 


# In[210]:


#Few points we can conclude from abpve plot:
  #Loan purposes with 'Repairs' are facing more difficulites in payment on time.
    #There are few places where loan payment is significant higher than facing difficulties.They are 'Buying a garage', 'Business developemt', 'Buying land','Buying a new car' and 'Education' Hence we can focus on these purposes for which the client is having for minimal payment difficulties.


# # Performing bivariate analysis

# In[211]:


# Box plotting for Credit amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data =merge_df, x='NAME_CASH_LOAN_PURPOSE',hue='NAME_INCOME_TYPE',y='AMT_CREDIT_PREV')
plt.title('Prev Credit amount vs Loan Purpose')
plt.show()


# In[212]:


#From the above we can conclude some points-

    #The credit amount of Loan purposes like 'Buying a home','Buying a land','Buying a new car' and'Building a house' is higher.
    #Income type of state servants have a significant amount of credit applied
    #Money for third person or a Hobby is having less credits applied for.


# In[213]:


# Box plotting for Credit amount prev vs Housing type in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
sns.barplot(data =merge_df, y='AMT_CREDIT_PREV',hue='TARGET',x='NAME_HOUSING_TYPE')
plt.title('Prev Credit amount vs Housing type')
plt.show()


# In[214]:


#Here for Housing type, office appartment is having higher credit of target 0 and co-op apartment is having higher
#   credit of target 1. So, we can conclude that bank should avoid giving loans to the housing type of co-op apartment 
#   as they are having difficulties in payment. Bank can focus mostly on housing type with parents or House\appartment 
#   or miuncipal appartment for successful payments


# # CONCLUSION
# 

# In[215]:



#1. Banks should focus more on contract type ‘Student’ ,’pensioner’ and ‘Businessman’ with housing ‘type other than 
#   ‘Co-op apartment’ for successful payments.
#2. Banks should focus less on income type ‘Working’ as they are having most number of unsuccessful payments.
#3. Also with loan purpose ‘Repair’ is having higher number of unsuccessful payments on time.
#4. Get as much as clients from housing type ‘With parents’ as they are having least number of unsuccessful payments.


# In[ ]:




