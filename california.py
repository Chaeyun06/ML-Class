#!/usr/bin/env python
# coding: utf-8

# # 데이터 가져오기

# ## 데이터 다운로드하기

# In[164]:


#데이터를 읽기 위한 판다스 가져오기
import pandas as pd


# In[165]:


#housing을 학번으로 바꿈
#housing 데이터 다운은 kaggle에서 미리 받아놓음 (data디렉토리 밑에 저장)
housing_195002 = pd.read_csv('data/housing.csv', engine='python')


# ## 데이터 구조 훑어 보기

# In[166]:


#head()를 통해 데이터 구조를 훑어볼 수 있음
#각 행은 10개의 특성을 가지고 있음
#위도와 경도, 연식, 방의 개수, 인구수, 세대수, 수입, 집 가격 등의 특성이 있음.
housing_195002.head()


# In[167]:


housing_195002.info()
#총 20640개의 데이터가 존재
#ocean_proximity는 범주형 데이터임(object)
#total_bedrooms은 총 20640개가 있어야 하는데 20433개만 있는 걸 보니 207개의 null이 존재


# In[168]:


housing_195002["ocean_proximity"].value_counts()
#value_counts()은 데이터의 고유값별로 갯수를 알려준다
# 카테고리 종류와 카테고리당 구역(행)의 수
#해안근접도는 5개의 범주로 구분됨


# In[169]:


get_ipython().run_line_magic('matplotlib', 'inline')
#노트북 안에서 그래프를 그려줌


# In[170]:


housing_195002.hist(bins = 50, figsize=(20,15))
#특성들의 데이터 분포를 나타내줌
#bins 값을 높일 수록 세분화 해서 표현됨
#housing_median age, median_house_value 최대값 이상은 똑같은 값으로 치환한 것으로 보임


# In[171]:


housing_195002.corr()
#변수들 간 상관관계를 나타내줌(-1~1사이로)


# In[172]:


#변수들 간 상관관계를 그림으로 한꺼번에 표현해줌.
#from pandas.plotting import scatter_matrix
#scatter_matrix(housing_195002, figsize=(20,15 ))


# # 데이터 전처리

# In[173]:


#total_bedrooms가 null인 것들.
null_index = housing_195002[housing_195002.total_bedrooms.isnull()].index


# In[174]:


#housing 데이터에서 total_bedrooms이 null값인 것들이 있었으므로 이것을 처리해야 함.
#total_rooms와 total_bedrooms의 비율로 null을 채워 넣고자 함.
median = housing_195002["total_bedrooms"].median()
housing_195002["total_bedrooms"].fillna(median, inplace=True)


# In[175]:


housing_195002.iloc[null_index]
#결측치가 모두 435.0으로 채워짐


# In[176]:


#원핫인코딩을 적용해서 ocean_proximity를 문자형에서 숫자형으로 인코딩
housing_195002 = pd.get_dummies(housing)


# In[177]:


housing_195002.info()


# # 새로운 특징 추가

# In[178]:


#가구당 방 수
housing_195002['rooms_per_household'] = housing_195002.total_rooms/housing_195002.households


# In[179]:


#방 수대 침실 수
housing_195002['bedrooms_per_room'] = housing_195002.total_bedrooms/housing_195002.total_rooms
#구당 인원
housing_195002['population_per_household'] = housing_195002.population/housing_195002.households


# In[180]:


housing_195002.head()


# In[181]:


housing_195002.drop(['total_rooms','total_bedrooms'], axis=1, inplace = True)


# In[182]:


housing_195002.corr()['median_house_value'].sort_values(ascending=False)


# ## 테스트 세트 만들기

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_features = housing_195002.drop('median_house_value', axis = 1)
y_target = housing_195002.median_house_value
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target,
                                                   test_size = 0.3,
                                                   random_state = 100)


# In[ ]:


y_train.hist()


# In[ ]:


y_test.hist()


# # 랜덤포레스트, 그리드서치적용

# In[194]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[196]:


rf_reg = RandomForestRegressor(n_estimators=30)

rf_reg.fit(X_train, y_train)

