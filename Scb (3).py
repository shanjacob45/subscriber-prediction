#!/usr/bin/env python
# coding: utf-8

# # Term Deposit Marketing

# Data Description:
# 
# The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

# In[3]:


#import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('max_columns', 200)


# In[4]:


#create dataframe for dataset
df = pd.read_csv(r"C:\Users\Shan Jacob\Downloads\term-deposit-marketing-2020.csv")


# In[5]:


df.head()


# Attributes:
# 
# age : age of customer (numeric)
# 
# job : type of job (categorical)
# 
# marital : marital status (categorical)
# 
# education (categorical)
# 
# default: has credit in default? (binary)
# 
# balance: average yearly balance, in euros (numeric)
# 
# housing: has a housing loan? (binary)
# 
# loan: has personal loan? (binary)
# 
# contact: contact communication type (categorical)
# 
# day: last contact day of the month (numeric)
# 
# month: last contact month of year (categorical)
# 
# duration: last contact duration, in seconds (numeric)
# 
# campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# Output (desired target):
# 
# y - has the client subscribed to a term deposit? (binary)

# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


for col in df.select_dtypes(include='object').columns:
    print(col)
    print(df[col].unique())


# In[10]:


features_na = [features for features in df.columns if df[features].isnull().sum() > 0]
for feature in features_na:
    print(feature, np.round(df[feature].isnull().mean(), 4),  ' % missing values')
else:
    print("No missing value found")


# In[11]:


categorical_features=[feature for feature in df.columns if ((df[feature].dtypes=='O') & (feature not in ['deposit']))]
categorical_features


# In[12]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))


# In[13]:


#check count based on categorical features
plt.figure(figsize=(15,80), facecolor='white')
plotnumber =1
for categorical_feature in categorical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.countplot(y=categorical_feature,data=df)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber+=1
plt.show()


# In[14]:


plt.figure(figsize=(15,80), facecolor='white')

ax = plt.subplot(12,3,1)
sns.countplot(y='job',data=df)
plt.xlabel(categorical_feature)
plt.title(categorical_feature)
    
plt.show()


# In[18]:


average_y_by_j = df.groupby('job')['y'].mean()


# In[19]:


average_y_by_j


# In[20]:


average_y_by_j

# Assuming df2 is your DataFrame

# Calculate the count of each category
count_by_jobs = df['job'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'skyblue'
ax1.set_xlabel('job')
ax1.set_ylabel('Average deposit', color=color)
ax1.plot(average_y_by_j.index, average_y_by_j.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_gp.index, count_by_gp.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

# Set the title
plt.title('Average Happiness and Count vs. Satisfaction')

# Show the plot
plt.show()


# In[21]:


for categorical_feature in categorical_features:
    sns.catplot(x='y', col=categorical_feature, kind='count', data= df)
plt.show()


# In[24]:


for categorical_feature in categorical_features:
    print(df.groupby(['y',categorical_feature]).size())


# In[25]:


# list of numerical variables
numerical_features = [feature for feature in df.columns if ((df[feature].dtypes != 'O') & (feature not in ['deposit']))]
print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df[numerical_features].head()


# In[26]:


discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[27]:


df.head()


# In[28]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['deposit']]
print("Continuous feature Count {}".format(len(continuous_features)))


# In[29]:


plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for continuous_feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.histplot(df[continuous_feature],kde=True)
    plt.xlabel(continuous_feature)
    plotnumber+=1
plt.show()


# In[30]:


#boxplot to show target distribution with respect numerical features
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(x="y", y= df[feature], data=df)
    plt.xlabel(feature)
    plotnumber+=1
plt.show()


# In[31]:


plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for numerical_feature in numerical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(df[numerical_feature])
    plt.xlabel(numerical_feature)
    plotnumber+=1
plt.show()


# In[32]:


cor_mat=df.corr()
fig = plt.figure(figsize=(15,7))
sns.heatmap(cor_mat,annot=True)


# In[33]:


df.head()


# In[34]:


sns.countplot(x='y',data=df)
plt.show()


# In[35]:


df['y'].groupby(df['y']).count()


# In[36]:


df2=df.copy()


# In[37]:


df2.head()


# In[38]:


df2.shape


# In[39]:


df2.groupby(['y','default']).size()


# In[40]:


df2.drop(['default'],axis=1, inplace=True)


# In[41]:


df2.head()


# In[42]:


df2.groupby('age',sort=True)['age'].count()


# In[43]:


df2.groupby(['y','balance'],sort=True)['balance'].count()


# In[44]:


df2.groupby(['y','duration'],sort=True)['duration'].count()


# In[45]:


df3.groupby(['y','campaign'],sort=True)['campaign'].count()


# In[46]:


df.head()


# In[47]:


df5=df.copy()


# In[48]:


bool_columns = ['housing', 'loan', 'y']
for col in  bool_columns:
    df5[col+'_new']=df5[col].apply(lambda x : 1 if x == 'yes' else 0)
    df5.drop(col, axis=1, inplace=True)


# In[49]:


df5.tail()


# In[50]:


average_y_by_j = df5.groupby('job')['y_new'].mean()

# Assuming df2 is your DataFrame

# Calculate the count of each category
count_by_jobs = df5['job'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'skyblue'
ax1.set_xlabel('job')
ax1.set_ylabel('Average deposit', color=color)
ax1.plot(average_y_by_j.index, average_y_by_j.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_jobs.index, count_by_jobs.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

# Set the title
plt.title('Average Subscription with jobs')

# Show the plot
plt.show()


# In[51]:


df5['quartile'] = pd.qcut(df5['balance'], 10) 


# In[52]:


df5.head(25)


# In[53]:



# Step 3: Group by Quartiles and Calculate Mean
average_y_by_quartile = df5.groupby('quartile')['y_new'].mean()

count_by_quart = df5['quartile'].value_counts()

# Step 4: Plot the Averages
plt.figure(figsize=(10, 6))
average_y_by_quartile.plot(color='skyblue')
plt.title('Average y-label by Quartile')
plt.xlabel('Quartile')
plt.ylabel('Average y-label')
plt.xticks(rotation=0)  # Optionally, rotate x-axis labels if necessary
plt.show()


# In[54]:


plt.bar(quartiles, counts)
plt.xlabel('Quartile Interval')
plt.ylabel('Count')


# Rotating x-axis labels for better visibility
plt.xticks(rotation=45)

ax2 = plt.gca().twinx()


average_y_by_quartile.plot(color='skyblue')
plt.title('Count by Quartile and Average y-label by Quartile')
plt.xlabel('Quartile')
plt.ylabel('Average y-label')
plt.xticks(rotation=0)  # Optionally, rotate x-axis labels if necessary
plt.show()


# In[ ]:


df5.tail()


# In[55]:


df5['quartile'] = pd.qcut(df5['balance'], 10) 


# In[56]:


average_y_by_d = df5.groupby('default')['y_new'].mean()

# Assuming df2 is your DataFrame

# Calculate the count of each category
count_by_default = df5['default'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'skyblue'
ax1.set_xlabel('default')
ax1.set_ylabel('Average deposit', color=color)
ax1.plot(average_y_by_d.index, average_y_by_d.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_default.index, count_by_default.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

# Set the title
plt.title('Average Subscription with default')

# Show the plot
plt.show()


# In[57]:


average_y_by_c = df5.groupby('campaign')['y_new'].mean()

# Assuming df2 is your DataFrame

# Calculate the count of each category
count_by_c = df5['campaign'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'skyblue'
ax1.set_xlabel('campaign')
ax1.set_ylabel('Average deposit', color=color)
ax1.plot(average_y_by_c.index, average_y_by_c.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_c.index, count_by_c.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

# Set the title
plt.title('Average Subscription with campaign')

# Show the plot
plt.show()


# In[58]:


df5['campaign']>30


# In[59]:


print(df5[df5['campaign'] > 30]['campaign'])


# In[60]:


average_y_by_du = df5.groupby('duration')['y_new'].mean()

# Assuming df2 is your DataFrame

# Calculate the count of each category
count_by_du = df5['duration'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'skyblue'
ax1.set_xlabel('duration')
ax1.set_ylabel('Average deposit', color=color)
ax1.plot(average_y_by_du.index, average_y_by_du.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_du.index, count_by_du.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

# Set the title
plt.title('Average Subscription with duration')

# Show the plot
plt.show()


# In[61]:


df5['quartile_duration'] = pd.qcut(df5['duration'], 10) 


# In[62]:


df5.head()


# In[63]:


average_y_by_dquartile = df5.groupby('quartile_duration')['y_new'].mean()

count_by_dquartile = df5['quartile_duration'].value_counts()

dquartiles = [f'{interval.left}-{interval.right}' for interval in count_by_dquartile.index]
dcounts = count_by_dquartile.values


# In[64]:


plt.bar(dquartiles, dcounts)
plt.xlabel('Duration Quartile Interval')
plt.ylabel('Count')


# Rotating x-axis labels for better visibility
plt.xticks(rotation=45)

ax2 = plt.gca().twinx()


average_y_by_dquartile.plot(color='skyblue')
plt.title('Count by duration Quartile and Average y-label by Quartile')
plt.xlabel('Quartile')
plt.ylabel('Average y-label')
plt.xticks(rotation=0)  # Optionally, rotate x-axis labels if necessary
plt.show()


# In[65]:


df5['quartile_duration'].unique()


# In[66]:


df5['quartile_duration'].min()


# In[67]:


df5['quartile_duration'].sort_values()


# In[68]:


df['job'].unique()


# In[69]:


df3=df.copy()


# In[70]:


def classify_occupation(occupation):
    professional = ['management', 'technician', 'admin', 'services','blue-collar']
    non_professional = ['unknown', 'housemaid', 'student','unemployed']
    
    if occupation in professional:
        return 'Professional'
    elif occupation in non_professional:
        return 'Non-Professional'
    else:
        return 'Other'
df3['occupation_class'] = df['job'].apply(classify_occupation)
df3 = df3.drop(columns=['job'])


# In[71]:


df.head()


# In[72]:


df3.head(15)


# In[73]:


def issalday(day):
    salday = [30,31]
    
    
    if issalday in salday:
        return 'Yes'
    else:
        return 'No'
df3['salary_day'] = df['day'].apply(issalday)
df3 = df3.drop(columns=['day'])


# In[74]:


df3 = df3.drop(columns=['month'])


# In[75]:


df3.head()


# In[76]:


df2.drop(['default'],axis=1, inplace=True)


# In[77]:


df2 = df2[df2['campaign'] < 33]


# In[78]:


df2['campaign'].max()


# In[79]:


df2.head()


# In[80]:


bool_columns = ['housing', 'loan', 'y']
for col in  bool_columns:
    df2[col+'_new']=df2[col].apply(lambda x : 1 if x == 'yes' else 0)
    df2.drop(col, axis=1, inplace=True)


# In[81]:


df2.head()


# In[82]:


df4=df2.copy()


# In[83]:


cat_columns = ['job', 'marital', 'education', 'contact', 'month']
df_encoded = pd.get_dummies(df4, columns=cat_columns)


# In[84]:


df_encoded.head()


# In[85]:


df4.head()


# In[86]:


df_encoded['y_new'].head()


# In[87]:


X = df3.drop('y',axis=1)
Y = df3['y']


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[89]:


X_train.head()


# In[90]:


bool_columns = ['housing', 'loan','default','salary_day']
for col in  bool_columns:
    X_train[col+'_new']=X_train[col].apply(lambda x : 1 if x == 'yes' else 0)
    X_train.drop(col, axis=1, inplace=True)


# In[91]:


bool_columns = ['housing', 'loan','default','salary_day']
for col in  bool_columns:
    X_test[col+'_new']=X_test[col].apply(lambda x : 1 if x == 'yes' else 0)
    X_test.drop(col, axis=1, inplace=True)


# In[92]:


y_train = y_train.apply(lambda x: 1 if x == 'yes' else 0)


# In[93]:


y_test = y_test.apply(lambda x: 1 if x == 'yes' else 0)


# In[94]:


X_train.head()


# In[ ]:





# In[95]:


cat_columns = ['marital', 'education', 'contact', 'occupation_class']
X_train = pd.get_dummies(X_train, columns=cat_columns)


# In[96]:


cat_columns = ['marital', 'education', 'contact', 'occupation_class']
X_test = pd.get_dummies(X_test, columns=cat_columns)


# In[97]:


X_test.head()


# In[157]:


X_train.head()


# In[158]:


X_tr=X_train
X_tr['Subscription'] = y_train
X_tr.head()


# In[167]:


average_y_by_a = X_tr.groupby('age')['Subscription'].mean()

# Assuming df2 is your DataFrame

# Calculate the count of each category
count_by_a = X_tr['age'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'skyblue'
ax1.set_xlabel('duration')
ax1.set_ylabel('Average deposit', color=color)
ax1.plot(average_y_by_a.index, average_y_by_a.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_a.index, count_by_a.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

# Set the title
plt.title('Average Subscription with age')

# Show the plot
plt.show()


# In[165]:


average_y_by_du = X_tr.groupby('duration')['Subscription'].mean()

# Assuming df2 is your DataFrame

# Calculate the count of each category
count_by_du = X_tr['duration'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'skyblue'
ax1.set_xlabel('duration')
ax1.set_ylabel('Average deposit', color=color)
ax1.plot(average_y_by_du.index, average_y_by_du.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_du.index, count_by_du.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

# Set the title
plt.title('Average Subscription with duration')

# Show the plot
plt.show()


# In[98]:


# Initialize the XGBoost classifier
model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)


# In[99]:


y_pred =model.predict(X_test)


# In[100]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[101]:


from sklearn.metrics import confusion_matrix


# Assuming y_pred and y_test are defined
conf_mat = confusion_matrix(y_test, y_pred)

# Create a figure and a set of subplots
plt.figure(figsize=(6, 4))
sns.set(font_scale=1.2)  # Adjust the font scale for better readability
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Yes', 'No'],
            yticklabels=['Yes', 'No'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[102]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_probabilities = model.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# In[114]:


feature_importance = model.feature_importances_


# In[104]:


columnssc = [
    'age',
    'balance',
    'duration',
    'campaign',
    'housing_new',
    'loan_new',
    'default_new',
    'salary_day_new',
    'marital_divorced',
    'marital_married',
    'marital_single',
    'education_primary',
    'education_secondary',
    'education_tertiary',
    'education_unknown',
    'contact_cellular',
    'contact_telephone',
    'contact_unknown',
    'occupation_class_Non-Professional',
    'occupation_class_Other',
    'occupation_class_Professional'
]


# In[105]:


plt.bar(columnssc,feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()


# In[106]:


from sklearn.metrics import classification_report


# Calculate the classification report
report = classification_report(y_test, y_pred)

# Print the report
print(report)


# In[121]:


print(feature_importance)


# In[123]:


feature_importance_table = pd.DataFrame({
    'Feature': columnssc,
    'Importance': feature_importance
})

# Sort the table by importance in descending order
feature_importance_table = feature_importance_table.sort_values(by='Importance', ascending=False)

# Print the table
print(feature_importance_table)


# In[127]:


X_test.head()


# In[ ]:





# In[143]:


average_h_by_s = X_ts.groupby('age')['Subscripiton'].mean()
average_hp_by_s = X_ts.groupby('age')['Subscripiton_predicted'].mean()


# In[ ]:


# Calculate the count of each category
count_by_gp = df2['Satisfication'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'skyblue'
ax1.set_xlabel('Satisfaction')
ax1.set_ylabel('Average Happiness', color=color)
ax1.plot(average_h_by_s.index, average_h_by_s.values, color=color)
ax1.plot(average_hp_by_s.index, average_hp_by_s.values, color='green')
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_gp.index, count_by_gp.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

# Set the title
plt.title('Average Happiness and Count vs. Satisfaction')

# Show the plot
plt.show()


# In[130]:


print(y_pred)


# In[131]:


yp=y_pred.tolist()


# In[134]:


yp.len()


# In[135]:


X_ts=X_test


# In[136]:


X_ts['Subscription_predicted'] = yp


# In[140]:


X_ts['Subscription_predicted'].describe()


# In[141]:


X_ts['Subscription'] = y_test


# In[142]:


X_ts


# In[146]:


average_h_by_s = X_ts.groupby('age')['Subscription'].mean()
average_hp_by_s = X_ts.groupby('age')['Subscription_predicted'].mean()


# In[155]:


count_by_gp = X_ts['age'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'blue'
ax1.set_xlabel('age')
ax1.set_ylabel('Average Subscription', color=color)
line1, = ax1.plot(average_h_by_s.index, average_h_by_s.values, color=color, label='Average Subscription')
line2, = ax1.plot(average_hp_by_s.index, average_hp_by_s.values, color='green', label='Average Subscription predicted')
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_gp.index, count_by_gp.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

lines = [line1, line2]
ax1.legend(lines, [line.get_label() for line in lines], loc='upper left')



# Set the title
plt.title('Average subscription and Count vs. age')

# Show the plot
plt.show()


# In[156]:


average_s_by_d = X_ts.groupby('duration')['Subscription'].mean()
average_sp_by_d = X_ts.groupby('duration')['Subscription_predicted'].mean()

count_by_d = X_ts['duration'].value_counts()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average happiness
color = 'blue'
ax1.set_xlabel('duration')
ax1.set_ylabel('Average Subscription', color=color)
line1, = ax1.plot(average_s_by_d.index, average_s_by_d.values, color=color, label='Average Subscription')
line2, = ax1.plot(average_sp_by_d.index, average_sp_by_d.values, color='green', label='Average Subscription predicted')
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the count
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('Count', color=color)
ax2.bar(count_by_d.index, count_by_d.values, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

lines = [line1, line2]
ax1.legend(lines, [line.get_label() for line in lines], loc='upper left')



# Set the title
plt.title('Average subscription and Count vs. duration')

# Show the plot
plt.show()


# In[ ]:




