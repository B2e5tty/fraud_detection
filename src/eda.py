import pandas as pd
import numpy as np
import logging
import ipaddress
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder


class EDA:
    def __init__(self, dataframe):
        self.info_log = logging.getLogger('info')
        self.info_log.setLevel(logging.INFO)

        info_handler = logging.FileHandler('../logs/info.log')
        info_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        info_handler.setFormatter(info_formatter)
        self.info_log.addHandler(info_handler)

        self.error_log = logging.getLogger('error')
        self.error_log.setLevel(logging.ERROR)

        error_handler = logging.FileHandler('../logs/error.log')
        error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        error_handler.setFormatter(error_formatter)
        self.error_log.addHandler(error_handler)

        try :
            self.df = dataframe
            self.info_log('Loading dataframe')
        except:
            self.error_log.error("Error occurred when loading the dataframe")

    # return the dataframe
    def get_dataframe(self):
        return self.df
    
    # check null and duplicate values
    def null_duplicates(self):
        self.info_log.info('Null and duplicated values in the dataframe')
        print("Null values in the dataframe:")
        print(self.df.isnull().any())
        print('\n')
        print('Duplicated values in the dataframe:')
        print(self.df.duplicated().any())

    # change datatypes
    def change_datatypes(self):
        self.info_log.info('Change datatypes')
        columns = self.df.columns.tolist()

        for col in columns:
            if col == 'signup_time' or col == 'purchase_time':
                self.df[col] = pd.to_datetime(self.df[col])

            elif col in ['source','browser','sex','age','country','class','ip_address','Class']:  # class for credit card dataset
                self.df[col] = self.df[col].astype('category')

            elif col == 'lower_bound_ip_address' or col == 'upper_bound_ip_address':
                self.df[col] = self.df[col].astype(int)

            else: pass

        return self.df.info()
    
    # univariate analysis
    def univariate_analysis(self, turn):
        self.info_log('Univariate Analysis')
        # function for bar plot
        def bar_plot(data, col, title):
            sns.countplot(data, x=col)
            plt.title(title)
            plt.xlabel(col)
            plt.show()

        if turn == 0:
            self.info_log.info('Univariate analysis')
            # unique values in each columns
            columns = self.df.columns.tolist()
            fig = plt.figure(figsize=(10,5))

            for col in columns:
                if col == 'source':
                    bar_plot(self.df,col,'List of sources and number of users')

                elif col == 'browser':
                    bar_plot(self.df,col,'List of browsers')

                elif col == 'sex':
                    bar_plot(self.df,col,'Count of users in each sex group')

                elif col == 'age':
                    print(f'Minimum age in the dataframe: {self.df[col].value_counts().sort_values().head(1)}')
                    print(f'Maximum age in the dataframe: {self.df[col].value_counts().sort_values(ascending=False).head(1)}')

                elif col == 'country':
                    print('Countries of users')
                    print(self.df[col].value_counts())

                elif col == 'class':
                    bar_plot(self.df,col,'Count of users in each class')
                else: pass

        elif turn == 1:
            print('Monthly signup and purchase activity')
            columns = ['signup_time_month','purchase_time_month']

            fig, axs = plt.subplots(2, 1, figsize=(10,8))
            title = ['Signup activity','Transaction activity']

            for i,col in enumerate(columns):
                sns.countplot(data=self.df, x=col, ax=axs[i])
                axs[i].set_xlabel(col)
                axs[i].set_title(title[i])

            plt.tight_layout()
            plt.show()

        elif turn == 2:
            print('Day of the week signup and purchase activity')
            columns = ['signup_time_day_of_week','purchase_time_day_of_week']

            fig, axs = plt.subplots(2, 1, figsize=(10,8))
            title = ['Signup activity','Transaction activity']

            for i,col in enumerate(columns):
                sns.countplot(data=self.df, x=col, ax=axs[i])
                axs[i].set_xlabel(col)
                axs[i].set_title(title[i])

            plt.tight_layout()
            plt.show()

        elif turn == 3:
            print('Time of the day signup and purchase activity')
            columns = ['signup_time_d','purchase_time_day_of_week']

            plt.figure(figsize=(10,8))
            sns.countplot(data=self.df, x='time_of_day')
            plt.xlabel('Time of Day')
            plt.title('Transaction activity')

            plt.tight_layout()
            plt.show()
    # bivariate analysis
    def bivariate_analysis(self,turn):
        self.info_log('Bivariate Analysis')
        # sex per class
        if turn == 0:
            print('Number of sex per class')
            print(self.df.groupby(['class','sex'])['class'].count())

        # class per countries
        elif turn == 1:
            df_class_0 = self.df[self.df['class'] == 0]
            df_class_1 = self.df[self.df['class'] == 1]

            countries_in_class0 = df_class_0['country'].unique()
            countries_in_class1 = df_class_1['country'].unique()

            total_users_per_countries = self.df['country'].value_counts()
            nonfraud_users_per_country = df_class_0.groupby('country')['user_id'].count().sort_values(ascending = False).reset_index()
            fraud_users_per_country = df_class_1.groupby('country')['user_id'].count().sort_values(ascending = False).reset_index()
            

            proportion_class0 = {}
            proportion_class1 = {}

            for c in countries_in_class0:
                proportion_class0[c] = [round((int(nonfraud_users_per_country[nonfraud_users_per_country['country'] == c]['user_id'].values) / int(total_users_per_countries[c]))*100,2),total_users_per_countries[c]]

            for c in countries_in_class1:
                proportion_class1[c] = [round((int(fraud_users_per_country[fraud_users_per_country['country'] == c]['user_id'].values) / int(total_users_per_countries[c]))*100,2),total_users_per_countries[c]]

            proportion_class0_df = pd.DataFrame(proportion_class0).T
            proportion_class1_df = pd.DataFrame(proportion_class1).T

            # proportion_df
            print("Top 10 Countries with most non-fraud users")
            print(proportion_class0_df.sort_values(by=[1,0], ascending=False).head(10)) 

            print('\n')

            print("Top 10 Countries with most fraud users")
            print(proportion_class1_df.sort_values(by=[0,1], ascending=False).head(10)) 

        # fraud activity per browser
        elif turn == 2:
            fig, axs = plt.subplots(2, 3, figsize=(10,5))
            axs = axs.flatten()
            browsers = self.df['browser'].unique()

            for i,bro in enumerate(browsers):
                browser_df = self.df[self.df['browser'] == bro]
                sns.countplot(data=browser_df, x='class',ax=axs[i])
                axs[i].set_xlabel('Non-Fraud and Fraud')
                axs[i].set_ylabel(bro)
                axs[i].set_title(bro + ' fraud activity')
                    
            plt.tight_layout()        
            plt.show()
                
        # time feature analysis
        elif turn == 3:
            # function for bar plot
            def bar_plot(data, col, title):
                plt.figure(figsize=(10,8))
                sns.countplot(data, x=col)
                plt.title(title)
                plt.xlabel(col)
                plt.show()

            print("Which days does most fraud happens")
            fraud_df = self.df[self.df['class'] == 1]
            bar_plot(fraud_df,'purchase_time_day_of_week', 'Days fraud transactions')

            print("Which time of the day does most fraud happens")
            fraud_df = self.df[self.df['class'] == 1]
            bar_plot(fraud_df,'time_of_day', 'Days fraud transactions')

    # feature engineering
    def feature_eng(self, turn):
        self.info_log('Feature Engineering')
        if turn == 0:
            def categorize_time_of_day(hour):
                if 5 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 17:
                    return 'Noon'
                elif 17 <= hour < 21:
                    return 'Evening'
                elif 21 <= hour < 24:
                    return 'Night'
                else:
                    return 'Midnight'
                
            datetime_columns = self.df.select_dtypes(include=['datetime']).columns

            for date in datetime_columns:
                self.df[date + '_month'] = self.df[date].dt.strftime('%B')
                self.df[date + '_day_of_week'] = self.df[date].dt.day_name()
                self.df[date + '_hour'] = self.df[date].dt.hour
            
            self.df['time_of_day'] = self.df['purchase_time_hour'].apply(categorize_time_of_day)

        

            # show
            return self.df.head()
        
    # scale
    def scale(self):
        self.info_log('Scaling amount')
        scaler = StandardScaler()
        self.df['Amount'] = scaler.fit_transform(self.df['Amount'])

    # encoding categorical features
    def encode(self):
        le = LabelEncoder()
        columns = self.df.select_dtypes(include='category').columns

        for col in columns:
            self.df[col] = le.fit_transform(self.df[col])

    # closes logging
    def close_log(self):
        self.info_log.info('Closing logging')
        handlers = self.info_log.handlers[:]

        # close info logging 
        for handler in handlers:
            handler.close()
            self.info_log.removeHandler(handler)

        # close error logging
        handlers = self.error_log.handlers[:]
        for handler in handlers:
            handler.close()
            self.error_log.removeHandler(handler)

def merge_dataframe(dataframeOne, dataframeTwo):
    # convert the ip address in each dataset to integer
    dataframeTwo['lower_bound_ip_address'] = dataframeTwo['lower_bound_ip_address'].apply(lambda x: int(float(x)))
    dataframeTwo['upper_bound_ip_address'] = dataframeTwo['upper_bound_ip_address'].apply(lambda x: int(float(x)))
    dataframeOne['ip_address'] = dataframeOne['ip_address'].apply(lambda x: int(float(x)))

    # check
    dataframeOne.info()
    print('\n')
    dataframeTwo.info()

    # merge the data
    dataframeTwo.sort_values('lower_bound_ip_address', inplace=True)
    dataframeOne.sort_values('ip_address', inplace=True)
    # fraud_on
    result = pd.merge_asof(dataframeOne, dataframeTwo, left_on='ip_address', right_on='lower_bound_ip_address', direction='backward')

    result = result[(result['ip_address'] >= result['lower_bound_ip_address']) & (result['ip_address'] <= result['upper_bound_ip_address'])]

    return result