import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class DataLoader():
    def __init__(self, **kwargs):
        self.data = None
    
    def load_dataset(self, path="C:\\Users\\roshni_sharma\\Documents\\Personal\\Learnings\\XAI learnings\\XAI\\Data\\healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self):
        #One-hot encode all categorical columns
        categorical_col = ["gender",
                           "ever_married",
                           "work_type",
                           "Residence_type",
                           "smoking_status"]
        encoded = pd.get_dummies(self.data[categorical_col],
                                 prefix=categorical_col)
        
        #Update data with new columns
        self.data = pd.concat([encoded, self.data], axis=1)
        self.data.drop(categorical_col, axis=1, inplace=True)

        #Impute missing values of BMI
        self.data.bmi = self.data.bmi.fillna(0)

        #dropping unwanted columns
        self.data.drop(["id"], axis=1, inplace=True)
    
    #spliting the dataset into training and test dataset
    def get_data_split(self):
        x = self.data.iloc[:,:-1]
        y = self.data.iloc[:,-1]
        return train_test_split(x,y,test_size=0.20, random_state=2021)

     #Smapling the dataset
    def oversample(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority')
        #convert Numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np,y_np = oversample.fit_resample(x_np,y_np)
        #convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over

