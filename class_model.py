from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from datetime import datetime
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
main_dataset = load_boston()

df = pd.DataFrame(main_dataset.data, columns=main_dataset.feature_names)
df['target_1'] = main_dataset.target
df['target'] = np.where(df['target_1'] >= 25, 1, 0)
del df['target_1']

class my_model():
    def __init__(self, dataset=None):
        if dataset is None:
         self.dataset = {}
        else:
            self.dataset = dataset.copy()
        self.train = None
        self.test = None
        self.validation = None

    def numeric_fill_na(self):
        numeric_columns = list(self.dataset.select_dtypes(
        include="number").columns.values)
        for col in numeric_columns:
            self.dataset[col] = self.dataset[col].fillna(
                0)  # filling missing vlaues with -1
        return self.dataset

    def categorical_fill_na(self):
        cat_columns = list(self.dataset.select_dtypes(
            include="object").columns.values)
        for col in cat_columns:
            self.dataset[col] = self.dataset[col].fillna('UNKNOWN')
        return self.dataset

    def one_hot_encoding_train(self,dataset, normalize=False, levels_limit=200):
        if normalize == True:
            '''Normalize numeric data'''
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.externals import joblib
            '''Get numeric columns'''
            numeric_columns = list(self.train.select_dtypes(
                include="number").columns.values)
            scaler = MinMaxScaler()
            self.train[numeric_columns] = scaler.fit_transform(
                self.train[numeric_columns])
            joblib.dump(scaler, './scaler.pkl')
        import pickle
        '''Collect all the categorical columns'''
        cat_columns = list(self.train.select_dtypes(include="object").columns.values)
        for col in cat_columns:
            column_length = (len(self.train[col].unique()))
            if column_length > levels_limit:
                self.train.drop(str(col), axis=1, inplace=True)
                cat_columns.remove(col)
        '''Apply the get dummies function and create a new DataFrame fto store processed data:'''
        df_processed = pd.get_dummies(self.train, prefix_sep="__",
                                    columns=cat_columns)
        '''Keep a list of all the one hot encodeded columns in order
        to make sure that we can build the exact same columns on the test dataset.'''
        cat_dummies = [col for col in df_processed
                    if "__" in col
                    and col.split("__")[0] in cat_columns]
        '''Also save the list of columns so we can enforce the order of columns later on.'''
        processed_columns = list(df_processed.columns[:])
        '''Save all the nesecarry lists into pickles'''
        with open('cat_columns.pkl', 'wb') as f:
            pickle.dump(cat_columns, f)
        with open('cat_dummies.pkl', 'wb') as f:
            pickle.dump(cat_dummies, f)
        with open('processed_columns.pkl', 'wb') as f:
            pickle.dump(processed_columns, f)
        return self.train

    def one_hot_encoding_test(self, normalize=False):
        if normalize == True:
            '''Normalize numeric data'''
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.externals import joblib
            '''Get numeric columns'''
            numeric_columns = list(self.test.select_dtypes(
                include="number").columns.values)
            scaler = joblib.load('scaler.pkl')
            self.test[numeric_columns] = scaler.fit_transform(
                self.test[numeric_columns])
        import pickle
        '''Process the unseen (test) data!'''
        '''Load nessecary lists from pickles'''
        with open('cat_columns.pkl', 'rb') as f:
            cat_columns = pickle.load(f)
        with open('cat_dummies.pkl', 'rb') as f:
            cat_dummies = pickle.load(f)
        with open('processed_columns.pkl', 'rb') as f:
            processed_columns = pickle.load(f)
        df_test_processed = pd.get_dummies(self.test, prefix_sep="__",
                                        columns=cat_columns)
        for col in df_test_processed.columns:
            if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in cat_dummies:
                print("Removing (not in training) additional feature  {}".format(col))
                df_test_processed.drop(col, axis=1, inplace=True)
        for col in cat_dummies:
            if col not in df_test_processed.columns:
                print("Adding missing feature {}".format(col))
                df_test_processed[col] = 0
        '''Reorder the columns based on the training dataset'''
        df_test_processed = df_test_processed[processed_columns]
        return self.test

    def one_hot_encoding_validation(self, normalize=False):
        if normalize == True:
            '''Normalize numeric data'''
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.externals import joblib
            '''Get numeric columns'''
            numeric_columns = list(self.validation.select_dtypes(
                include="number").columns.values)
            scaler = joblib.load('scaler.pkl')
            self.validation[numeric_columns] = scaler.fit_transform(
                self.validation[numeric_columns])
        import pickle
        '''Process the unseen (test) data!'''
        '''Load nessecary lists from pickles'''
        with open('cat_columns.pkl', 'rb') as f:
            cat_columns = pickle.load(f)
        with open('cat_dummies.pkl', 'rb') as f:
            cat_dummies = pickle.load(f)
        with open('processed_columns.pkl', 'rb') as f:
            processed_columns = pickle.load(f)
        df_test_processed = pd.get_dummies(self.validation, prefix_sep="__",
                                           columns=cat_columns)
        for col in df_test_processed.columns:
            if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in cat_dummies:
                print("Removing (not in training) additional feature  {}".format(col))
                df_test_processed.drop(col, axis=1, inplace=True)
        for col in cat_dummies:
            if col not in df_test_processed.columns:
                print("Adding missing feature {}".format(col))
                df_test_processed[col] = 0
        '''Reorder the columns based on the training dataset'''
        df_test_processed = df_test_processed[processed_columns]
        return self.validation

    def create_train_test_split(self):
        train_1, self.test = train_test_split(
        self.dataset, test_size=0.1, random_state=123)
        '''Now split the train to a smaller train and validation dataset '''
        self.train, self.validation = train_test_split(train_1, test_size=0.15, random_state=123)
  

    def xgb_evaluate(self,min_child_weight, gamma, subsample, colsample_bytree, max_depth, learning_rate, scale_pos_weight, num_round, reg_lambda, reg_alpha):  # max_depth
        data_dmatrix = xgb.DMatrix(data=self.X_train, label=self.Y_train)
        validation_matrix = xgb.DMatrix(data=self.X_validation, label=self.Y_validation)
        max_depth = int(max_depth)
        num_round = int(num_round)
        params = {'objective': 'binary:logistic',
                'eval_metric': 'error',
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'eta': learning_rate,
                'max_depth': max_depth,
                'gamma': gamma,
                'scale_pos_weight': scale_pos_weight,
                'reg_lambda': reg_lambda,
                'reg_alpha': reg_alpha
                }
        bst = xgb.train(params, data_dmatrix, num_round)
        valid_y_pred = bst.predict(validation_matrix)
        valid_y_pred[valid_y_pred > 0.50] = 1
        valid_y_pred[valid_y_pred <= 0.50] = 0
        #accuracy=accuracy_score(Y_validation, valid_y_pred)
        metric = f1_score(self.Y_validation, valid_y_pred)
        #accuracy=recall_score(Y_validation, valid_y_pred)
        #accuracy=precision_score(Y_validation, valid_y_pred)
        return metric
       
    
    def timer(self,start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

    def spliting_X_Y(self,target_variable):
        target_idx = self.train.columns.get_loc(target_variable)
        self.X_train = self.train.loc[:, self.train.columns != target_variable]
        self.Y_train = self.train[self.train.columns[target_idx]]
        target_idx = self.test.columns.get_loc(target_variable)
        self.X_test = self.test.loc[:, self.test.columns != target_variable]
        self.Y_test = self.test[self.test.columns[target_idx]]
        target_idx = self.validation.columns.get_loc(target_variable)
        self.X_validation = self.validation.loc[:, self.validation.columns != target_variable]
        self.Y_validation = self.validation[self.validation.columns[target_idx]]
        

    def train_baysian_optimization(self):
        xgb_bo = BayesianOptimization(self.xgb_evaluate,
                                      {'min_child_weight': (0.5, 100),
                                       'gamma': (0, 10),  # 30
                                          'subsample': (0.6, 1),
                                          'colsample_bytree': (0.4, 1),
                                          'max_depth': (1, 15),  # 25
                                          'learning_rate': (0.05, 0.3),
                                          'scale_pos_weight': (1, 1),
                                          'num_round': (100, 300),
                                          'reg_lambda': (0, 100),
                                          'reg_alpha': (0, 100)
                                       })
        # timing starts from this point for "start_time" variableper
        start_time = self.timer(None)
        xgb_bo.maximize(init_points=20, n_iter=10, acq='ei')
        self.timer(start_time)
        params = xgb_bo.max['params']
        num_round = int(params["num_round"])
        params["max_depth"] = int(params["max_depth"])
        params['eval_metric'] = 'error'
        params['objective'] = 'binary:logistic'
        print("train on hyperparamteres")
        data_dmatrix = xgb.DMatrix(data=self.X_train, label=self.Y_train)
        validation_matrix = xgb.DMatrix(
            data=self.X_validation, label=self.Y_validation)
        test_matrix = xgb.DMatrix(data=self.X_test, label=self.Y_test)
        watchlist = [(validation_matrix, 'eval'), (data_dmatrix, 'train')]
        bst = xgb.train(params, data_dmatrix, num_round,
                        watchlist)
        plot_importance(bst, max_num_features=10, importance_type='weight')


def main():
  my_model_training = my_model(df)
  my_model_training.categorical_fill_na()
  my_model_training.numeric_fill_na()
  my_model_training.create_train_test_split()
  my_model_training.spliting_X_Y('target')
  my_model_training.train_baysian_optimization()


if __name__ == "__main__":
  main()