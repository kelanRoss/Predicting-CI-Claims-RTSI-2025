#Proposed approach for probabilistic modelling of critical illness insurance claims.
#Approach structured as follows:
# 1. Data partitioning - Splits the dataset based on a defined year. 
#    All coverages ending after this year form the test set while all ending before the year form the training set.
# 2. Removal of duplicates (coverages, clients) - This was handled in the SQL code.
# 3. Outlier handling - Winsorisation is used to adjust columns based on defined inlier bounds.
# 4. Handling of missing data - This was handled in the SQL code.
# 5. Feature engineering - Some features were created in the SQL and python notebook. 
#    Health claim aggregated features are created as part of this code at runtime.
# 6. Min-max standardisation of feature values.
# 7. One of 5 machine learning models:
#    i. Logistic Regression with Elastic Net (LREN).
#    ii. Support Vector Machine (SVM).
#    iii. Extreme Gradient Boosting (XGBoost).
#    iv. Random Forest (RF). 
#    v. Artificial Neural Network (ANN).
# 8. L1, L2 and implicit regularisation through the machine learning algorithm implementations.
# 9. Nested TimeSeriesSplit CV for hyperparameter optimisation through Randomised Grid Search, and
#    Probability Calibration.
# 10. Probability estimation and evaluation with  the Log Loss Score (LLS), Log Loss Skill Score (LLSS) and ROC-AUC.
# 11. Ranking of clients by risk and evaluation with Precision-at-K (PAK) and Recall-at-K (RAK)

#Setup - libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.calibration import CalibratedClassifierCV

#Setup - external scripts
from create_data_partitions import create_train_test, create_claim_train_test

#Scikit-learn pipeline requires each step to be a class with fit and transform methods.
#Helper class for implementing outlier handling through Winsorisation
class WinsorisationClass:
    def __init__(self, bounds):
        #Bounds are the limits used in winsorisation
        #Expected format is an array of tuples
        #Example:
        #[(column1, lower_bound, upper_bound), (column2, lower_bound, upper_bound)]
        self.bounds = bounds
        
    def fit(self, df, y=None):
        return self
    
    def transform(self, data):
        df = data.copy()
        for column, lower_bound, upper_bound in self.bounds:
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound
        return df


class ProposedMethodology:
    def __init__(self, df, model_cols, outlier_bounds, claims=None, merge_on='CVG_ID', runs=10):
        self.df=df #Dataset of insurance coverages
        self.model_cols=model_cols #Modelling columns
        self.test_years=[2023] #List of years that form the test sets
        self.year=2023 #Current year being tested
        self.test_intervals=[0, 90, 180, 360] #List of date offset values
        self.interval=0 #Current date offset
        self.outlier_bounds=outlier_bounds #Outlier bound values for Winsorisation
        self.claims=claims #Health claim dataset
        self.merge_on=merge_on #Field to merge dataset on. 
                               #Either CVG_ID to have one row per coverage or MDM_ID to have one row per client.
        self.runs=runs #Number of times to repeat the experiment

    
    #Allow user to specify test years and intervals, as well as current working year and interval for single tests
    def set_params(self, test_years=[2023, 2024], year=2023, test_intervals=[0,90,180,360], interval=0):
        self.test_years=test_years
        self.year=year
        self.test_intervals=test_intervals
        self.interval=interval
        return self

    
    #Function to obtain data partitions. Based on whether the experiement includes health claims or not.
    #Full details in create_data_partitions.py
    def train_test_split(self):
        if self.claims is None:
            X_feat, X_label, Y_feat, Y_label, Y_id = create_train_test(self.df, self.model_cols, self.year, self.interval)
        else:
            X_feat, X_label, Y_feat, Y_label, Y_id = create_claim_train_test(self.df, self.claims, self.model_cols, self.year, self.interval)
        
        return(X_feat, X_label, Y_feat, Y_label, Y_id)

    
    #Nested cross-validation implementation
    #In this implementation, the outer cv loop is a randomised grid search cv for hyperparamter optimisation,
    #while the inner cv loop is a calibrated classifier cv for probability calibration.
    #The classifier is also setup and embedded in the inner cv.
    #The classifier is implemented as a pipeline with the following steps:
    #1. Outlier handling via Winsorisation.
    #2. Min-max standardisation.
    #3. The machine learning algorithm implementation, clf.

    #clf is the classifier to be used
    #cv_params should contain the parameter grid for the grid search
    #n_iter is the number of iterations to use in the randomised grid search
    #folds is the number of inner and outer cv folds
    def create_nested_cv(self, clf, cv_params, n_iter, folds=5):
        n_jobs = -1
        #Init outlier step
        outlier_step = WinsorisationClass(self.outlier_bounds)
        #Init standardisation step. Only columns prefixed with "SCALAR_" are affected.
        scaler =  MinMaxScaler()
        scaler_step = make_column_transformer((scaler, make_column_selector(pattern='SCALAR_')), remainder='passthrough', n_jobs=n_jobs)
        #Define pipeline for classifier
        process = [
            ('outlier_removal', outlier_step),
            ('scaler', scaler_step),
            ('clf', clf)
        ]
        pipeline = Pipeline(process)

        #Both the inner and outer cv loops are TimeSeriesSplitCV implementations
        outer_cv = TimeSeriesSplit(n_splits=folds)
        inner_cv = TimeSeriesSplit(n_splits=folds)

        #Calibration inner cv. Probability calibration via logistic/Platt method.
        ccv = CalibratedClassifierCV(estimator=pipeline, 
                                     method='sigmoid',
                                     cv=inner_cv,
                                     n_jobs=n_jobs,
                                     ensemble=True)

        #Randomised grid search outer cv, optimised on the negative log loss metric
        nested_cv = RandomizedSearchCV(estimator=ccv, 
                                       n_iter=n_iter, 
                                       param_distributions=cv_params, 
                                       cv=outer_cv, 
                                       n_jobs=n_jobs, 
                                       scoring='neg_log_loss')
    
        return nested_cv

    
    #Here we define an implementation which gives a single execution of the proposed approach
    def classification_method(self, clf, cv_params, n_iter, folds=5):
        #df for storing results
        scores = pd.DataFrame(columns=['year', 'interval', 'log_loss', 'log_skill_score', 'roc_auc'])
        #df for storing hyperparameters returned by grid search
        search_results = pd.DataFrame(columns= ['year', 'interval'] + list(cv_params.keys()))

        #Call train_test_split to create data partitions
        X_feat, X_label, Y_feat, Y_label, _ = self.train_test_split()

        #Create a nested cv instance for the experiment
        nested_cv = self.create_nested_cv(clf=clf, cv_params=cv_params, n_iter=n_iter, folds=folds)
        nested_cv.fit(X=X_feat, y=X_label) #Fit to the training data
        best_params = nested_cv.best_params_ #Extract best hyperparameter results from grid search
        best_params.update({'year': self.year, 'interval': self.interval})
        search_results.loc[len(search_results), :] = pd.Series(best_params) #Save params to results df

        #Extract probability estimates from fit model, calculate log loss, log skill score and roc-auc
        y_pred = nested_cv.predict_proba(X = Y_feat)[:,1]
        log_score = log_loss(Y_label, y_pred)
        auc = roc_auc_score(Y_label, y_pred)
        #Calculate log skill score by using a dummy classifier output
        base_preds = np.repeat(Y_label.mean(), len(Y_label))
        log_base = log_loss(Y_label, base_preds)
        log_skill = 1 - (log_score/log_base)

        #Save metric to results df
        metrics = [self.year, self.interval, log_score, log_skill, auc]
        scores.loc[len(scores), :] = metrics
        
        return(scores, search_results)
    

    #Now a method is setup to obtain repeated probability estimates in a single experiment
    #This is controlled by the "runs" class property.
    def probability_estimation(self, clf, cv_params, n_iter, folds=5):
        #Setup dataframes for probability evaluation results and grid search results
        scores = pd.DataFrame(columns=['year', 'interval', 'log_loss', 'log_skill_score', 'roc_auc'])
        search_results = pd.DataFrame(columns= ['year', 'interval'] + list(cv_params.keys()))

        #Begin the complete test. One year forms one test set.
        #Experiments are also repeated for each offset interval.
        #Again experiments are repeated for "self.runs" times.
        for year in self.test_years:
            self.year = year
            for interval in self.test_intervals:
                self.interval = interval
                for i in range(0, self.runs, 1):
                    #Get a single probability estimate result
                    run_score, run_search = self.classification_method(clf=clf, 
                                                                       cv_params=cv_params,
                                                                       n_iter=n_iter,
                                                                       folds=folds,
                                                                      )
                    #Attach single result to complete result set
                    scores = pd.concat([scores, run_score], ignore_index=True)
                    search_results = pd.concat([search_results, run_search], ignore_index=True)
                    
        #Create mean scores dataframe
        mean_scores = scores.dropna(axis=1).groupby(['year', 'interval']).mean().reset_index()        
        return(scores.convert_dtypes(), mean_scores.convert_dtypes(), search_results.convert_dtypes())


    #To assess the performance of any classifier, it is useful to compare it to a dummy classifier
    def dummy_results(self):
        scores = pd.DataFrame(columns=['year', 'log_loss', 'roc_auc'])
        for year in self.test_years:
            self.year = year
            #Call train_test_split to create data partitions
            X_feat, X_label, Y_feat, Y_label, _ = self.train_test_split()
            #Repeat the same probability estimate for all coverages
            base_preds = np.repeat(Y_label.mean(), len(Y_label))
            #Calculate the log loss and roc-auc for the dummy clf
            ll = log_loss(Y_label, base_preds)
            ra = roc_auc_score(Y_label, base_preds)
            scores.loc[len(scores), :] = [year, ll, ra]
        return scores.convert_dtypes()


    #Function to return an optimised classifier via nested cv. For observing feature importance scores.
    def get_best_estimator(self, clf, cv_params, n_iter, folds=5):
        n_jobs = -1
        X_feat, X_label, *_ = self.train_test_split()
        #Create a nested_cv instance
        nested_cv = self.create_nested_cv(clf=clf, cv_params=cv_params, n_iter=n_iter, folds=folds)
        nested_cv.fit(X=X_feat, y=X_label) #Fit to data
        #Extract optimised classifier, this is an ensemble due to CalibratedClassifierCV
        best_estimator = nested_cv.best_estimator_.calibrated_classifiers_
        return best_estimator


    #Function to obtain a single set of probability estimates and rank clients by risk level
    def ranking_analysis(self, clf, cv_params, n_iter, folds):
        #Get data partitions
        X_feat, X_label, Y_feat, Y_label, Y_id = self.train_test_split()
        #Get probability estimates from best classifier
        nested_cv = self.create_nested_cv(clf=clf, cv_params=cv_params, n_iter=n_iter, folds=folds)
        nested_cv.fit(X=X_feat, y=X_label)
        y_pred = nested_cv.predict_proba(X=Y_feat)[:,1]

        #Create results df
        rankscores = Y_feat.copy()
        rankscores.loc[:, 'CVG_ID'] = Y_id.copy()
        rankscores.loc[:, 'y_true'] = Y_label.copy()
        rankscores.loc[:, 'y_pred'] = y_pred.copy()
        
        #Prepare results
        #Remove label used for standardisation step
        rankscores.columns = [col.replace('SCALAR_', '').replace('CATEGORY_', '') for col in rankscores.columns]

        #Specify columns desired for results display
        out_cols = ['CVG_ID', 'y_true', 'y_pred']
        
        #Rank items by probability estimate
        # rankscores = rankscores.loc[:, out_cols].sort_values('y_pred', ascending=False)

        #Precision-at-K and Recall-at-K analysis
        #Calculate for K=10
        total_items = Y_label.sum()
        true_items = rankscores.nlargest(10, 'y_pred', 'all').loc[:, 'y_true'].sum()
        pak10 = true_items/10
        rak10 = true_items/total_items

        #Calculate for K=20
        true_items = rankscores.nlargest(20, 'y_pred', 'all').loc[:, 'y_true'].sum()
        pak20 = true_items/20
        rak20 = true_items/total_items

        #Calculate for K = number of expected CI claimants based on training data
        claim_rate = 1 - X_label.mean()
        expected_claims = rankscores.loc[rankscores['y_pred'] > rankscores['y_pred'].quantile(claim_rate), :].copy()
        k_cr = expected_claims.shape[0]
        true_items = expected_claims.loc[:, 'y_true'].sum()
        pak_cr = true_items/k_cr
        rak_cr = true_items/total_items

        #Ranking results
        data = {'year': np.repeat(self.year, 3),
                'interval': np.repeat(self.interval, 3),
                'id':['Top10', 'Top20', 'Claim Rate'],
                'k':[10, 20, k_cr],
                'precision_at_k': [pak10, pak20, pak_cr],
                'recall_at_k': [rak10, rak20, rak_cr],
               }
        rank_results = pd.DataFrame(data=data)
        return (rank_results)


    #Function to perform repeated probability estimation and ranking analyis for numerical stability
    def repeated_ranking_analysis(self, clf, cv_params, n_iter, folds, runs=1):
        #Results df
        scores = pd.DataFrame(columns=['year', 'interval', 'id', 'k', 'precision_at_k', 'recall_at_k'])
        #Experiments are repeated for each test year and offset interval
        #Experiments are also repeated a user-defined number of times "runs"
        for year in self.test_years:
            self.year = year
            for interval in self.test_intervals:
                self.interval = interval
                for i in range(0, runs, 1): 
                    rank_result = self.ranking_analysis(clf, cv_params, n_iter, folds)
                    scores = pd.concat([scores if not scores.empty else None, rank_result], ignore_index=True)
        #Also return mean scores
        mean_scores = scores.groupby(['year', 'interval', 'id', 'k']).mean().reset_index()        
        return (scores.convert_dtypes(), mean_scores.convert_dtypes())
    
        