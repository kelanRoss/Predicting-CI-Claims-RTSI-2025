#Code to create data partitions based on a requested year of data.
#As part of the model evaluation, date and time fields are offset to simulate early predictions.

#Setup - libraries
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

#First function creates train test partition for coverage information and client demographics only
def create_train_test(data, model_cols, year, interval):
        cut_dt = f"{year}-01-01" #Sets date which splits the partitions
        df = data.sort_values(by='END_DT').copy() #Sort by ending date of coverage
        train_df = df.query(f"END_DT < '{cut_dt}'") #All coverages ending before the partition date are training data
        X_feat = train_df.loc[:, model_cols] #Train feature set
        X_label = train_df.loc[:, 'CLAIM'] #Train target variable
    
        test_df = df.query(f"END_DT >= '{cut_dt}'").copy() #All coverages ending after the partition date are test data
        #Check if there are any rows in the test set
        if test_df.shape[0] > 0:
            #The end date of the coverage is offset by the interval value.
            #If this causes the end date to be before the start date, take the start date.
            test_df.loc[:, 'ADJ_END_DT'] = test_df.apply(lambda x: max(x['START_DT'], (x['END_DT'] - pd.tseries.offsets.Day(interval))), axis=1)
            #Recalculate coverage tenure, time to mature and client age based on the adjusted end date
            test_df.loc[:, 'TENURE'] = (test_df['ADJ_END_DT'] - test_df['START_DT']).dt.days
            test_df.loc[:, 'TIME_TO_MATURE'] = (test_df['MATURITY_DT'] - test_df['ADJ_END_DT']).dt.days
            test_df.loc[:, 'LAST_AGE'] = test_df.apply(lambda x: relativedelta(x['ADJ_END_DT'], x['START_DT']).years + x['START_AGE'], axis=1)
            
        Y_feat = test_df.loc[:, model_cols] #Test feature set
        Y_label = test_df.loc[:, 'CLAIM'] #Test target variable
        Y_id = test_df.loc[:, 'CVG_ID'] #Keep client IDs for results delivery

        #Label categorical columns so that they pass through the standardisation step
        cat_cols = data.columns[np.where(data.nunique() == 2)[0].tolist()]
        labelled_cols = {col: 'CATEGORY_' + col if col in cat_cols else 'SCALAR_' + col for col in model_cols}
        X_feat.rename(columns=labelled_cols, inplace=True)
        Y_feat.rename(columns=labelled_cols, inplace=True)
    
        return(X_feat, X_label, Y_feat, Y_label, Y_id)


###################################################################################################################
#Second function integrates client health claims and creates data partitions based on desired year
#Health claims are made on health insurance coverage and are linked through the CRM system.
#Several aggregate features are created from the health claims.
def create_claim_train_test(data, claim_data, model_cols, year, interval):
        merge_on = 'CVG_ID'
        #Health claims made after closure of the client's critical illness coverage are not applicable and must be removed.
        #If the end date of the CI coverage was offset for analysis, claims after this date must be removed
        claims = claim_data.copy()
        #The end date of the coverage is offset by the interval value.
        #If this causes the end date to be before the start date, take the start date.
        claims.loc[:, 'ADJ_END_DT'] = claims.apply(lambda x: max(x['START_DT'], (x['END_DT'] - pd.tseries.offsets.Day(interval))), axis=1)
        #Claim date must be before end date of critical illness coverage
        claims = claims.query(f"CLAIM_DT <= ADJ_END_DT").copy()
    
        #Aggregate claims information
        #Procedures
        #For every critical illness coverage record, get a count of distinct procedures claimed for by the client, per procedure group
        claim_procedures = claims.groupby([merge_on, 'PROCEDURE_GROUP'])\
                                 .agg(NUM_PROCEDURE = ('PROCEDURE_CODE', 'nunique'))\
                                 .unstack(fill_value=0)\
                                 .reset_index()
        claim_procedures.columns = [ "_".join(col).strip('_') for col in claim_procedures.columns]
    
        #Diagnoses
        #Similarly, get a count of distinct diagnoses claimed for by the client, per diagnosis group
        claim_diagnosis = claims.groupby([merge_on, 'DIAGNOSIS_GROUP'])\
                                .agg(NUM_DIAGNOSIS = ('DIAGNOSIS_CODE', 'nunique'))\
                                .unstack(fill_value=0)\
                                .reset_index()
        claim_diagnosis.columns = [ "_".join(col).strip('_') for col in claim_diagnosis.columns]
    
        #Durations
        #Here we want aggregate time information on the claim
        #A claim can have multiple "lines" which we want to reduce to one.
        #These properties should be the same for each line, but we take the max value to be safe.
        claim_durations = claims.groupby([merge_on, 'CLAIM_ID'])\
                                .agg(CLAIM_DURATION = ('CLAIM_DURATION', 'max'),
                                     DAYS_FROM_PREVIOUS = ('DAYS_FROM_PREVIOUS', 'max'),
                                     CLAIM_DT = ('CLAIM_DT', 'max'))\
                                .reset_index()
        #Now for each client, we want their average claim duration (procedure length), max claim duration (longest procedure),
        #and the frequency at which they made claims
        claim_durations_agg = claim_durations.groupby(merge_on)\
                                             .agg(AVG_CLAIM_DURATION = ('CLAIM_DURATION', 'mean'),
                                                  MAX_CLAIM_DURATION = ('CLAIM_DURATION', 'max'),
                                                  CLAIM_FREQUENCY = ('DAYS_FROM_PREVIOUS', 'mean'))\
                                             .reset_index()
    
        #General info
        #For each client, get the distinct count of claims made, the total count of distinct procedures and diagnoses received,
        #and the total and average charges across all claims.
        claims_agg = claims.groupby(merge_on)\
                           .agg(NUM_CLAIMS = ('CLAIM_ID', 'nunique'),
                                NUM_PROCEDURE = ('PROCEDURE_CODE', 'nunique'),
                                NUM_DIAGNOSIS = ('DIAGNOSIS_CODE','nunique'),
                                TOTAL_CHARGES = ('CHARGES', 'sum'),
                                AVG_CHARGE = ('CHARGES', 'mean'))\
                           .reset_index()

        #Now merge the aggregate health claim features to the critical illness coverage dataset
        df_mod = data.copy()
        df_mod = df_mod.merge(claims_agg, how='inner', on=merge_on, validate='one_to_one')
        df_mod = df_mod.merge(claim_procedures, how='inner', on=merge_on, validate='one_to_one')
        df_mod = df_mod.merge(claim_diagnosis, how='inner', on=merge_on, validate='one_to_one')
        df_mod = df_mod.merge(claim_durations_agg, how='inner', on=merge_on, validate='one_to_one')
        #Fill any missing values with 0
        df_mod.fillna(0, inplace=True)

        #Ensure all counts and maximum values are integers
        for col in df_mod.columns:
            if any(n in col for n in ['NUM', 'MAX']):
                df_mod[col] = df_mod[col].astype('int')
    
        #Now create the data partitions as normal
        X_feat, X_label, Y_feat, Y_label, Y_id = create_train_test(df_mod, model_cols, year, interval)
        return(X_feat, X_label, Y_feat, Y_label, Y_id)