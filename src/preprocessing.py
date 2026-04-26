import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

AcceptedLoan_df['term'] = AcceptedLoan_df['term'].str.extract(r'(\d+)')
AcceptedLoan_df['term'] = pd.to_numeric(AcceptedLoan_df['term'], errors='coerce')
AcceptedLoan_df.replace([' ', '', 'NA', 'N/A', 'null', 'None'], np.nan, inplace=True)

for col in AcceptedLoan_df.columns:
    series = AcceptedLoan_df[col]
    converted = pd.to_numeric(series, errors='coerce')

    if converted.notnull().sum() > 0.7 * len(series):
        AcceptedLoan_df[col] = converted


column_drop = []
size = AcceptedLoan_df.shape[0]

for col in AcceptedLoan_df.columns:
    series = AcceptedLoan_df[col]
    null_ratio = series.isnull().sum() / size * 100

    if null_ratio >= 30:
        column_drop.append(col)
        continue

    if series.notnull().sum() == 0:
        column_drop.append(col)
        continue

    if null_ratio > 0:
        if pd.api.types.is_numeric_dtype(series):
            AcceptedLoan_df[col] = series.fillna(series.median())
        else:
            AcceptedLoan_df[col] = series.fillna(series.mode().iloc[0])


print("Columns to drop:", column_drop)
AcceptedLoan_df.drop(columns=column_drop, inplace=True)
print(AcceptedLoan_df.shape)

AcceptedLoan_df.drop(columns=['id', 'url', 'title', 'emp_title', 'issue_d', 'last_pymnt_d', 'last_credit_pull_d', 'total_pymnt', 'total_pymnt_inv',
'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee','recoveries', 'collection_recovery_fee','last_pymnt_amnt', 'last_fico_range_high', 'last_fico_range_low', 
'funded_amnt_inv', 'out_prncp_inv', 'policy_code', 'pymnt_plan', 'application_type', 'initial_list_status', 'zip_code', 'addr_state', 'sub_grade', 'title',
'out_prncp', 'policy_code', 'hardship_flag', 'debt_settlement_flag'],
 inplace=True)

num_cols = AcceptedLoan_df.select_dtypes(include=[np.number]).columns
cat_cols = AcceptedLoan_df.select_dtypes(exclude=[np.number]).columns

low_variance_cols = [col for col in AcceptedLoan_df.columns 
                     if AcceptedLoan_df[col].nunique() <= 1]

AcceptedLoan_df.drop(columns=low_variance_cols, inplace=True)

cat_cols = AcceptedLoan_df.select_dtypes(exclude=[np.number]).columns

cols_to_drop = []

for col in cat_cols:
    top_freq = AcceptedLoan_df[col].value_counts(normalize=True).values[0]
    if top_freq > 0.95:
        cols_to_drop.append(col)

AcceptedLoan_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

num_cols = AcceptedLoan_df.select_dtypes(include=np.number).columns

corr_matrix = AcceptedLoan_df[num_cols].corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.9)]

AcceptedLoan_df.drop(columns=to_drop_corr, inplace=True, errors='ignore')

num_cols = AcceptedLoan_df.select_dtypes(include=np.number).columns

for col in num_cols:
    Q1 = AcceptedLoan_df[col].quantile(0.25)
    Q3 = AcceptedLoan_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    AcceptedLoan_df[col] = np.clip(AcceptedLoan_df[col], lower, upper)

leakage_keywords = ['pymnt', 'rec', 'recover', 'collection']
[col for col in AcceptedLoan_df.columns if any(k in col for k in leakage_keywords)]

leakage_cols = [col for col in AcceptedLoan_df.columns if any(k in col for k in leakage_keywords)]

AcceptedLoan_df.drop(columns=leakage_cols, inplace=True, errors='ignore')

AcceptedLoan_df['target'] = target.map(
    lambda x: 0 if x == 'Fully Paid' 
    else 1 if x in ['Charged Off', 'Default'] 
    else np.nan
)

AcceptedLoan_df.dropna(subset=['target'], inplace=True)

X = AcceptedLoan_df.drop(columns=['target'])
y = AcceptedLoan_df['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns (VERY IMPORTANT)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)