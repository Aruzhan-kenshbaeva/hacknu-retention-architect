import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

train_users = pd.read_csv('train_users.csv')
train_prop = pd.read_csv('train_users_properties.csv')
train_quiz = pd.read_csv('train_users_quizzes.csv', low_memory=False)
train_purch = pd.read_csv('train_users_purchases.csv')
train_trans = pd.read_csv('train_users_transaction_attempts.csv', low_memory=False)

test_users = pd.read_csv('test_users.csv')
test_prop = pd.read_csv('test_users_properties.csv')
test_quiz = pd.read_csv('test_users_quizzes.csv', low_memory=False)
test_purch = pd.read_csv('test_users_purchases.csv')
test_trans = pd.read_csv('test_users_transaction_attempts.csv', low_memory=False)

def make_features(users, prop, quiz, purch, trans):
    df = users[['user_id']].copy()
    
    prop_clean = prop[['user_id', 'subscription_plan', 'country_code']].copy()
    df = df.merge(prop_clean, on='user_id', how='left')
    
    quiz_clean = quiz[['user_id', 'source', 'role', 'experience', 
                        'frustration', 'usage_plan']].copy()
    df = df.merge(quiz_clean, on='user_id', how='left')
    
    purch_agg = purch.groupby('user_id').agg(
        num_purchases=('transaction_id', 'count'),
        total_spent=('purchase_amount_dollars', 'sum')
    ).reset_index()
    df = df.merge(purch_agg, on='user_id', how='left')
    df['num_purchases'] = df['num_purchases'].fillna(0)
    df['total_spent'] = df['total_spent'].fillna(0)
    
    failed = trans[trans['failure_code'].notna()]
    if 'user_id' in trans.columns:
        failed_agg = failed.groupby('user_id').agg(
            num_failed_payments=('transaction_id', 'count')
        ).reset_index()
        df = df.merge(failed_agg, on='user_id', how='left')
    else:
        df['num_failed_payments'] = 0
    df['num_failed_payments'] = df['num_failed_payments'].fillna(0)
    
    return df

train_df = make_features(train_users, train_prop, train_quiz, train_purch, train_trans)
test_df = make_features(test_users, test_prop, test_quiz, test_purch, test_trans)

train_df = train_df.merge(train_users[['user_id', 'churn_status']], on='user_id', how='left')
train_df = train_df.dropna(subset=['churn_status'])

cat_cols = ['subscription_plan', 'country_code', 'source', 
            'role', 'experience', 'frustration', 'usage_plan']

for col in cat_cols:
    train_df[col] = train_df[col].fillna('unknown')
    test_df[col] = test_df[col].fillna('unknown')
    train_df[col] = train_df[col].astype('category').cat.codes
    test_df[col] = test_df[col].astype('category').cat.codes

feature_cols = ['subscription_plan', 'country_code', 'source', 'role', 
                'experience', 'frustration', 'usage_plan',
                'num_purchases', 'total_spent', 'num_failed_payments']

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['churn_status']
X_test = test_df[feature_cols].fillna(0)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred, average='weighted')
print(f"F1: {f1:.4f}")
print(classification_report(y_train, train_pred))

test_pred = model.predict(X_test)

result = pd.DataFrame({
    'user_id': test_users['user_id'],
    'churn_status': test_pred
})
result.to_csv('predictions.csv', index=False)
print("Готово! predictions.csv сохранён")
print(result['churn_status'].value_counts())