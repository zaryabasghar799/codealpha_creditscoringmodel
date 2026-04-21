# ============================================================
# TASK 1: Credit Scoring Model — CodeAlpha ML Internship
# ============================================================
# Install: pip install pandas numpy scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'age':             np.random.randint(21, 65, n),
    'income':          np.random.randint(20000, 120000, n),
    'loan_amount':     np.random.randint(1000, 50000, n),
    'loan_tenure':     np.random.randint(6, 60, n),
    'num_credit_lines':np.random.randint(1, 10, n),
    'payment_history': np.random.choice(['Excellent','Good','Fair','Poor'], n,
                                        p=[0.3, 0.4, 0.2, 0.1]),
    'debt_to_income':  np.round(np.random.uniform(0.05, 0.65, n), 2),
    'employment_years':np.random.randint(0, 30, n),
    'num_late_payments':np.random.randint(0, 10, n),
    'existing_loans':  np.random.randint(0, 5, n),
})

# Target: creditworthy (1) or not (0)
score = (
    (data['income'] / 120000) * 30 +
    (data['employment_years'] / 30) * 20 +
    (1 - data['debt_to_income']) * 25 +
    (1 - data['num_late_payments'] / 10) * 15 +
    (data['payment_history'].map({'Excellent':10,'Good':7,'Fair':4,'Poor':1})) +
    np.random.normal(0, 5, n)
)
data['creditworthy'] = (score > score.median()).astype(int)

print("=" * 55)
print("   TASK 1: CREDIT SCORING MODEL — CodeAlpha")
print("=" * 55)
print(f"\n📊 Dataset Shape : {data.shape}")
print(f"✅ Creditworthy  : {data['creditworthy'].sum()} ({data['creditworthy'].mean()*100:.1f}%)")
print(f"❌ Not Worthy    : {(data['creditworthy']==0).sum()} ({(1-data['creditworthy'].mean())*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
le = LabelEncoder()
data['payment_history_enc'] = le.fit_transform(data['payment_history'])

features = ['age','income','loan_amount','loan_tenure','num_credit_lines',
            'payment_history_enc','debt_to_income','employment_years',
            'num_late_payments','existing_loans']
X = data[features]
y = data['creditworthy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "─" * 55)
print("  MODEL PERFORMANCE COMPARISON")
print("─" * 55)
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("─" * 55)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_proba),
    }
    r = results[name]
    print(f"{name:<22} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
          f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['roc_auc']:>6.3f}")

# Best model
best = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n🏆 Best Model: {best} (AUC = {results[best]['roc_auc']:.3f})")
print("\n" + classification_report(y_test, results[best]['y_pred'],
      target_names=['Not Creditworthy','Creditworthy']))

# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Task 1: Credit Scoring Model — CodeAlpha', fontsize=15, fontweight='bold')

# (a) Confusion Matrix
cm = confusion_matrix(y_test, results[best]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=['Not Creditworthy','Creditworthy'],
            yticklabels=['Not Creditworthy','Creditworthy'])
axes[0,0].set_title(f'Confusion Matrix — {best}')
axes[0,0].set_ylabel('Actual'); axes[0,0].set_xlabel('Predicted')

# (b) ROC Curves
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    axes[0,1].plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})", lw=2)
axes[0,1].plot([0,1],[0,1],'k--', lw=1)
axes[0,1].set_title('ROC Curves — All Models')
axes[0,1].set_xlabel('False Positive Rate'); axes[0,1].set_ylabel('True Positive Rate')
axes[0,1].legend(fontsize=9)

# (c) Feature Importance (Random Forest)
rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
fi.plot(kind='barh', ax=axes[1,0], color='steelblue')
axes[1,0].set_title('Feature Importance — Random Forest')
axes[1,0].set_xlabel('Importance Score')

# (d) Metrics Bar Chart
metrics_df = pd.DataFrame({n: {k: v for k,v in r.items()
             if k in ['accuracy','precision','recall','f1','roc_auc']}
             for n, r in results.items()}).T
metrics_df.plot(kind='bar', ax=axes[1,1], rot=20, colormap='Set2')
axes[1,1].set_title('Metrics Comparison — All Models')
axes[1,1].set_ylim(0.5, 1.0)
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/task1_credit_scoring_output.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Plot saved: task1_credit_scoring_output.png")
