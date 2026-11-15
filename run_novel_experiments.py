"""DS340W novelty experiments: five parent-paper-style models + PyTorch MLP.

Run from PyCharm or terminal:
    python run_novel_experiments.py

This deliberately NEVER reads validation_unseen.csv. The 10% unseen file is
reserved for an entirely separate, final-evaluation script after model choice.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from explain_models import explain_all, safe_name
from neural_network import train_neural_network

HERE = Path(__file__).resolve().parent
TARGET = 'diabetic'
OUTPUT = HERE / 'novel_results'
SEED = 42


def normalize_target(series):
    values = series.astype(str).str.strip().str.lower()
    mapping = {'no': 0, 'yes': 1, '0': 0, '1': 1, 'false': 0, 'true': 1}
    result = values.map(mapping)
    if result.isna().any():
        unexpected = sorted(values[result.isna()].unique())
        raise ValueError(f'Unknown {TARGET} values: {unexpected}')
    return result.astype(int)


def find_dataset():
    for candidate in [HERE / 'Diabetes_Final_Data_V2.csv',
                      HERE / 'data' / 'Diabetes_Final_Data_V2.csv']:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        'Place Diabetes_Final_Data_V2.csv beside this script or inside data/.')


def load_or_create_splits():
    split_dirs = [HERE / 'data' / 'splits', HERE / 'splits', HERE]
    for directory in split_dirs:
        train_path, test_path = directory / 'train.csv', directory / 'test.csv'
        if train_path.exists() and test_path.exists():
            if not (directory / 'validation_unseen.csv').exists():
                raise RuntimeError(
                    f'Found train/test in {directory}, but validation_unseen.csv is missing. '
                    'Use your Week 11 70/20/10 split or move the old files aside '
                    'so the script can generate new splits from the original CSV.'
                )
            print(f'Loading existing training/test files from {directory}')
            return pd.read_csv(train_path), pd.read_csv(test_path), 'existing split files'
        if train_path.exists() != test_path.exists():
            raise RuntimeError(f'Only one of train.csv/test.csv exists in {directory}')

    dataset = pd.read_csv(find_dataset())
    if TARGET not in dataset:
        raise KeyError(f'Dataset must contain the {TARGET!r} column')
    targets = normalize_target(dataset[TARGET])
    train, remaining = train_test_split(
        dataset, test_size=0.30, random_state=SEED, stratify=targets
    )
    # Remaining 30% -> 20% test + 10% NEVER-SEEN validation.
    test, unseen = train_test_split(
        remaining, test_size=1/3, random_state=SEED,
        stratify=normalize_target(remaining[TARGET])
    )
    directory = HERE / 'data' / 'splits'
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / 'train.csv', index=False)
    test.to_csv(directory / 'test.csv', index=False)
    unseen.to_csv(directory / 'validation_unseen.csv', index=False)
    print('Created 70/20/10 split, including sealed validation_unseen.csv')
    print('The validation file will NOT be opened by this program.')
    del unseen, remaining, dataset
    return train, test, 'new 70/20/10 stratified split'


def make_preprocessor(X_train):
    numeric = X_train.select_dtypes(include='number').columns.tolist()
    categorical = X_train.select_dtypes(exclude='number').columns.tolist()
    blocks = []
    if numeric:
        blocks.append(('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric))
    if categorical:
        blocks.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical))
    if not blocks:
        raise ValueError('No feature columns found')
    return ColumnTransformer(blocks), len(numeric)


def score_model(name, y_true, probs, family):
    labels = (np.asarray(probs) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    return {
        'model': name,
        'family': family,
        'accuracy': round(accuracy_score(y_true, labels), 4),
        'precision': round(precision_score(y_true, labels, zero_division=0), 4),
        'recall': round(recall_score(y_true, labels, zero_division=0), 4),
        'f1': round(f1_score(y_true, labels, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_true, probs), 4),
        'true_negative': int(tn), 'false_positive': int(fp),
        'false_negative': int(fn), 'true_positive': int(tp),
    }


def comparison_chart(table):
    labels = table['model'].to_list()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 0.2, table['recall'], width=0.4, label='Diabetes recall')
    ax.bar(x + 0.2, table['f1'], width=0.4, label='Diabetes F1')
    ax.set_xticks(x, labels, rotation=30, ha='right')
    ax.set_ylabel('Score (0–1)')
    ax.set_ylim(0, 1)
    ax.set_title('Diabetes detection: recall and F1 by model (held-out TEST set)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'model_comparison.png', dpi=160)
    plt.close(fig)


def run():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_folder = OUTPUT / 'saved_models'
    model_folder.mkdir(exist_ok=True)

    train_df, test_df, split_source = load_or_create_splits()
    y_train = normalize_target(train_df[TARGET])
    y_test = normalize_target(test_df[TARGET])
    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])
    X_test = X_test[X_train.columns]  # same column order as training

    preprocessor, num_numeric = make_preprocessor(X_train)
    X_train_prepared = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    X_test_prepared = np.asarray(preprocessor.transform(X_test), dtype=np.float32)
    feature_names = [str(s).replace('num__', '').replace('cat__', '')
                     for s in preprocessor.get_feature_names_out()]

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=SEED),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1),
    }

    rows = []
    explain_functions = {}
    nn_training_info = {}
    for name, estimator in models.items():
        print(f'Training {name} ...', flush=True)
        estimator.fit(X_train_prepared, y_train)
        probability_fn = lambda X, m=estimator: m.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
        probs = probability_fn(X_test_prepared)
        rows.append(score_model(name, y_test, probs, 'scikit-learn'))
        explain_functions[name] = probability_fn
        joblib.dump({'preprocessor': preprocessor, 'model': estimator,
                     'feature_names': feature_names},
                    model_folder / f'{safe_name(name)}.joblib')

    experiments = [
        ('PyTorch MLP', 0.00, False),
        ('PyTorch MLP Weighted', 0.00, True),
        ('PyTorch MLP Weighted + Dropout', 0.25, True),
    ]
    for name, dropout, weighted in experiments:
        print(f'Training {name} ...', flush=True)
        predictor, info, history = train_neural_network(
            X_train_prepared, y_train.to_numpy(), dropout=dropout,
            weighted=weighted, max_epochs=100, patience=12, seed=SEED
        )
        probs = predictor.positive_probability(X_test_prepared)
        rows.append(score_model(name, y_test, probs, 'PyTorch'))
        explain_functions[name] = predictor.positive_probability
        nn_training_info[name] = info
        pd.DataFrame(history).to_csv(
            OUTPUT / f'{safe_name(name)}_training_history.csv', index=False
        )
        torch.save({
            'model_state': predictor.model.cpu().state_dict(),
            'input_dim': X_train_prepared.shape[1], 'dropout': dropout,
            'feature_names': feature_names,
        }, model_folder / f'{safe_name(name)}.pt')
        predictor.device = torch.device('cpu')
        joblib.dump(preprocessor, model_folder / 'pytorch_preprocessor.joblib')

    results = pd.DataFrame(rows).sort_values('f1', ascending=False)
    results.to_csv(OUTPUT / 'model_comparison.csv', index=False)
    comparison_chart(results)
    print('\n=== TEST-SET MODEL COMPARISON ===')
    print(results.to_string(index=False))

    # Explanations use TRAIN as background, and a few examples from TEST.
    # Unseen validation is intentionally never read.
    overlaps = explain_all(
        explain_functions, X_train_prepared, X_test_prepared,
        y_test.to_numpy(), feature_names, OUTPUT / 'explanations',
        num_numeric=num_numeric
    )
    metadata = {
        'train_rows': len(train_df), 'test_rows': len(test_df),
        'input_features_raw': len(X_train.columns),
        'input_features_encoded': len(feature_names),
        'positive_cases_train': int(y_train.sum()),
        'positive_cases_test': int(y_test.sum()),
        'split_source': split_source,
        'validation_unseen_opened': False,
        'preprocessing_fitted_on': 'training split only',
        'early_stopping': 'internal 15% holdout within training split only',
        'evaluation_threshold': 0.5,
        'neural_training': nn_training_info,
        'shap_lime_compared_models': len(overlaps),
        'notes': 'SHAP and LIME explanations are approximate and are not causal or clinical evidence.',
    }
    (OUTPUT / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('\nALL DONE. Check novel_results/ for tables, plots, explanations and saved models.')
    print('Do not use validation_unseen.csv until the final frozen model evaluation.')


if __name__ == '__main__':
    run()
