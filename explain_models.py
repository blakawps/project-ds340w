"""Model-agnostic SHAP and LIME explanations for every fitted classifier.

Explanations are computed on the transformed (scaled / one-hot-encoded)
feature matrix. The *same* held-out test cases are explained for each model.
SHAP/LIME outputs describe model predictions, not clinical causality.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer


def safe_name(text):
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')


def make_sample_indices(labels, limit=3):
    """Prefer examples from both classes for interpretable comparisons."""
    labels = np.asarray(labels)
    selected = []
    for label in [1, 0]:
        idx = np.flatnonzero(labels == label)
        if idx.size:
            selected.append(int(idx[0]))
    for i in range(len(labels)):
        if len(selected) >= limit:
            break
        if i not in selected:
            selected.append(i)
    return selected[:limit]


def explain_all(models, X_train, X_test, y_test, feature_names, output_dir, *, num_numeric):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    labels = np.asarray(y_test, dtype=int)
    names = list(map(str, feature_names))
    sample_indices = make_sample_indices(labels)
    sample_data = X_test[sample_indices]

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), size=min(20, len(X_train)), replace=False)
    background = X_train[bg_idx]

    # One-hot indicators should be treated as discrete 0/1 variables by LIME.
    encoded_columns = list(range(num_numeric, X_train.shape[1]))
    categorical_names = {i: ['Absent (0)', 'Present (1)'] for i in encoded_columns}
    lime = LimeTabularExplainer(
        training_data=X_train,
        feature_names=names,
        class_names=['No diabetes', 'Diabetes'],
        mode='classification',
        categorical_features=encoded_columns,
        categorical_names=categorical_names,
        discretize_continuous=True,
        random_state=42,
    )

    shap_records = []
    lime_records = []
    overlaps = []

    for model_name, positive_fn in models.items():
        print(f'\nGenerating SHAP + LIME: {model_name}', flush=True)
        slug = safe_name(model_name)
        model_dir = output_dir / slug
        model_dir.mkdir(exist_ok=True)

        def predict_two_classes(data, fn=positive_fn):
            p = np.asarray(fn(np.asarray(data, dtype=float)), dtype=float).reshape(-1)
            return np.column_stack([1.0 - p, p])

        # Single-output function -> SHAP attributions for diabetes probability.
        def predict_positive(data, fn=positive_fn):
            return np.asarray(fn(np.asarray(data, dtype=float)), dtype=float).ravel()

        shap_explainer = shap.KernelExplainer(
            predict_positive, background, feature_names=names
        )
        shap_values = shap_explainer.shap_values(
            sample_data, nsamples=120, silent=True
        )
        values = np.asarray(shap_values, dtype=float)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[:, :, 0]
        if values.shape != sample_data.shape:
            raise RuntimeError(f'Unexpected SHAP output shape: {values.shape}')
        shap_abs = np.mean(np.abs(values), axis=0)

        # Approximate feature importance over a small, explicitly reported
        # set of test cases; do not interpret as population-global importance.
        importance = np.argsort(shap_abs)[::-1][:12][::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.array(names)[importance], shap_abs[importance])
        ax.set_title(f'SHAP — {model_name} ({len(sample_indices)} test cases)')
        ax.set_xlabel('Mean |SHAP value| in diabetes-probability units')
        fig.tight_layout()
        fig.savefig(model_dir / 'shap_feature_importance.png', dpi=160)
        plt.close(fig)

        for local_idx, original_idx in enumerate(sample_indices):
            for j, feature in enumerate(names):
                shap_records.append({
                    'model': model_name,
                    'test_position': original_idx,
                    'actual_diabetes': int(labels[original_idx]),
                    'feature': feature,
                    'shap_value': float(values[local_idx, j]),
                    'feature_value_transformed': float(sample_data[local_idx, j]),
                })

        lime_abs = np.zeros(len(names))
        for original_idx in sample_indices:
            exp = lime.explain_instance(
                X_test[original_idx],
                predict_two_classes,
                labels=(1,),
                num_features=len(names),
                num_samples=800,
            )
            exp.save_to_file(str(model_dir / f'lime_test_case_{original_idx}.html'))
            for feat_idx, weight in exp.as_map()[1]:
                lime_abs[feat_idx] += abs(float(weight)) / len(sample_indices)
                lime_records.append({
                    'model': model_name,
                    'test_position': original_idx,
                    'actual_diabetes': int(labels[original_idx]),
                    'feature': names[feat_idx],
                    'lime_weight': float(weight),
                })

        lime_importance = np.argsort(lime_abs)[::-1][:12][::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.array(names)[lime_importance], lime_abs[lime_importance])
        ax.set_title(f'LIME — {model_name} ({len(sample_indices)} test cases)')
        ax.set_xlabel('Mean |LIME local weight| (approximate probability impact)')
        fig.tight_layout()
        fig.savefig(model_dir / 'lime_feature_importance.png', dpi=160)
        plt.close(fig)

        top_shap = set(np.argsort(shap_abs)[-5:].tolist())
        top_lime = set(np.argsort(lime_abs)[-5:].tolist())
        overlap = top_shap & top_lime
        overlaps.append({
            'model': model_name,
            'top_5_feature_overlap': len(overlap),
            'top_5_overlap_fraction': len(overlap) / 5,
            'shared_features': '; '.join(names[i] for i in sorted(overlap)),
            'number_of_cases_explained': len(sample_indices),
        })

    pd.DataFrame(shap_records).to_csv(output_dir / 'shap_values.csv', index=False)
    pd.DataFrame(lime_records).to_csv(output_dir / 'lime_weights.csv', index=False)
    pd.DataFrame(overlaps).to_csv(output_dir / 'shap_lime_comparison.csv', index=False)
    print('\nExplanations saved to', output_dir, flush=True)
    return overlaps
