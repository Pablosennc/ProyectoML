from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings(
    "ignore",
    message="MARS no genera probabilidades nativas estrictas.*",
    category=UserWarning,
)

import joblib

from core.eda import AutomatedEDA
from core.loader import DataLoader
from core.preprocessor import AdaptivePreprocessor
from models.supervised.linear_models import RegularizedLinearModel
from models.supervised.tree_models import CARTModel
from evaluation.metrics import ModelEvaluator
from evaluation.visualizer import HDVisualizer
from models.base import BaseModel
from models.supervised.prototype_models import PrototypeModel
from models.supervised.spline_models import SplineModel
from models.unsupervised.kmeans_engine import KMeansEngine


def run_eda_stage(df: pd.DataFrame, visualizer: HDVisualizer, reports_dir: Path) -> dict:
    eda = AutomatedEDA(df)
    numeric = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    figure = None
    if len(eda._get_valid_numeric_df().columns) >= 2:
        cov, pearson, spearman = eda.calculate_matrices()
        figure = str(visualizer.plot_eda_matrices(cov, pearson, spearman))
    stats_csv = reports_dir / "eda_descriptive_summary.csv"
    stats = df.describe(include="all").T
    stats["missing_count"] = df.isna().sum()
    stats["missing_pct"] = df.isna().mean().mul(100)
    stats.to_csv(stats_csv)
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "numeric_cols": list(numeric.columns),
        "quality": eda.get_quality_diagnostics(),
        "categorical_cardinality": df.drop(columns=numeric.columns).nunique().to_dict(),
        "low_variance": numeric.columns[numeric.var().fillna(0) <= 1e-12].tolist(),
        "high_correlations": eda.detect_multicollinearity(),
        "outliers_iqr": eda.detect_outliers_iqr(),
        "eda_figure": figure,
        "eda_summary_csv": str(stats_csv),
    }


def load_dataset(data_path: str | None, demo_mode: str | None, random_state: int) -> tuple[pd.DataFrame, str | None]:
    """Carga un dataset tabular desde disco o genera un dataset sintético de demostración."""
    if demo_mode is not None:
        mode = demo_mode.lower()
        if mode == "classification":
            X_np, y_np = make_classification(
                n_samples=180,
                n_features=6,
                n_informative=4,
                n_redundant=1,
                n_classes=3,
                n_clusters_per_class=1,
                random_state=random_state,
            )
            cols = [f"feature_{i+1}" for i in range(X_np.shape[1])]
            df = pd.DataFrame(X_np, columns=cols)
            df["segmento_cliente"] = np.where(df["feature_1"] > 0, "Norte", "Sur")
            df.loc[df.index[:5], "feature_2"] = np.nan
            label_map = {0: "Bajo", 1: "Medio", 2: "Alto"}
            df["target"] = pd.Series(y_np).map(label_map)
            return df, "target"

        if mode == "regression":
            X_np, y_np = make_regression(
                n_samples=180,
                n_features=5,
                n_informative=4,
                noise=8.5,
                random_state=random_state,
            )
            cols = [f"var_{i+1}" for i in range(X_np.shape[1])]
            df = pd.DataFrame(X_np, columns=cols)
            df["zona"] = np.where(df["var_1"] >= 0, "Urbana", "Rural")
            df.loc[df.index[:4], "var_3"] = np.nan
            df["target"] = y_np
            return df, "target"

        if mode == "unsupervised":
            X_np, _ = make_blobs(
                n_samples=180,
                n_features=4,
                centers=3,
                cluster_std=0.85,
                random_state=random_state,
            )
            cols = [f"metrica_{i+1}" for i in range(X_np.shape[1])]
            df = pd.DataFrame(X_np, columns=cols)
            return df, None

        raise ValueError("--demo debe ser 'classification', 'regression' o 'unsupervised'.")

    if not data_path:
        raise ValueError("Debe proporcionar --data <ruta_archivo> o activar --demo <modo>.")

    return DataLoader(data_path).load_data(), None


def build_supervised_catalog(task_type: str, selected_models: list[str], random_state: int):
    """Registra candidatos y sus parámetros, en orden de desempate."""
    def linear(**params):
        return RegularizedLinearModel(task_type=task_type, **params)

    def tree(**params):
        return CARTModel(task_type=task_type, random_state=random_state, **params)

    def prototype(algorithm, **params):
        return PrototypeModel(algorithm=algorithm, task_type=task_type, **params)

    catalog = {
        "MARS": [({}, SplineModel(algorithm="mars", task_type=task_type))],
        "GAM": [({}, SplineModel(algorithm="gam", task_type=task_type))],
        "KNN": [
            ({"n_neighbors": k}, prototype("knn", n_neighbors=k, weights="distance"))
            for k in (3, 5, 9)
        ],
        "CART": [({"max_depth": d}, tree(max_depth=d)) for d in (3, 6, None)],
        "RANDOM_FOREST": [
            ({"max_depth": d}, tree(algorithm="random_forest", max_depth=d))
            for d in (6, None)
        ],
    }
    if task_type == "classification":
        catalog.update({
            "LOGISTIC": [({"C": c}, linear(C=c)) for c in (0.1, 1.0, 10.0)],
            "DANN": [
                ({"epsilon": e}, prototype("dann", epsilon=e)) for e in (0.5, 1.0, 2.0)
            ],
            "LVQ1": [
                ({"learning_rate": rate}, prototype("lvq1", learning_rate=rate,
                                                  epochs=20, random_state=random_state))
                for rate in (0.02, 0.05, 0.1)
            ],
        })
    else:
        catalog["OLS"] = [({}, linear(penalty=None))]
        for label, penalty in (("RIDGE", "l2"), ("LASSO", "l1")):
            catalog[label] = [
                ({"alpha": a}, linear(penalty=penalty, alpha=a)) for a in (0.1, 1.0, 10.0)
            ]

    wanted = {m.upper() for m in selected_models}
    unknown = wanted.difference(catalog, {"ALL"})
    if unknown:
        raise ValueError(f"Modelos no disponibles para {task_type}: {sorted(unknown)}")
    return {label: candidates for label, candidates in catalog.items()
            if "ALL" in wanted or label in wanted}


def run_supervised_pipeline(
    df: pd.DataFrame,
    target_col: str,
    requested_task: str,
    selected_models: list[str],
    cv_folds: int,
    test_size: float,
    random_state: int,
    visualizer: HDVisualizer,
    reports_dir: Path,
) -> dict[str, Any]:
    """Ejecuta el pipeline supervisado completo de extremo a extremo."""
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset.")

    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean.drop(columns=[target_col])
    y_raw = df_clean[target_col]

    if X.shape[1] == 0:
        raise ValueError("El dataset no contiene variables predictoras tras separar el objetivo.")

    _, inferred = DataLoader.resolve_target(df_clean, target_col)
    task_type = inferred if requested_task == "auto" else requested_task
    le = None
    class_names: list[str] | None = None

    if task_type == "classification":
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y_raw.astype(str)), index=df_clean.index)
        class_names = [str(c) for c in le.classes_]
        min_class_count = int(y_encoded.value_counts().min())
        stratify_arg = y_encoded if min_class_count >= 2 else None
    else:
        y_encoded = pd.to_numeric(y_raw, errors="raise")
        stratify_arg = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    if task_type == "classification":
        train_counts = y_train.value_counts()
        if len(train_counts) != len(class_names) or train_counts.min() < 2:
            raise ValueError("Se requieren al menos dos observaciones por clase en entrenamiento para validar.")
    elif len(y_test) < 2 or len(y_train) < 4:
        raise ValueError("La regresión requiere al menos cuatro filas de entrenamiento y dos de prueba.")

    catalog = build_supervised_catalog(task_type, selected_models, random_state)
    models_dir = reports_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    cls_eval_results: dict[str, dict] = {}
    generated_figures: list[str] = []
    saved_models: dict[str, str] = {}

    search_rows = []
    feature_counts = {}
    selected_parameters = {}
    for model_label, candidates in catalog.items():
        scale = model_label not in {"CART", "RANDOM_FOREST"}
        metric = "f1_weighted" if task_type == "classification" else "rmse"
        ranked = []
        for params, candidate in candidates:
            result = ModelEvaluator.cross_validate(
                candidate, X_train, y_train, task_type, cv_folds, random_state, scale=scale
            )
            score = result["summary"][metric]["mean"]
            ranked.append((score, candidate, result, params))
            search_rows.append({"model": model_label, **params, "metric": metric, "cv_mean": score})
        _, model_template, cv_res, best_params = (max if task_type == "classification" else min)(
            ranked, key=lambda entry: entry[0]
        )
        selected_parameters[model_label] = best_params
        lc_res = ModelEvaluator.compute_learning_curve(
            model_template=model_template,
            X=X_train,
            y=y_train,
            task_type=task_type,
            k_folds=cv_folds,
            random_state=random_state,
            scale=scale,
        )
        lc_fig = visualizer.plot_learning_curve(lc_res, model_name=model_label)
        generated_figures.append(str(lc_fig))

        # 2. Ajuste final en X_train_proc y evaluación en X_test_proc
        preprocessor = AdaptivePreprocessor()
        X_train_proc = preprocessor.fit_transform_train(X_train, scale=scale)
        X_test_proc = preprocessor.transform_test(X_test)
        feature_counts[model_label] = X_train_proc.shape[1]
        fitted_model = model_template.fit(X_train_proc, y_train)
        y_pred = fitted_model.predict(X_test_proc)

        model_path = fitted_model.save(models_dir / f"{model_label.lower()}.pkl")
        _ = BaseModel.load(model_path)
        saved_models[model_label] = str(model_path)
        joblib.dump({"preprocessor": preprocessor, "label_encoder": le},
                    model_path.with_suffix(".preprocessing.pkl"))
        if isinstance(fitted_model, CARTModel):
            fitted_model.get_feature_importance().to_csv(
                reports_dir / f"{model_label.lower()}_feature_importance.csv", index=False
            )

        if task_type == "classification":
            y_proba = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_proba = fitted_model.predict_proba(X_test_proc)
            except NotImplementedError:
                y_proba = None

            eval_res = ModelEvaluator.evaluate_classification(
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                class_names=class_names,
            )
            cls_eval_results[model_label] = eval_res
            surface = visualizer.plot_decision_slice(
                fitted_model, preprocessor, X_train, y_train, model_label, class_names
            )
            if surface is not None:
                generated_figures.append(str(surface))
            cm_fig = visualizer.plot_confusion_matrix(eval_res, model_name=model_label)
            generated_figures.append(str(cm_fig))

            row = {
                "model": model_label,
                "test_accuracy": eval_res["metrics"]["accuracy"],
                "test_f1_weighted": eval_res["metrics"]["f1_weighted"],
                "test_f1_macro": eval_res["metrics"]["f1_macro"],
                "roc_auc_macro": eval_res["metrics"].get("roc_auc_macro", np.nan),
                "cv_f1_weighted": cv_res["summary"]["f1_weighted"]["mean"],
                "cv_accuracy_mean": cv_res["summary"]["accuracy"]["mean"],
                "cv_accuracy_std": cv_res["summary"]["accuracy"]["std"],
            }
            comparison_rows.append(row)
        else:
            eval_res = ModelEvaluator.evaluate_regression(y_true=y_test, y_pred=y_pred)
            reg_fig = visualizer.plot_regression_diagnostics(eval_res, model_name=model_label)
            generated_figures.append(str(reg_fig))

            row = {
                "model": model_label,
                "test_r2": eval_res["metrics"]["r2"],
                "test_rmse": eval_res["metrics"]["rmse"],
                "test_mae": eval_res["metrics"]["mae"],
                "cv_r2_mean": cv_res["summary"]["r2"]["mean"],
                "cv_rmse_mean": cv_res["summary"]["rmse"]["mean"],
            }
            comparison_rows.append(row)

    if task_type == "classification" and cls_eval_results:
        roc_pr_fig = visualizer.plot_roc_and_pr_comparison(cls_eval_results)
        if roc_pr_fig is not None:
            generated_figures.append(str(roc_pr_fig))

    df_comp = pd.DataFrame(comparison_rows)
    pd.DataFrame(search_rows).to_csv(reports_dir / "hyperparameter_search.csv", index=False)
    sort_col = "cv_f1_weighted" if task_type == "classification" else "cv_rmse_mean"
    df_comp = df_comp.sort_values(by=sort_col, ascending=task_type == "regression", kind="stable").reset_index(drop=True)

    comp_csv = reports_dir / "supervised_model_comparison.csv"
    df_comp.to_csv(comp_csv, index=False)

    best_model_name = str(df_comp.loc[0, "model"])

    return {
        "mode": "supervised",
        "task_type": task_type,
        "target_column": target_col,
        "train_samples": int(len(X_train_proc)),
        "test_samples": int(len(X_test_proc)),
        "n_features_processed": int(feature_counts[best_model_name]),
        "best_model": best_model_name,
        "comparison_table": df_comp.astype(object).where(df_comp.notna(), None).to_dict(orient="records"),
        "selected_parameters": selected_parameters,
        "hyperparameter_csv": str(reports_dir / "hyperparameter_search.csv"),
        "comparison_csv": str(comp_csv),
        "saved_models": saved_models,
        "figures": generated_figures,
    }


def run_unsupervised_pipeline(
    df: pd.DataFrame,
    k_range: tuple[int, int],
    k_criterion: str,
    k_init: str,
    random_state: int,
    visualizer: HDVisualizer,
    reports_dir: Path,
) -> dict[str, Any]:
    """Ejecuta el pipeline no supervisado K-Means con auditoría multi-criterio y PCA 2D."""
    preprocessor = AdaptivePreprocessor()
    X_proc = preprocessor.fit_transform_train(df)

    engine = KMeansEngine(
        k_range=k_range,
        selection_criterion=k_criterion,
        init=k_init,
        n_init=10,
        random_state=random_state,
    ).fit(X_proc)

    audit_report = engine.get_audit_report()
    km_figs = visualizer.plot_kmeans_audit(audit_report)

    models_dir = reports_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = engine.save(models_dir / "kmeans_engine.pkl")
    _ = BaseModel.load(model_path)

    joblib.dump({"preprocessor": preprocessor}, model_path.with_suffix(".preprocessing.pkl"))
    metrics_csv = reports_dir / "kmeans_k_sweep_metrics.csv"
    audit_report["metrics_table"].to_csv(metrics_csv, index=False)

    df_segmented = df.copy()
    df_segmented["cluster_id"] = audit_report["labels"]
    df_segmented["silhouette_sample"] = audit_report["sample_silhouettes"]
    seg_csv = reports_dir / "kmeans_segmented_dataset.csv"
    df_segmented.to_csv(seg_csv, index=False)

    return {
        "mode": "unsupervised",
        "algorithm": "K-Means",
        "init": k_init,
        "optimal_k": int(audit_report["optimal_k"]),
        "selection_criterion": audit_report["selection_criterion"],
        "avg_silhouette": float(audit_report["avg_silhouette"]),
        "pca_explained_variance": [float(v) for v in audit_report["pca_explained_variance"]],
        "metrics_csv": str(metrics_csv),
        "segmented_dataset_csv": str(seg_csv),
        "saved_model": str(model_path),
        "figures": {k: str(v) for k, v in km_figs.items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquestador Principal del Framework Modular de Aprendizaje Automático."
    )
    parser.add_argument("--data", type=str, default=None, help="Ruta al archivo CSV, XLSX o Parquet.")
    parser.add_argument(
        "--demo",
        type=str,
        choices=["classification", "regression", "unsupervised"],
        default=None,
        help="Ejecuta una demostración end-to-end con datos sintéticos.",
    )
    parser.add_argument("--target", type=str, default=None, help="Nombre de la columna objetivo.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "supervised", "unsupervised"],
        default="auto",
        help="Paradigma de aprendizaje: supervised, unsupervised o auto.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["auto", "classification", "regression"],
        default="auto",
        help="Tipo de tarea supervisada.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Modelos supervisados a ejecutar (ej. all, mars gam knn dann lvq1).",
    )
    parser.add_argument(
        "--k-range",
        nargs=2,
        type=int,
        default=[2, 8],
        metavar=("K_MIN", "K_MAX"),
        help="Rango de clústeres K a auditar en K-Means.",
    )
    parser.add_argument(
        "--k-criterion",
        type=str,
        choices=["elbow", "silhouette"],
        default="elbow",
        help="Criterio automático de selección de K óptimo.",
    )
    parser.add_argument("--k-init", choices=["random", "k-means++"], default="k-means++")
    parser.add_argument("--cv-folds", type=int, default=5, help="Número de folds para validación cruzada.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción de test (0 < test_size < 1).")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directorio raíz de salida.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla aleatoria global.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    warnings.filterwarnings(
        "ignore",
        message="MARS no genera probabilidades nativas estrictas.*",
        category=UserWarning,
    )
    args = parse_args(argv)
    reports_dir = Path(args.output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"

    visualizer = HDVisualizer(output_dir=figures_dir, dpi=300)

    df, demo_target = load_dataset(args.data, args.demo, args.random_state)
    target_col = args.target if args.target is not None else demo_target

    if args.mode != "unsupervised":
        target_col, _ = DataLoader.resolve_target(df, target_col)

    if args.mode == "auto":
        if target_col is None:
            warnings.warn("No se detectó una columna objetivo inequívoca. Se ejecutará K-Means; use --target para indicar el objetivo.")
        resolved_mode = "supervised" if target_col is not None else "unsupervised"
    else:
        resolved_mode = args.mode

    if args.cv_folds < 2:
        raise ValueError("--cv-folds debe ser al menos 2.")

    eda_summary = run_eda_stage(df, visualizer=visualizer, reports_dir=reports_dir)

    if resolved_mode == "supervised":
        if target_col is None:
            raise ValueError("El modo supervisado requiere especificar --target <columna>.")
        pipeline_summary = run_supervised_pipeline(
            df=df,
            target_col=target_col,
            requested_task=args.task,
            selected_models=args.models,
            cv_folds=args.cv_folds,
            test_size=args.test_size,
            random_state=args.random_state,
            visualizer=visualizer,
            reports_dir=reports_dir,
        )
    else:
        X_unsup = df.drop(columns=[target_col]) if (target_col and target_col in df.columns) else df
        pipeline_summary = run_unsupervised_pipeline(
            df=X_unsup,
            k_range=(args.k_range[0], args.k_range[1]),
            k_criterion=args.k_criterion,
            k_init=args.k_init,
            random_state=args.random_state,
            visualizer=visualizer,
            reports_dir=reports_dir,
        )

    final_report = {
        "eda": eda_summary,
        "pipeline": pipeline_summary,
    }

    report_json_path = reports_dir / "execution_report.json"
    report_json_path.write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    print("=" * 72)
    print(f"EJECUCIÓN COMPLETADA | MODO: {resolved_mode.upper()}")
    print("=" * 72)
    print(f"Dataset analizado : {eda_summary['n_rows']} filas x {eda_summary['n_cols']} columnas")
    print(f"Figura EDA (300 DPI): {eda_summary['eda_figure']}")

    if resolved_mode == "supervised":
        print(f"Tarea             : {pipeline_summary['task_type']}")
        print(f"Mejor modelo      : {pipeline_summary['best_model']}")
        print(f"Tabla comparativa : {pipeline_summary['comparison_csv']}")
    else:
        print(f"K óptimo (K*)     : {pipeline_summary['optimal_k']} ({pipeline_summary['selection_criterion']})")
        print(f"Silueta promedio  : {pipeline_summary['avg_silhouette']:.4f}")
        print(f"Dataset segmentado: {pipeline_summary['segmented_dataset_csv']}")

    print(f"Reporte JSON      : {report_json_path}")
    print("=" * 72)

    return final_report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError) as exc:
        raise SystemExit(f"Error: {exc}") from None