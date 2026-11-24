from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings(
    "ignore",
    message="MARS no genera probabilidades nativas estrictas.*",
    category=UserWarning,
)

from evaluation.metrics import ModelEvaluator
from evaluation.visualizer import HDVisualizer
from models.base import BaseModel
from models.supervised.prototype_models import PrototypeModel
from models.supervised.spline_models import SplineModel
from models.unsupervised.kmeans_engine import KMeansEngine


def _discover_supervised_classes() -> dict[str, type[BaseModel]]:
    """
    Descubre dinámicamente todas las subclases concretas de BaseModel
    disponibles dentro del paquete `models.supervised`.
    """
    discovered: dict[str, type[BaseModel]] = {
        "SplineModel": SplineModel,
        "PrototypeModel": PrototypeModel,
    }
    try:
        pkg = importlib.import_module("models.supervised")
        for mod_info in pkgutil.iter_modules(pkg.__path__):
            mod_name = f"models.supervised.{mod_info.name}"
            module = importlib.import_module(mod_name)
            for attr_name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseModel)
                    and obj is not BaseModel
                    and not inspect.isabstract(obj)
                ):
                    discovered[attr_name] = obj
    except Exception as exc:
        warnings.warn(f"Descubrimiento parcial en models.supervised: {exc}")
    return discovered


def _try_run_project_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    """
    Intenta delegar el análisis exploratorio a `AutomatedEDA` si existe en el proyecto;
    retorna None si no está presente para usar el motor estadístico integrado.
    """
    for candidate_mod in ("eda.automated_eda", "eda.analyzer", "data.eda", "eda"):
        try:
            mod = importlib.import_module(candidate_mod)
            eda_cls = getattr(mod, "AutomatedEDA", None)
            if eda_cls is not None:
                eda_inst = eda_cls(df) if "df" in inspect.signature(eda_cls).parameters else eda_cls()
                for method_name in ("run", "analyze", "compute_matrices", "generate_report"):
                    fn = getattr(eda_inst, method_name, None)
                    if callable(fn):
                        sig = inspect.signature(fn)
                        res = fn(df) if len(sig.parameters) >= 1 else fn()
                        if isinstance(res, dict) and all(
                            k in res for k in ("covariance", "pearson", "spearman")
                        ):
                            return res
                if all(
                    hasattr(eda_inst, attr)
                    for attr in ("cov_matrix_", "pearson_matrix_", "spearman_matrix_")
                ):
                    return {
                        "covariance": eda_inst.cov_matrix_,
                        "pearson": eda_inst.pearson_matrix_,
                        "spearman": eda_inst.spearman_matrix_,
                    }
        except Exception:
            continue
    return None


def run_eda_stage(
    df: pd.DataFrame, visualizer: HDVisualizer, reports_dir: Path
) -> dict[str, Any]:
    """Ejecuta auditoría EDA multivariada y exporta matrices en 300 DPI."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        raise ValueError(
            "Se requieren al menos 2 variables numéricas para generar matrices de covarianza y correlación."
        )

    external_res = _try_run_project_eda(df)
    if external_res is not None:
        cov_mat = external_res["covariance"]
        pearson_mat = external_res["pearson"]
        spearman_mat = external_res["spearman"]
    else:
        cov_mat = num_df.cov()
        pearson_mat = num_df.corr(method="pearson")
        spearman_mat = num_df.corr(method="spearman")

    eda_fig_path = visualizer.plot_eda_matrices(
        cov_matrix=cov_mat,
        pearson_matrix=pearson_mat,
        spearman_matrix=spearman_mat,
        filename="eda_correlation_covariance_hd.png",
    )

    summary_stats = num_df.describe().T
    summary_stats["missing_count"] = num_df.isna().sum()
    summary_stats["missing_pct"] = (num_df.isna().mean() * 100.0).round(2)
    stats_csv = reports_dir / "eda_descriptive_summary.csv"
    summary_stats.to_csv(stats_csv)

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "numeric_cols": list(num_df.columns),
        "eda_figure": str(eda_fig_path),
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

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, sep=None, engine="python")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Formato de archivo no soportado: '{suffix}'. Use CSV, XLSX o Parquet.")

    return df, None


def infer_task_type(y: pd.Series, requested_task: str) -> str:
    """Infiere automáticamente si el problema supervisado es clasificación o regresión."""
    if requested_task != "auto":
        return requested_task.lower()
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_object_dtype(y) or isinstance(
        y.dtype, pd.CategoricalDtype
    ):
        return "classification"
    n_unique = int(y.nunique(dropna=True))
    if pd.api.types.is_integer_dtype(y) and n_unique <= max(15, int(0.05 * len(y))):
        return "classification"
    return "regression"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construye un preprocesador reproducible sin fuga de datos:
    - Numéricas: imputación por mediana + estandarización Z-score.
    - Categóricas: imputación por moda + One-Hot Encoding denso.
    """
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _transform_to_dataframe(
    preprocessor: ColumnTransformer, X_tr: pd.DataFrame, X_te: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Ajusta el ColumnTransformer en train y devuelve DataFrames con nombres de columnas limpios."""
    X_tr_arr = preprocessor.fit_transform(X_tr)
    try:
        feat_names = [str(c) for c in preprocessor.get_feature_names_out()]
    except Exception:
        feat_names = [f"feat_{i}" for i in range(X_tr_arr.shape[1])]

    df_tr = pd.DataFrame(X_tr_arr, columns=feat_names, index=X_tr.index)
    if X_te is None:
        return df_tr, None

    X_te_arr = preprocessor.transform(X_te)
    df_te = pd.DataFrame(X_te_arr, columns=feat_names, index=X_te.index)
    return df_tr, df_te


def build_supervised_catalog(
    task_type: str, selected_models: list[str], random_state: int
) -> dict[str, BaseModel]:
    """
    Construye el catálogo de modelos supervisados compatibles con `task_type`,
    integrando SplineModel, PrototypeModel y cualquier otra subclase de BaseModel
    presente en `models.supervised`.
    """
    discovered = _discover_supervised_classes()
    catalog: dict[str, BaseModel] = {}

    # 1. SplineModel (MARS y GAM)
    catalog["MARS"] = SplineModel(algorithm="mars", task_type=task_type)
    catalog["GAM"] = SplineModel(algorithm="gam", task_type=task_type)

    # 2. PrototypeModel (k-NN en regresión/clasificación; DANN y LVQ1 en clasificación)
    catalog["KNN"] = PrototypeModel(algorithm="knn", task_type=task_type, n_neighbors=5)
    if task_type == "classification":
        catalog["DANN"] = PrototypeModel(
            algorithm="dann", task_type="classification", n_neighbors=5, epsilon=1.0
        )
        catalog["LVQ1"] = PrototypeModel(
            algorithm="lvq1",
            task_type="classification",
            learning_rate=0.05,
            epochs=20,
            random_state=random_state,
        )

    # 3. Otras familias descubiertas en models.supervised (p. ej. LinearModel, TreeModel)
    for cls_name, cls_type in discovered.items():
        if cls_type in {SplineModel, PrototypeModel}:
            continue
        candidate_algos = (
            ["logistic", "ridge_classifier", "cart", "random_forest", "gradient_boosting"]
            if task_type == "classification"
            else ["ols", "ridge", "lasso", "cart", "random_forest", "gradient_boosting"]
        )
        for algo in candidate_algos:
            try:
                inst = cls_type(algorithm=algo, task_type=task_type, random_state=random_state)
                catalog[algo.upper()] = inst
            except Exception:
                continue

    if "all" not in [m.lower() for m in selected_models]:
        wanted = {m.upper() for m in selected_models}
        catalog = {k: v for k, v in catalog.items() if k.upper() in wanted}

    if not catalog:
        raise ValueError(
            f"Ningún modelo coincidió con --models {selected_models} para task_type='{task_type}'."
        )

    return catalog


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

    task_type = infer_task_type(y_raw, requested_task)
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

    preprocessor = build_preprocessor(X_train)
    X_train_proc, X_test_proc = _transform_to_dataframe(preprocessor, X_train, X_test)

    catalog = build_supervised_catalog(task_type, selected_models, random_state)
    models_dir = reports_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    cls_eval_results: dict[str, dict] = {}
    generated_figures: list[str] = []
    saved_models: dict[str, str] = {}

    for model_label, model_template in catalog.items():
        # 1. Validación cruzada y curva de aprendizaje sobre train sin fuga
        cv_res = ModelEvaluator.cross_validate(
            model_template=model_template,
            X=X_train_proc,
            y=y_train,
            task_type=task_type,
            k_folds=cv_folds,
            random_state=random_state,
        )
        lc_res = ModelEvaluator.compute_learning_curve(
            model_template=model_template,
            X=X_train_proc,
            y=y_train,
            task_type=task_type,
            k_folds=cv_folds,
            random_state=random_state,
        )
        lc_fig = visualizer.plot_learning_curve(lc_res, model_name=model_label)
        generated_figures.append(str(lc_fig))

        # 2. Ajuste final en X_train_proc y evaluación en X_test_proc
        fitted_model = model_template.fit(X_train_proc, y_train)
        y_pred = fitted_model.predict(X_test_proc)

        model_path = fitted_model.save(models_dir / f"{model_label.lower()}.pkl")
        _ = BaseModel.load(model_path)
        saved_models[model_label] = str(model_path)

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
            cm_fig = visualizer.plot_confusion_matrix(eval_res, model_name=model_label)
            generated_figures.append(str(cm_fig))

            row = {
                "model": model_label,
                "test_accuracy": eval_res["metrics"]["accuracy"],
                "test_f1_weighted": eval_res["metrics"]["f1_weighted"],
                "test_f1_macro": eval_res["metrics"]["f1_macro"],
                "roc_auc_macro": eval_res["metrics"].get("roc_auc_macro", np.nan),
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
    sort_col = "test_f1_weighted" if task_type == "classification" else "test_r2"
    df_comp = df_comp.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    comp_csv = reports_dir / "supervised_model_comparison.csv"
    df_comp.to_csv(comp_csv, index=False)

    best_model_name = str(df_comp.loc[0, "model"])

    return {
        "mode": "supervised",
        "task_type": task_type,
        "target_column": target_col,
        "train_samples": int(len(X_train_proc)),
        "test_samples": int(len(X_test_proc)),
        "n_features_processed": int(X_train_proc.shape[1]),
        "best_model": best_model_name,
        "comparison_table": df_comp.to_dict(orient="records"),
        "comparison_csv": str(comp_csv),
        "saved_models": saved_models,
        "figures": generated_figures,
    }


def run_unsupervised_pipeline(
    df: pd.DataFrame,
    k_range: tuple[int, int],
    k_criterion: str,
    random_state: int,
    visualizer: HDVisualizer,
    reports_dir: Path,
) -> dict[str, Any]:
    """Ejecuta el pipeline no supervisado K-Means con auditoría multi-criterio y PCA 2D."""
    preprocessor = build_preprocessor(df)
    X_proc, _ = _transform_to_dataframe(preprocessor, df)

    engine = KMeansEngine(
        k_range=k_range,
        selection_criterion=k_criterion,
        init="k-means++",
        n_init=10,
        random_state=random_state,
    ).fit(X_proc)

    audit_report = engine.get_audit_report()
    km_figs = visualizer.plot_kmeans_audit(audit_report)

    models_dir = reports_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = engine.save(models_dir / "kmeans_engine.pkl")
    _ = BaseModel.load(model_path)

    metrics_csv = reports_dir / "kmeans_k_sweep_metrics.csv"
    audit_report["metrics_table"].to_csv(metrics_csv, index=False)

    df_segmented = df.copy()
    df_segmented["cluster_id"] = audit_report["labels"]
    df_segmented["silhouette_sample"] = audit_report["sample_silhouettes"]
    seg_csv = reports_dir / "kmeans_segmented_dataset.csv"
    df_segmented.to_csv(seg_csv, index=False)

    return {
        "mode": "unsupervised",
        "algorithm": "K-Means++",
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

    if args.mode == "auto":
        resolved_mode = "supervised" if target_col is not None else "unsupervised"
    else:
        resolved_mode = args.mode

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
        json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8"
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
    main()