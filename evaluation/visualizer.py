from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class HDVisualizer:
    """
    Motor de visualización científica en alta resolución (>= 300 DPI).
    Consume exclusivamente salidas estructuradas de AutomatedEDA, ModelEvaluator y KMeansEngine,
    manteniendo desacoplada la lógica de cálculo de la capa gráfica y liberando memoria tras guardar.
    """

    def __init__(self, output_dir: str | Path = "reports/figures", dpi: int = 300):
        if dpi < 300:
            raise ValueError("La resolución debe ser al menos 300 DPI según especificación técnica.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(dpi)

        sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """Guarda la figura a >= 300 DPI con recorte ajustado y libera memoria RAM."""
        filepath = self.output_dir / filename
        try:
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        finally:
            plt.close(fig)
        return filepath

    def plot_eda_matrices(
        self,
        cov_matrix: pd.DataFrame,
        pearson_matrix: pd.DataFrame,
        spearman_matrix: pd.DataFrame,
        filename: str = "eda_correlation_covariance_hd.png",
    ) -> Path:
        """
        Genera un panel comparativo 1x3 con mapas de calor para Covarianza,
        Correlación lineal de Pearson y Correlación monótona de Spearman.
        """
        n_vars = cov_matrix.shape[0]
        show_annot = n_vars <= 12
        fig_w = max(21.0, n_vars * 0.95)
        fig_h = max(6.5, n_vars * 0.35)

        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

        sns.heatmap(
            cov_matrix,
            ax=axes[0],
            cmap="mako",
            annot=show_annot,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        axes[0].set_title("Matriz de Covarianza", fontweight="bold")

        sns.heatmap(
            pearson_matrix,
            ax=axes[1],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            annot=show_annot,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        axes[1].set_title("Correlación de Pearson (Lineal)", fontweight="bold")

        sns.heatmap(
            spearman_matrix,
            ax=axes[2],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            annot=show_annot,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        axes[2].set_title("Correlación de Spearman (Monótona)", fontweight="bold")

        fig.suptitle(
            "Diagnóstico Multivariado Automático (EDA)",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        return self._save_figure(fig, filename)

    def plot_confusion_matrix(
        self,
        eval_result: dict,
        model_name: str,
        filename: str | None = None,
    ) -> Path:
        """
        Grafica la matriz de confusión normalizada por filas (Recall por clase)
        superponiendo tanto el porcentaje como el conteo absoluto de muestras.
        """
        cm_norm = eval_result["confusion_matrix_normalized"]
        cm_raw = eval_result["confusion_matrix_raw"]
        class_names = eval_result["class_names"]
        n_classes = len(class_names)

        annot_matrix = np.empty_like(cm_norm, dtype=object)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                annot_matrix[i, j] = f"{cm_norm[i, j]*100:.1f}%\n({cm_raw[i, j]})"

        side = max(7.5, n_classes * 0.85)
        annot_size = 10 if n_classes <= 6 else (8 if n_classes <= 12 else 6)

        fig, ax = plt.subplots(figsize=(side, side * 0.85))
        sns.heatmap(
            cm_norm,
            annot=annot_matrix,
            annot_kws={"size": annot_size},
            fmt="",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Proporción por Clase Real (Recall)"},
        )
        ax.set_xlabel("Clase Predicha", fontweight="bold")
        ax.set_ylabel("Clase Real", fontweight="bold")
        acc = eval_result["metrics"]["accuracy"]
        f1w = eval_result["metrics"]["f1_weighted"]
        ax.set_title(
            f"Matriz de Confusión Normalizada — {model_name}\n"
            f"(Accuracy: {acc:.3f} | F1-Weighted: {f1w:.3f})",
            fontweight="bold",
        )

        out_name = filename or f"cm_{model_name.lower().replace(' ', '_')}.png"
        return self._save_figure(fig, out_name)

    def plot_roc_and_pr_comparison(
        self,
        eval_results_dict: dict[str, dict],
        filename: str = "roc_pr_curves_hd.png",
    ) -> Path | None:
        """
        Grafica curvas ROC y Precision-Recall en alta resolución.
        - Si se pasa 1 modelo: grafica cada clase One-vs-Rest + promedio Macro.
        - Si se pasan múltiples modelos: superpone el promedio Macro de cada modelo.
        """
        prob_models = {
            name: res
            for name, res in eval_results_dict.items()
            if res.get("has_probabilities") and res.get("roc_curves") and res.get("pr_curves")
        }
        if not prob_models:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

        if len(prob_models) == 1:
            model_name, res = next(iter(prob_models.items()))
            roc_data = res["roc_curves"]
            pr_data = res["pr_curves"]

            for label, curve in roc_data.items():
                is_macro = label == "macro_average"
                axes[0].plot(
                    curve["fpr"],
                    curve["tpr"],
                    lw=2.5 if is_macro else 1.6,
                    ls="--" if is_macro else "-",
                    label=(
                        f"{'Macro-Promedio' if is_macro else f'Clase {label}'} "
                        f"(AUC = {curve['auc']:.3f})"
                    ),
                )

            for label, curve in pr_data.items():
                axes[1].plot(
                    curve["recall"],
                    curve["precision"],
                    lw=1.8,
                    label=f"Clase {label} (AP = {curve['ap']:.3f})",
                )
            title_suffix = model_name
        else:
            for model_name, res in prob_models.items():
                macro_roc = res["roc_curves"]["macro_average"]
                axes[0].plot(
                    macro_roc["fpr"],
                    macro_roc["tpr"],
                    lw=2.2,
                    label=f"{model_name} (Macro AUC = {macro_roc['auc']:.3f})",
                )

                pr_curves = res["pr_curves"]
                last_class = list(pr_curves.keys())[-1]
                ap_macro = res["metrics"]["pr_auc_macro"]
                axes[1].plot(
                    pr_curves[last_class]["recall"],
                    pr_curves[last_class]["precision"],
                    lw=2.2,
                    label=f"{model_name} (Macro AP = {ap_macro:.3f})",
                )
            title_suffix = "Comparativa Multi-Modelo"

        axes[0].plot(
            [0, 1],
            [0, 1],
            color="gray",
            lw=1.2,
            ls=":",
            label="Clasificador Aleatorio (AUC = 0.500)",
        )
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.03])
        axes[0].set_xlabel("Tasa de Falsos Positivos (FPR)", fontweight="bold")
        axes[0].set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontweight="bold")
        axes[0].set_title(f"Curvas ROC — {title_suffix}", fontweight="bold")
        axes[0].legend(loc="lower right", frameon=True)

        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.03])
        axes[1].set_xlabel("Recall (Exhaustividad)", fontweight="bold")
        axes[1].set_ylabel("Precision (Precisión)", fontweight="bold")
        axes[1].set_title(f"Curvas Precision-Recall — {title_suffix}", fontweight="bold")
        axes[1].legend(loc="lower left", frameon=True)

        return self._save_figure(fig, filename)

    def plot_regression_diagnostics(
        self,
        eval_result: dict,
        model_name: str,
        filename: str | None = None,
    ) -> Path:
        """
        Grafica valores Reales vs. Predichos con banda de confianza al 95% (±1.96 sigma)
        junto al gráfico de dispersión de residuales para auditar homocedasticidad.
        """
        y_true = eval_result["y_true"]
        y_pred = eval_result["y_pred"]
        residuals = eval_result["residuals"]
        band = eval_result["confidence_band_half_width"]
        m = eval_result["metrics"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].scatter(
            y_true,
            y_pred,
            alpha=0.65,
            edgecolors="k",
            linewidths=0.4,
            color="#1f77b4",
            label="Observaciones",
        )
        min_val = float(min(y_true.min(), y_pred.min()))
        max_val = float(max(y_true.max(), y_pred.max()))
        grid = np.linspace(min_val, max_val, 200)

        axes[0].plot(grid, grid, "r--", lw=2, label="Ajuste Ideal ($y = \\hat{y}$)")
        axes[0].fill_between(
            grid,
            grid - band,
            grid + band,
            color="red",
            alpha=0.14,
            label=f"Banda 95% ($\\pm {band:.2f}$)",
        )
        axes[0].set_xlabel("Valor Real ($y$)", fontweight="bold")
        axes[0].set_ylabel("Valor Predicho ($\\hat{y}$)", fontweight="bold")
        axes[0].set_title(
            f"Real vs. Predicho — {model_name}\n"
            f"($R^2$: {m['r2']:.3f} | RMSE: {m['rmse']:.3f} | MAE: {m['mae']:.3f})",
            fontweight="bold",
        )
        axes[0].legend(loc="upper left")

        axes[1].scatter(
            y_pred,
            residuals,
            alpha=0.65,
            edgecolors="k",
            linewidths=0.4,
            color="#2ca02c",
        )
        axes[1].axhline(0.0, color="red", linestyle="--", lw=1.8)
        axes[1].axhline(band, color="gray", linestyle=":", lw=1.2)
        axes[1].axhline(-band, color="gray", linestyle=":", lw=1.2)
        axes[1].set_xlabel("Valor Predicho ($\\hat{y}$)", fontweight="bold")
        axes[1].set_ylabel("Residual ($y - \\hat{y}$)", fontweight="bold")
        axes[1].set_title(f"Diagnóstico de Residuales — {model_name}", fontweight="bold")

        out_name = filename or f"regression_{model_name.lower().replace(' ', '_')}.png"
        return self._save_figure(fig, out_name)

    def plot_learning_curve(
        self,
        lc_result: dict,
        model_name: str,
        filename: str | None = None,
    ) -> Path:
        """
        Grafica las curvas de aprendizaje (Entrenamiento vs. Validación) con bandas de ±1 std
        para diagnosticar visualmente sesgo (underfitting) y varianza (overfitting).
        """
        sizes = lc_result["sample_sizes"]
        tr_mean, tr_std = lc_result["train_error_mean"], lc_result["train_error_std"]
        val_mean, val_std = lc_result["val_error_mean"], lc_result["val_error_std"]

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(sizes, tr_mean, "o-", color="#d62728", lw=2, label="Error de Entrenamiento")
        ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, color="#d62728", alpha=0.15)

        ax.plot(sizes, val_mean, "s-", color="#1f77b4", lw=2, label="Error de Validación Cruzada")
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, color="#1f77b4", alpha=0.15)

        ax.set_xlabel("Tamaño del Conjunto de Entrenamiento (Muestras)", fontweight="bold")
        ax.set_ylabel(lc_result["metric_label"], fontweight="bold")
        ax.set_title(
            f"Curva de Aprendizaje (Sesgo vs. Varianza) — {model_name}", fontweight="bold"
        )
        ax.legend(loc="best")

        out_name = filename or f"learning_curve_{model_name.lower().replace(' ', '_')}.png"
        return self._save_figure(fig, out_name)

    def plot_kmeans_audit(self, audit_report: dict) -> dict[str, Path]:
        """
        Genera los dos paneles diagnósticos del motor no supervisado K-Means:
        1. Barrido de K: Inercia (Codo), Silueta promedio, Calinski-Harabasz y Davies-Bouldin.
        2. Diagrama de Siluetas individuales por clúster + Proyección PCA 2D (o 1D) con centroides.
        """
        df_m = audit_report["metrics_table"]
        opt_k = int(audit_report["optimal_k"])
        k_vals = df_m["k"]

        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
        panels = [
            ("inertia", "Inercia Intra-Clúster (WCSS — Método del Codo)", "#1f77b4", axes1[0, 0]),
            ("silhouette", "Coeficiente de Silueta Promedio (Mayor es mejor)", "#2ca02c", axes1[0, 1]),
            ("calinski_harabasz", "Índice Calinski-Harabasz (Mayor es mejor)", "#9467bd", axes1[1, 0]),
            ("davies_bouldin", "Índice Davies-Bouldin (Menor es mejor)", "#d62728", axes1[1, 1]),
        ]

        for col, title, color, ax in panels:
            ax.plot(k_vals, df_m[col], marker="o", lw=2, color=color)
            ax.axvline(
                opt_k,
                color="black",
                linestyle="--",
                lw=1.4,
                label=f"$K^*$ seleccionado = {opt_k}",
            )
            ax.set_xlabel("Número de Clústeres ($K$)", fontweight="bold")
            ax.set_title(title, fontweight="bold")
            ax.legend(loc="best")

        fig1.suptitle(
            "Auditoría Multi-Criterio para Selección de $K$",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        path_sweep = self._save_figure(fig1, "kmeans_k_selection_sweep.png")

        labels = audit_report["labels"]
        sample_sil = audit_report["sample_silhouettes"]
        avg_sil = float(audit_report["avg_silhouette"])
        X_pca = audit_report["X_pca"]
        centroids_pca = audit_report["centroids_pca"]
        var_ratio = audit_report["pca_explained_variance"]
        is_2d = X_pca.shape[1] >= 2

        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6.5))

        y_lower = 10
        cmap = plt.get_cmap("viridis", max(opt_k, 2))
        for i in range(opt_k):
            ith_sil_vals = np.sort(sample_sil[labels == i])
            size_cluster_i = ith_sil_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cmap(i)

            axes2[0].fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_sil_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
            )
            axes2[0].text(-0.04, y_lower + 0.5 * size_cluster_i, f"C{i}", fontweight="bold")
            y_lower = y_upper + 10

        axes2[0].axvline(
            avg_sil,
            color="red",
            linestyle="--",
            lw=1.8,
            label=f"Silueta Media = {avg_sil:.3f}",
        )
        axes2[0].set_xlim([-0.15, 1.0])
        axes2[0].set_ylim([0, len(labels) + (opt_k + 1) * 10])
        axes2[0].set_yticks([])
        axes2[0].set_xlabel("Coeficiente de Silueta Individual $s(i)$", fontweight="bold")
        axes2[0].set_ylabel("Clústeres Ordenados por Cohesión", fontweight="bold")
        axes2[0].set_title(f"Análisis de Silueta por Observación ($K={opt_k}$)", fontweight="bold")
        axes2[0].legend(loc="upper right")

        for i in range(opt_k):
            mask = labels == i
            y_coords = X_pca[mask, 1] if is_2d else np.zeros(int(mask.sum()))
            axes2[1].scatter(
                X_pca[mask, 0],
                y_coords,
                s=38,
                color=cmap(i),
                alpha=0.72,
                edgecolors="k",
                linewidths=0.3,
                label=f"Clúster {i} (n={mask.sum()})",
            )

        cent_y = centroids_pca[:, 1] if is_2d else np.zeros(centroids_pca.shape[0])
        axes2[1].scatter(
            centroids_pca[:, 0],
            cent_y,
            marker="X",
            s=220,
            color="red",
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
            label="Centroides Proyectados",
        )
        axes2[1].set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% varianza)", fontweight="bold")
        axes2[1].set_ylabel(
            f"PC2 ({var_ratio[1]*100:.1f}% varianza)" if is_2d else "Eje 1D (Proyección)",
            fontweight="bold",
        )
        axes2[1].set_title(
            f"Segmentación K-Means en Subespacio PCA ({'2D' if is_2d else '1D'}, $K={opt_k}$)",
            fontweight="bold",
        )
        axes2[1].legend(loc="best")

        path_clusters = self._save_figure(fig2, "kmeans_silhouette_and_pca.png")

        return {"k_selection": path_sweep, "silhouette_pca": path_clusters}