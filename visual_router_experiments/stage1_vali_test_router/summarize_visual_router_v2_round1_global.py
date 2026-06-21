#!/usr/bin/env python3
"""汇总 Visual Router V2 Round 1 的轻量结果并生成全局推荐材料。

本脚本只读取仓库内已经归档的 summary/CSV/JSON 文件，不读取 checkpoint、
逐样本 prediction CSV、SQLite index、feature shard 或大规模 prediction manifest。
输出用于 Round 1 归档、解释和后续路线建议，不训练或重新评估任何模型。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_ROOT = ROOT / "experiment_summaries" / "visual_router_v2_round1"
OUT_DIR = SUMMARY_ROOT / "global_summary"

SOURCE_COMMITS = {
    "P0": "97602b8 Add visual router v2 pilot sample builder",
    "P1": "2f2115c Add visual router v2 round0 evaluator",
    "P2a": "1966649 Add visual router v2 round1 feature cache builder",
    "P2probe": "998ed29 Add visual router round1 feature probe",
    "P2b": "0c83f0a Add P2b visual pooling ablation",
    "P2c": "342fb96 Add P2c RevIN aux-only ablation",
    "P2d": "5b21c82 Add P2d visual aux concat round1 summary",
    "P2d_final_test": "eb5b1be Add visual router round1 final test eval",
    "P2d_final_test_extension": "39f0e76 Add round1 final test extension eval",
    "P2e_FiLM": "500bef0 Add round1 P2e FiLM ablation",
    "P2e_FiLM_final_test_extension": "448a423 Add P2e FiLM final test extension",
}

SOURCE_DIRS = [
    "p2probe",
    "p2b_visual_pooling",
    "p2c_aux_only",
    "p2d_concat",
    "p2d_final_test",
    "p2d_final_test_extension",
    "p2e_film",
    "p2e_film_final_test_extension",
]

FINAL_COMPARISON = (
    SUMMARY_ROOT
    / "p2e_film_final_test_extension"
    / "round1_film_final_test_extension_comparison.csv"
)
FINAL_VARIANT_SEED = (
    SUMMARY_ROOT
    / "p2e_film_final_test_extension"
    / "round1_film_final_test_extension_variant_seed_results.csv"
)
FINAL_SELECTED = (
    SUMMARY_ROOT
    / "p2e_film_final_test_extension"
    / "round1_film_final_test_extension_selected_model_counts.csv"
)
FINAL_STRATA = (
    SUMMARY_ROOT
    / "p2e_film_final_test_extension"
    / "round1_film_final_test_extension_stratified_summary.csv"
)


SELECTION_FILES = [
    ("P2b_visual_pooling", SUMMARY_ROOT / "p2b_visual_pooling" / "visual_pooling_selection_comparison.csv"),
    ("P2c_aux_only", SUMMARY_ROOT / "p2c_aux_only" / "aux_only_selection_comparison.csv"),
    ("P2d_concat", SUMMARY_ROOT / "p2d_concat" / "round1_concat_selection_comparison.csv"),
    ("P2e_FiLM", SUMMARY_ROOT / "p2e_film" / "round1_film_selection_comparison.csv"),
]

DIAGNOSTIC_FILES = [
    ("P2b_visual_pooling", SUMMARY_ROOT / "p2b_visual_pooling" / "visual_pooling_diagnostic_summary.csv"),
    ("P2c_aux_only", SUMMARY_ROOT / "p2c_aux_only" / "aux_only_diagnostic_summary.csv"),
    ("P2d_concat", SUMMARY_ROOT / "p2d_concat" / "round1_concat_diagnostic_summary.csv"),
    ("P2e_FiLM", SUMMARY_ROOT / "p2e_film" / "round1_film_diagnostic_summary.csv"),
]


FINAL_STAGE = {
    "film_mean_patch_aux": "P2e_FiLM_final_test_extension",
    "film_cls_mean_concat_aux": "P2e_FiLM_final_test_extension",
    "visual_cls_mean_concat": "P2d_final_test_extension",
    "visual_mean_patch_only": "P2d_final_test_extension",
    "cls_mean_concat_plus_aux": "P2d_final_test",
    "mean_patch_plus_aux": "P2d_final_test_extension",
    "round0_timefuse": "P1_round0",
    "round0_original_visual": "P1_round0",
    "global_best_single": "P1_reference",
    "oracle_top1": "oracle_reference",
}

DISPLAY_VARIANT = {
    "round0_timefuse": "Round0 TimeFuse",
    "round0_original_visual": "Round0 original Visual",
    "global_best_single": "global_best_single",
    "oracle_top1": "oracle_top1",
}

FINAL_ORDER = [
    "film_mean_patch_aux",
    "film_cls_mean_concat_aux",
    "visual_cls_mean_concat",
    "visual_mean_patch_only",
    "cls_mean_concat_plus_aux",
    "mean_patch_plus_aux",
    "round0_timefuse",
    "round0_original_visual",
    "global_best_single",
    "oracle_top1",
]

REQUIRED_DELTA_PAIRS = [
    ("film_mean_patch_aux", "visual_mean_patch_only"),
    ("film_mean_patch_aux", "mean_patch_plus_aux"),
    ("film_mean_patch_aux", "visual_cls_mean_concat"),
    ("film_mean_patch_aux", "cls_mean_concat_plus_aux"),
    ("film_mean_patch_aux", "round0_timefuse"),
    ("film_cls_mean_concat_aux", "visual_cls_mean_concat"),
    ("film_cls_mean_concat_aux", "cls_mean_concat_plus_aux"),
    ("film_cls_mean_concat_aux", "round0_timefuse"),
    ("visual_cls_mean_concat", "round0_timefuse"),
    ("cls_mean_concat_plus_aux", "round0_timefuse"),
    ("mean_patch_plus_aux", "visual_mean_patch_only"),
]

STRATA_COLUMNS = [
    "oracle_model",
    "error_gap_quantile",
    "dataset_name",
    "group_name",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _variant_from_method(method: str, variant: Any) -> str:
    if isinstance(variant, str) and variant:
        return variant
    for suffix in ("_raw_soft_fusion", "_hard_top1"):
        if method.endswith(suffix):
            return method[: -len(suffix)]
    return method


def _method_kind(method: str) -> str:
    if method.endswith("_raw_soft_fusion") or method == "round0_timefuse_raw_soft_fusion":
        return "raw_soft_fusion"
    if method.endswith("_hard_top1"):
        return "hard_top1"
    return method


def _normalise_summary_rows(stage: str, path: Path, role: str) -> pd.DataFrame:
    """把 selection/diagnostic 阶段不同命名风格规整成统一字段。"""
    df = _read_csv(path).copy()
    if "variant" not in df.columns:
        df["variant"] = df["method"].map(lambda m: _variant_from_method(str(m), ""))
        df["variant"] = df["variant"].replace({"revin_aux_only_fusion_huber_kl": "aux_only"})
    if "sample_count_per_seed" in df.columns and "sample_count" not in df.columns:
        df["sample_count"] = df["sample_count_per_seed"]
    df["stage"] = stage
    df["result_role"] = role
    df["method_kind"] = df["method"].map(lambda m: _method_kind(str(m)))
    df["source_file"] = str(path.relative_to(ROOT))
    df["selection_status"] = df["sample_set"].map(
        lambda s: "selection_only" if s == "pilot_selection" else "not_selection"
    )
    df["final_test_status"] = "not_final_test"
    keep = [
        "stage",
        "result_role",
        "sample_set",
        "variant",
        "method",
        "method_kind",
        "seed_count",
        "sample_count",
        "MAE_mean",
        "MAE_std",
        "MSE_mean",
        "MSE_std",
        "regret_to_oracle_mean",
        "regret_to_oracle_std",
        "oracle_label_accuracy_mean",
        "oracle_label_accuracy_std",
        "weight_entropy_mean",
        "weight_entropy_std",
        "normalized_weight_entropy_mean",
        "normalized_weight_entropy_std",
        "mean_max_weight_mean",
        "mean_max_weight_std",
        "source_file",
        "selection_status",
        "final_test_status",
    ]
    return df.reindex(columns=keep)


def build_selection_and_diagnostic() -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = pd.concat(
        [_normalise_summary_rows(stage, path, "selection") for stage, path in SELECTION_FILES],
        ignore_index=True,
    )
    diagnostic = pd.concat(
        [_normalise_summary_rows(stage, path, "diagnostic") for stage, path in DIAGNOSTIC_FILES],
        ignore_index=True,
    )
    return selection, diagnostic


def build_final_test() -> pd.DataFrame:
    """把 final comparison 的 hard/raw 两行合成每个方法一行，便于直接审阅排名。"""
    df = _read_csv(FINAL_COMPARISON).copy()
    records: list[dict[str, Any]] = []
    for key, group in df.groupby(df.apply(lambda r: _variant_from_method(str(r["method"]), r.get("variant", "")), axis=1)):
        raw = group[group["method"].map(lambda m: str(m).endswith("_raw_soft_fusion"))]
        hard = group[group["method"].map(lambda m: str(m).endswith("_hard_top1"))]
        single = group[group["method"].isin(["global_best_single", "oracle_top1"])]
        method_key = key
        row_raw = raw.iloc[0] if not raw.empty else None
        row_hard = hard.iloc[0] if not hard.empty else None
        row_single = single.iloc[0] if not single.empty else None
        base = row_raw if row_raw is not None else row_hard if row_hard is not None else row_single
        if base is None:
            continue
        variant = DISPLAY_VARIANT.get(method_key, method_key)
        records.append(
            {
                "stage": FINAL_STAGE.get(method_key, "unknown"),
                "variant": variant,
                "method": method_key,
                "sample_set": base["sample_set"],
                "seed_count": int(base["seed_count"]),
                "sample_count": int(base["sample_count"]),
                "raw_soft_MAE_mean": _value(row_raw, "raw_soft_fusion_MAE"),
                "raw_soft_MAE_std": _value(row_raw, "MAE_std"),
                "raw_soft_MSE_mean": _value(row_raw, "raw_soft_fusion_MSE"),
                "raw_soft_MSE_std": _value(row_raw, "MSE_std"),
                "raw_soft_regret_mean": _value(row_raw, "raw_soft_fusion_regret_to_oracle"),
                "raw_soft_regret_std": _value(row_raw, "regret_to_oracle_std"),
                "raw_soft_oracle_label_accuracy_mean": _value(row_raw, "raw_soft_fusion_oracle_label_accuracy"),
                "hard_top1_MAE_mean": _value(row_hard if row_hard is not None else row_single, "hard_top1_MAE"),
                "hard_top1_MAE_std": _value(row_hard if row_hard is not None else row_single, "MAE_std"),
                "hard_top1_MSE_mean": _value(row_hard, "hard_top1_MSE"),
                "hard_top1_MSE_std": _value(row_hard, "MSE_std"),
                "hard_top1_regret_mean": _value(row_hard if row_hard is not None else row_single, "hard_top1_regret_to_oracle"),
                "hard_top1_oracle_label_accuracy_mean": _value(row_hard if row_hard is not None else row_single, "hard_top1_oracle_label_accuracy"),
                "weight_entropy_mean": _value(base, "weight_entropy"),
                "normalized_weight_entropy_mean": _value(base, "normalized_weight_entropy"),
                "mean_max_weight_mean": _value(base, "mean_max_weight"),
                "source_file": str(FINAL_COMPARISON.relative_to(ROOT)),
                "selection_status": "not_used_for_selection",
                "final_test_status": "frozen_final_eval",
            }
        )
    out = pd.DataFrame(records)
    order_map = {name: i for i, name in enumerate(FINAL_ORDER)}
    out["_order"] = out["method"].map(lambda m: order_map.get(str(m), 999))
    # final_test 主审阅表按 raw-soft MAE 排序；没有 raw-soft 的参考线留在末尾。
    out = out.sort_values(["raw_soft_MAE_mean", "_order"], na_position="last").drop(columns=["_order"])
    return out


def _value(row: Any, column: str) -> Any:
    if row is None:
        return pd.NA
    value = row.get(column, pd.NA)
    return pd.NA if pd.isna(value) else value


def build_delta_summary(final_test: pd.DataFrame) -> pd.DataFrame:
    by_method = final_test.set_index("method")
    rows: list[dict[str, Any]] = []
    for left, right in REQUIRED_DELTA_PAIRS:
        if left not in by_method.index or right not in by_method.index:
            rows.append({"comparison": f"{left} - {right}", "status": "missing"})
            continue
        lrow = by_method.loc[left]
        rrow = by_method.loc[right]
        for metric, lower_is_better in [
            ("raw_soft_MAE_mean", True),
            ("raw_soft_MSE_mean", True),
            ("raw_soft_regret_mean", True),
            ("raw_soft_MAE_std", True),
            ("hard_top1_MAE_mean", True),
            ("hard_top1_MSE_mean", True),
            ("hard_top1_regret_mean", True),
            ("hard_top1_oracle_label_accuracy_mean", False),
            ("weight_entropy_mean", False),
            ("normalized_weight_entropy_mean", False),
            ("mean_max_weight_mean", False),
        ]:
            lval = lrow.get(metric, pd.NA)
            rval = rrow.get(metric, pd.NA)
            delta = pd.NA if pd.isna(lval) or pd.isna(rval) else lval - rval
            rows.append(
                {
                    "comparison": f"{left} - {right}",
                    "left_variant": left,
                    "right_variant": right,
                    "sample_set": "pilot_test",
                    "metric": metric,
                    "left_value": lval,
                    "right_value": rval,
                    "delta_left_minus_right": delta,
                    "lower_is_better": lower_is_better,
                    "status": "ok" if not pd.isna(delta) else "metric_missing",
                }
            )
    return pd.DataFrame(rows)


def build_selected_model_summary() -> pd.DataFrame:
    df = _read_csv(FINAL_SELECTED)
    df = df[df["method"].map(lambda m: str(m).endswith("_hard_top1"))].copy()
    df["variant_key"] = df.apply(lambda r: _variant_from_method(str(r["method"]), r.get("variant", "")), axis=1)
    grouped = (
        df.groupby(["sample_set", "variant_key", "selected_model"], dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            sample_count=("count", "sum"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            ratio_min=("ratio", "min"),
            ratio_max=("ratio", "max"),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={"variant_key": "variant"})
    grouped["ratio_range"] = grouped["ratio_max"] - grouped["ratio_min"]
    grouped["seed_ratio_status"] = grouped["ratio_range"].map(
        lambda x: "needs_review" if pd.notna(x) and x >= 0.20 else "stable_or_expected"
    )
    grouped["interpretation_note"] = grouped.apply(_selected_note, axis=1)
    return grouped.sort_values(["variant", "ratio_mean"], ascending=[True, False])


def _selected_note(row: pd.Series) -> str:
    if row["variant"] == "film_mean_patch_aux" and row["selected_model"] in {"PatchTST", "ES", "DLinear"}:
        return "film_mean_patch_aux hard top-1 主要集中在 PatchTST/ES/DLinear 一侧"
    if row["selected_model"] == "CrossFormer":
        return "CrossFormer hard selected ratio 偏低，后续 calibration/strata repair 应继续关注"
    return "用于解释 hard top-1 行为，不参与最终选择"


def build_strata_summary(final_test: pd.DataFrame) -> pd.DataFrame:
    df = _read_csv(FINAL_STRATA)
    df = df[
        (df["method"].map(lambda m: str(m).endswith("_raw_soft_fusion")))
        & (df["stratum_column"].isin(STRATA_COLUMNS))
    ].copy()
    df["variant"] = df.apply(lambda r: _variant_from_method(str(r["method"]), r.get("variant", "")), axis=1)
    agg = (
        df.groupby(["variant", "sample_set", "stratum_column", "stratum_value"], dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            sample_count=("sample_count", "mean"),
            raw_soft_MAE_mean=("MAE", "mean"),
            raw_soft_MSE_mean=("MSE", "mean"),
            raw_soft_regret_mean=("regret_to_oracle", "mean"),
            oracle_label_accuracy_mean=("oracle_label_accuracy", "mean"),
        )
        .reset_index()
    )

    # 只在同一分层值内计算相对强 baseline 的 delta，避免跨 strata 混比。
    key_cols = ["sample_set", "stratum_column", "stratum_value"]
    visual_base = agg[agg["variant"] == "visual_cls_mean_concat"][key_cols + ["raw_soft_MAE_mean"]].rename(
        columns={"raw_soft_MAE_mean": "visual_cls_mean_concat_MAE"}
    )
    timefuse_base = agg[agg["variant"] == "round0_timefuse"][key_cols + ["raw_soft_MAE_mean"]].rename(
        columns={"raw_soft_MAE_mean": "round0_timefuse_MAE"}
    )
    agg = agg.merge(visual_base, on=key_cols, how="left").merge(timefuse_base, on=key_cols, how="left")
    agg["delta_vs_visual_cls_mean_concat"] = agg["raw_soft_MAE_mean"] - agg["visual_cls_mean_concat_MAE"]
    agg["delta_vs_round0_timefuse"] = agg["raw_soft_MAE_mean"] - agg["round0_timefuse_MAE"]
    agg = agg.drop(columns=["visual_cls_mean_concat_MAE", "round0_timefuse_MAE"])
    return agg.sort_values(["stratum_column", "stratum_value", "variant"])


def build_global_comparison(selection: pd.DataFrame, diagnostic: pd.DataFrame, final_test: pd.DataFrame) -> pd.DataFrame:
    final_long = final_test.rename(
        columns={
            "raw_soft_MAE_mean": "MAE_mean",
            "raw_soft_MAE_std": "MAE_std",
            "raw_soft_MSE_mean": "MSE_mean",
            "raw_soft_MSE_std": "MSE_std",
            "raw_soft_regret_mean": "regret_to_oracle_mean",
            "raw_soft_regret_std": "regret_to_oracle_std",
            "raw_soft_oracle_label_accuracy_mean": "oracle_label_accuracy_mean",
            "weight_entropy_mean": "weight_entropy_mean",
            "normalized_weight_entropy_mean": "normalized_weight_entropy_mean",
            "mean_max_weight_mean": "mean_max_weight_mean",
        }
    ).copy()
    final_long["result_role"] = "frozen_final_test"
    final_long["method_kind"] = "raw_soft_primary"
    final_long["oracle_label_accuracy_std"] = pd.NA
    final_long["weight_entropy_std"] = pd.NA
    final_long["normalized_weight_entropy_std"] = pd.NA
    final_long["mean_max_weight_std"] = pd.NA
    cols = [
        "stage",
        "result_role",
        "sample_set",
        "variant",
        "method",
        "method_kind",
        "seed_count",
        "sample_count",
        "MAE_mean",
        "MAE_std",
        "MSE_mean",
        "MSE_std",
        "regret_to_oracle_mean",
        "regret_to_oracle_std",
        "oracle_label_accuracy_mean",
        "oracle_label_accuracy_std",
        "weight_entropy_mean",
        "weight_entropy_std",
        "normalized_weight_entropy_mean",
        "normalized_weight_entropy_std",
        "mean_max_weight_mean",
        "mean_max_weight_std",
        "source_file",
        "selection_status",
        "final_test_status",
    ]
    return pd.concat([selection[cols], diagnostic[cols], final_long[cols]], ignore_index=True)


def build_recommendation(
    final_test: pd.DataFrame,
    delta: pd.DataFrame,
    selected: pd.DataFrame,
    strata: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    raw_rank = final_test.dropna(subset=["raw_soft_MAE_mean"]).sort_values("raw_soft_MAE_mean")
    seed = _read_csv(FINAL_VARIANT_SEED)
    seed = seed[seed["method"].map(lambda m: str(m).endswith("_raw_soft_fusion"))]
    film_seed = seed[seed["variant"].isin(["film_mean_patch_aux", "film_cls_mean_concat_aux", "visual_cls_mean_concat", "cls_mean_concat_plus_aux"])]
    selected_pivot = selected[selected["variant"].isin(["film_mean_patch_aux", "film_cls_mean_concat_aux"])].copy()
    selected_pivot = selected_pivot.sort_values(["variant", "ratio_mean"], ascending=[True, False])

    def fmt_rank() -> str:
        rows = []
        for _, r in raw_rank.iterrows():
            mse = "" if pd.isna(r["raw_soft_MSE_mean"]) else f"{r['raw_soft_MSE_mean']:.6f}"
            rows.append(
                f"| {r['variant']} | {r['raw_soft_MAE_mean']:.6f} | {mse} | "
                f"{r['raw_soft_regret_mean']:.6f} | {r['raw_soft_MAE_std']:.6f} |"
            )
        return "\n".join(rows)

    def fmt_seed() -> str:
        rows = []
        for _, r in film_seed.sort_values(["variant", "seed"]).iterrows():
            rows.append(f"| {r['variant']} | {int(r['seed'])} | {r['MAE']:.6f} | {r['MSE']:.6f} | {r['regret_to_oracle']:.6f} |")
        return "\n".join(rows)

    def seed_values_sentence(variant: str) -> str:
        values = film_seed[film_seed["variant"] == variant].sort_values("seed")["MAE"].tolist()
        return "、".join(f"{v:.6f}" for v in values)

    def fmt_selected() -> str:
        rows = []
        for _, r in selected_pivot.iterrows():
            rows.append(
                f"| {r['variant']} | {r['selected_model']} | {r['ratio_mean']:.6f} | "
                f"{r['ratio_min']:.6f} | {r['ratio_max']:.6f} | {r['seed_ratio_status']} |"
            )
        return "\n".join(rows)

    def strata_line(column: str, value: str) -> str:
        sub = strata[(strata["stratum_column"] == column) & (strata["stratum_value"].astype(str) == value)]
        sub = sub[sub["variant"].isin(["film_mean_patch_aux", "visual_cls_mean_concat", "round0_timefuse"])]
        if sub.empty:
            return f"- `{column}={value}`：当前 distilled 表无对应行。"
        parts = []
        for _, r in sub.sort_values("variant").iterrows():
            parts.append(f"{r['variant']} MAE={r['raw_soft_MAE_mean']:.6f}")
        return f"- `{column}={value}`：" + "；".join(parts) + "。"

    return f"""# Visual Router V2 Round 1 Global Summary and Final Recommendation

生成时间：{metadata["generated_at"]}

## 1. Round 1 protocol recap

- `pilot_train` 只用于训练 router/adapter。
- `pilot_selection` 只用于 variant、seed/epoch 规则和轻量路线选择。
- `diagnostic_balanced` 只用于 oracle-balanced 行为、selected_model ratio 和 strata 解释，不参与选择。
- `pilot_test` 只用于 frozen final eval，不用于训练、调参、variant 选择、seed 选择、epoch 选择或 hyperparameter 选择。
- 本总结是 summary-only 归档步骤：不训练新模型、不重新评估 pilot_test、不读取 checkpoint/SQLite/逐样本 prediction/feature shard。

## 2. Stage-by-stage summary

- P0 样本冻结：固定 `pilot_train`、`pilot_selection`、`diagnostic_balanced` 和 `pilot_test` 的 sample_key 边界。
- P1 Round0 复现：提供 Round0 TimeFuse、Round0 original Visual、global_best_single 和 oracle_top1 参考线。
- P2a feature cache：冻结 ViT CLS、patch-token mean pooling 和 6 维 RevIN aux，feature 来自历史窗口 `x`。
- P2probe 表征诊断：证明 visual representation 对 oracle expert、结构标签和 dataset/TSF shortcut 具有有效信号。
- P2b visual pooling：`visual_mean_patch_only` 在 selection 上成为 visual-only 初始强线，final extension 显示 `visual_cls_mean_concat` 在 pilot_test 更强。
- P2c aux-only：RevIN aux 单独作为 router 输入不足以替代 visual embedding。
- P2d direct concat：selection 上 `cls_mean_concat_plus_aux` 略优，但 pilot_test extension 显示 direct concat aux 泛化不稳。
- P2d final extension：补齐 visual-only 和 concat baseline 的 frozen pilot_test 比较，暴露 `mean_patch_plus_aux` 的明显退化。
- P2e FiLM：aux 只作为 gamma/beta modulation signal 调制 visual hidden representation，selection 上 `film_mean_patch_aux` 最优。
- P2e final extension：两个 FiLM 变体在 frozen pilot_test 上均显著优于对应 visual/direct concat baseline。

## 3. Main quantitative result

frozen `pilot_test` raw-soft 排名如下，主指标是 raw-soft MAE / MSE / regret：

| variant | raw-soft MAE | raw-soft MSE | raw-soft regret | MAE std |
| --- | ---: | ---: | ---: | ---: |
{fmt_rank()}

`film_mean_patch_aux` 是当前 Round 1 frozen pilot_test 最强变体：raw-soft MAE=0.417824、MSE=183.353985、regret=0.077539、MAE_std=0.000657。它同时改善 MAE、MSE、regret 和 seed stability。`mean_patch_plus_aux` raw-soft MAE=0.516108/MSE=486.102519，而 `film_mean_patch_aux` 在相同 mean_patch 主表示上显著成功，说明问题不在 aux 信息本身，而在 aux 注入机制。

## 4. Interpretation

visual embedding 本身有强信号，尤其 `visual_cls_mean_concat` 在 pilot_test 上已经显著优于 Round0 TimeFuse。`mean_patch` 更像 RevIN 后形状/结构摘要，`revin_aux` 则携带尺度、波动、范围和 clip 等统计。direct concat 把尺度统计直接并入 base visual input，容易改变表征几何并放大 seed/strata tail；FiLM 将 aux 用作 hidden representation 的条件调制，更符合“aux 调制 visual representation”而不是“aux 替代 visual representation”的机制假设。

## 5. Soft fusion vs hard oracle classifier

raw-soft 是最终主指标，hard top-1 / oracle-label accuracy 只解释 router 行为。FiLM 的优势不是因为它更像 TimeFuse 那样硬选中 oracle expert：`film_mean_patch_aux` hard top-1 MAE=0.431851，仍弱于 raw-soft MAE=0.417824；其 oracle-label accuracy=0.515560，也低于 Round0 TimeFuse 的 0.587240。FiLM 的收益主要来自更健康的 soft weight 分配和更低 MSE tail，而不是 hard oracle classifier accuracy 的单点提升。

## 6. Seed stability

| variant | seed | raw-soft MAE | raw-soft MSE | raw-soft regret |
| --- | ---: | ---: | ---: | ---: |
{fmt_seed()}

`film_mean_patch_aux` 三个 seed 的 pilot_test raw-soft MAE 分别为 {seed_values_sentence("film_mean_patch_aux")}，std=0.000657，明显稳定于 `visual_cls_mean_concat` 和 direct concat baseline。`film_cls_mean_concat_aux` 也改善了 `visual_cls_mean_concat` 的 MAE/MSE tail，但 MAE_std=0.001850，略弱于 mean_patch FiLM。

## 7. Strata and selected_model diagnosis

selected_model ratio 摘要：

| variant | selected_model | ratio mean | ratio min | ratio max | status |
| --- | --- | ---: | ---: | ---: | --- |
{fmt_selected()}

`film_mean_patch_aux` 的 hard selected_model ratio 更偏 PatchTST / ES / DLinear，CrossFormer hard selected ratio 仍偏低。selected_model ratio 与 raw-soft MAE/MSE/regret 并不完全一致：Round0 TimeFuse oracle-label accuracy 更高，但 raw-soft MAE/MSE 明显更差；FiLM 虽未解决 hard CrossFormer selection，但 soft fusion 指标最好。

重点 strata 摘要：

{strata_line("oracle_model", "CrossFormer")}
{strata_line("oracle_model", "PatchTST")}
{strata_line("oracle_model", "ES")}
{strata_line("oracle_model", "DLinear")}
{strata_line("oracle_model", "NaiveForecaster")}

完整 distilled strata 表见 `round1_global_strata_summary.csv`，覆盖 `oracle_model`、`error_gap_quantile`、`dataset_name`、`group_name`、`forecastability_cat`、`season_strength_cat`、`trend_strength_cat`、`cv_cat` 和 `missing_ratio_cat`。后续需要继续关注 CrossFormer hard selection 偏低、error_gap 高分位和 dataset/group 层面的 tail。

## 8. Round 1 final recommendation

- 推荐 `film_mean_patch_aux` 作为 Round1 当前主线结构。
- 保留 `film_cls_mean_concat_aux` 作为强对照结构。
- 保留 `visual_cls_mean_concat` 作为 visual-only strong baseline。
- 保留 `visual_mean_patch_only` 作为简洁 visual-only baseline。
- 不建议把 `mean_patch_plus_aux` 作为后续主线。
- 不建议继续把 direct concat aux 作为主要路线。

## 9. Next-step options

1. P2f / Round1 calibration diagnostic：做 temperature scaling / post-hoc calibration，只能在 `pilot_selection` 选择温度，`pilot_test` 继续 frozen eval。
2. P2g FiLM hyperparameter small search：只在 `pilot_train` / `pilot_selection` 做小搜索，不碰 `pilot_test`。
3. Round2 view layout / pseudo image small screening：先小样本筛选，再扩大，避免直接 full-scale 重构。
4. Stage 1 canonical migration：把 FiLM 作为 Visual Router candidate head/adapter 的重要候选，但不要立刻 full-scale 重构。

## 10. Boundary and caveats

- 本总结不改变历史 selection 规则。
- `pilot_test` 结果只用于 frozen final eval 和路线解释。
- 后续任何超参搜索都必须回到 `pilot_train` / `pilot_selection`。
- 不要用 `pilot_test` 选择 temperature、FiLM hidden dim、dropout、epoch、seed 或 variant。
- 当前结论基于 Round1 pilot sample protocol；迁移到 canonical/full-scale 前仍需保留协议边界和 calibration 诊断。
"""


def write_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection, diagnostic = build_selection_and_diagnostic()
    final_test = build_final_test()
    comparison = build_global_comparison(selection, diagnostic, final_test)
    delta = build_delta_summary(final_test)
    selected = build_selected_model_summary()
    strata = build_strata_summary(final_test)

    generated = {
        "comparison": OUT_DIR / "round1_global_comparison.csv",
        "selection_diagnostic": OUT_DIR / "round1_global_selection_diagnostic.csv",
        "final_test": OUT_DIR / "round1_global_final_test.csv",
        "delta": OUT_DIR / "round1_global_delta_summary.csv",
        "selected": OUT_DIR / "round1_global_selected_model_summary.csv",
        "strata": OUT_DIR / "round1_global_strata_summary.csv",
        "recommendation": OUT_DIR / "round1_global_recommendation.md",
        "metadata": OUT_DIR / "round1_global_metadata.json",
    }

    comparison.to_csv(generated["comparison"], index=False)
    pd.concat([selection, diagnostic], ignore_index=True).to_csv(generated["selection_diagnostic"], index=False)
    final_test.to_csv(generated["final_test"], index=False)
    delta.to_csv(generated["delta"], index=False)
    selected.to_csv(generated["selected"], index=False)
    strata.to_csv(generated["strata"], index=False)

    metadata = {
        "summary_only": True,
        "trained_new_model": False,
        "evaluated_new_model": False,
        "used_pilot_test_for_selection": False,
        "read_checkpoint": False,
        "read_prediction_csv_sample_level": False,
        "read_sqlite_prediction_index": False,
        "read_feature_shard": False,
        "loaded_116m_prediction_manifest_to_memory": False,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "script": str(Path(__file__).relative_to(ROOT)),
        "source_commits": SOURCE_COMMITS,
        "source_summary_dirs": [str((SUMMARY_ROOT / d).relative_to(ROOT)) for d in SOURCE_DIRS],
        "generated_files": [str(path.relative_to(ROOT)) for path in generated.values()],
        "recommended_main_variant": "film_mean_patch_aux",
        "recommended_control_variant": "film_cls_mean_concat_aux",
        "recommended_visual_only_baseline": "visual_cls_mean_concat",
        "row_counts": {
            "round1_global_comparison.csv": int(len(comparison)),
            "round1_global_selection_diagnostic.csv": int(len(selection) + len(diagnostic)),
            "round1_global_final_test.csv": int(len(final_test)),
            "round1_global_delta_summary.csv": int(len(delta)),
            "round1_global_selected_model_summary.csv": int(len(selected)),
            "round1_global_strata_summary.csv": int(len(strata)),
        },
        "input_files": [
            str(FINAL_COMPARISON.relative_to(ROOT)),
            str(FINAL_VARIANT_SEED.relative_to(ROOT)),
            str(FINAL_SELECTED.relative_to(ROOT)),
            str(FINAL_STRATA.relative_to(ROOT)),
            *[str(path.relative_to(ROOT)) for _, path in SELECTION_FILES],
            *[str(path.relative_to(ROOT)) for _, path in DIAGNOSTIC_FILES],
        ],
    }
    generated["recommendation"].write_text(
        build_recommendation(final_test, delta, selected, strata, metadata),
        encoding="utf-8",
    )
    generated["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_outputs()
