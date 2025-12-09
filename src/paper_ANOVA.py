### Functions for fitting ANOVA/ANCOVA models to pd.DataFrames, with inclulded post-hoc comparisons, effect sizes, and bootstrap confidence intervals.

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

import re
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


@dataclass
class EffectSizeResult:
    """Container for effect size results"""

    value: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    boot_distribution: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"EffectSize(value={self.value:.4f}, "
            f"CI{int(self.ci_level*100)}=[{self.ci_low:.4f}, {self.ci_high:.4f}])"
        )


@dataclass
class PairwiseComparison:
    """Container for a single pairwise comparison result."""

    level_a: str
    level_b: str
    mean_diff: float
    diff_ci_low: float
    diff_ci_high: float
    effect_size: float
    es_ci_low: float
    es_ci_high: float
    n_a: int
    n_b: int
    by_levels: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PairwiseComparison({self.level_a} vs {self.level_b}: "
            f"diff={self.mean_diff:.4f}, ES={self.effect_size:.4f})"
        )


@dataclass
class ANOVAResults:
    """Container for complete ANOVA analysis results."""

    anova_table: pd.DataFrame
    model: Any  # statsmodels OLS or MixedLM result
    effect_sizes: dict[str, EffectSizeResult]
    formula: str
    n_observations: int
    is_mixed: bool = False
    standardized_betas: dict[str, float] | None = None

    def __repr__(self) -> str:
        effects = list(self.effect_sizes.keys())
        return (
            f"ANOVAResults(n={self.n_observations}, effects={effects}, "
            f"mixed={self.is_mixed})"
        )


### Helpers

# Constant to not divide by zero
_EPSILON = 1e-12


def _ensure_tuple(value: Any) -> tuple:
    """Convert a value to a tuple for consistent handling."""
    if isinstance(value, tuple):
        return value
    if value == () or value is None:
        return ()
    return (value,)


def _compute_cell_stats(
    df: pd.DataFrame, y_col: str, factor_cols: list[str]
) -> pd.DataFrame:
    """
    Simple summary stats for (n, mean, variance) for factor combinations.

    Parameters
    ----------
    df : DataFrame
        Input data
    y_col : str
        Name of the dependent variable column
    factor_cols : list[str]
        List of factor column names to group by

    Returns
    -------
    DataFrame
        Statistics with columns: factor_cols..., 'n', 'mean', 'var'
    """
    grouped = df.groupby(factor_cols, observed=True, sort=False)[y_col]
    stats = grouped.agg(n="size", mean="mean", var="var").reset_index()
    # Replace NaN variance (n <= 1) with 0 to avoid issues in pooling
    stats["var"] = stats["var"].fillna(0.0)
    return stats


def _compute_pooled_variance(
    cell_stats: pd.DataFrame, grouping_cols: list[str]
) -> tuple[float, pd.DataFrame]:
    """
    Compute pooled variance across groups from cell-level statistics.

    Uses the standard pooled variance formula accounting for unequal sample sizes.

    Parameters
    ----------
    cell_stats : DataFrame
        Cell statistics with 'n', 'mean', 'var' columns
    grouping_cols : list[str]
        Columns defining the groups for pooling

    Returns
    -------
    tuple
        (pooled_variance, aggregated_group_stats_dataframe)
    """
    grouped = cell_stats.groupby(grouping_cols, observed=True, sort=False)

    def aggregate_group(group_df: pd.DataFrame) -> pd.Series:
        n_total = group_df["n"].sum()
        weighted_mean = np.average(group_df["mean"], weights=group_df["n"])

        # Within-group SSE = sum[(n_i-1)*var_i] + sum[n_i*(mean_i - group_mean)^2]
        within_ss = np.sum((group_df["n"] - 1) * group_df["var"])
        between_ss = np.sum(group_df["n"] * (group_df["mean"] - weighted_mean) ** 2)
        sse = within_ss + between_ss

        return pd.Series({"n": n_total, "mean": weighted_mean, "sse": sse})

    aggregated = grouped.apply(aggregate_group).reset_index()

    # Pooled variance = total_sse / (total_n - k)
    total_sse = aggregated["sse"].sum()
    total_n = aggregated["n"].sum()
    k = len(aggregated)

    denominator = max(total_n - k, 1)
    if total_n <= k:
        warnings.warn(
            f"Sample size ({total_n}) <= number of groups ({k}). "
            "Pooled variance estimate may be unreliable."
        )

    pooled_var = total_sse / denominator
    return pooled_var, aggregated.set_index(grouping_cols)


def _compute_cohens_d(mean_diff: float, pooled_sd: float) -> float:
    """
    Compute Cohen's d

    Parameters
    ----------
    mean_diff : float
        Difference between group means
    pooled_sd : float
        Pooled standard deviation

    Returns
    -------
    float
        Cohen's d
    """
    # Cohen's d
    d = mean_diff / (pooled_sd + _EPSILON)

    return d


def _bootstrap_resample_indices(
    strata_indices: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """
    Generate stratified bootstrap resample indices.

    Samples with replacement within each stratum to preserve the factorial
    design structure.

    Parameters
    ----------
    strata_indices : list[np.ndarray]
        List of index arrays, one per stratum
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Concatenated resampled indices
    """
    resampled = [
        stratum[rng.integers(0, len(stratum), size=len(stratum))]
        for stratum in strata_indices
    ]
    return np.concatenate(resampled)


def _single_bootstrap_anova(args: tuple) -> dict[str, float]:
    """
    Perform a single bootstrap replicate for ANOVA effect sizes (for paralelization).


    Parameters
    ----------
    args : tuple
        (df_dict, df_dtypes, formula, typ, strata_cols, seed, effects)

    Returns
    -------
    dict
        Effect name -> partial eta-squared for this replicate
    """
    df_dict, df_dtypes, formula, typ, strata_cols, seed, effects = args

    rng = np.random.default_rng(seed)

    # Reconstruct DataFrame with proper dtypes
    df = pd.DataFrame(df_dict)
    for col, dtype in df_dtypes.items():
        if "float" in dtype or "int" in dtype:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stratified resample - recompute strata indices from column names
    if strata_cols:
        strata_indices = [
            group.index.values
            for _, group in df.groupby(strata_cols, observed=True, sort=False)
        ]
        idx = _bootstrap_resample_indices(strata_indices, rng)
        df_boot = df.iloc[idx].reset_index(drop=True)
    else:
        idx = rng.integers(0, len(df), size=len(df))
        df_boot = df.iloc[idx].reset_index(drop=True)

    try:
        model = smf.ols(formula, data=df_boot).fit(cov_type="HC3")
        aov = anova_lm(model, typ=typ)

        # Find residual row
        if "Residual" in aov.index:
            resid_label = "Residual"
        else:
            resid_label = aov.index[aov["F"].isna()][0]

        ss_resid = float(aov.loc[resid_label, "sum_sq"])

        # Compute partial eta-squared for each effect
        result = {}
        for effect in effects:
            if effect in aov.index:
                ss_effect = float(aov.loc[effect, "sum_sq"])
                total = ss_effect + ss_resid
                result[effect] = ss_effect / total if total > 0 else np.nan
            else:
                result[effect] = np.nan

        return result

    except Exception as e:
        print(f"Bootstrap failed: {type(e).__name__}: {e}")
        return {effect: np.nan for effect in effects}


def _single_bootstrap_pairwise(args: tuple) -> dict[str, tuple[float, float]]:
    """
    Perform a single bootstrap replicate for pairwise comparisons.

    Parameters
    ----------
    args : tuple
        Contains all necessary data for one bootstrap iteration

    Returns
    -------
    dict
        Comparison key -> (mean_diff, effect_size) for this replicate
    """
    (
        df_dict,
        df_dtypes,
        y_col,
        factor,
        by_cols,
        strata_cols,
        seed,
        comparisons,
        all_factors,
    ) = args

    rng = np.random.default_rng(seed)

    # Reconstruct DataFrame with proper dtypes
    df = pd.DataFrame(df_dict)
    for col, dtype in df_dtypes.items():
        if "float" in dtype or "int" in dtype:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stratified resample...
    if strata_cols:
        strata_indices = [
            group.index.values
            for _, group in df.groupby(strata_cols, observed=True, sort=False)
        ]
        idx = _bootstrap_resample_indices(strata_indices, rng)
        df_boot = df.iloc[idx].reset_index(drop=True)
    else:
        idx = rng.integers(0, len(df), size=len(df))
        df_boot = df.iloc[idx].reset_index(drop=True)

    results = {}

    try:
        # marginal means
        mm = _compute_marginal_means(df_boot, y_col, factor, by_cols)
        cell_stats = _compute_cell_stats(df_boot, y_col, all_factors)

        for comp_key, (by_key, level_a, level_b) in comparisons.items():
            try:
                # Get marginal means
                tbl = _slice_marginal_means(mm, factor, by_cols, by_key)

                mean_a = float(tbl.loc[level_a, "mean"])
                mean_b = float(tbl.loc[level_b, "mean"])
                n_a = int(tbl.loc[level_a, "n"])
                n_b = int(tbl.loc[level_b, "n"])

                # Compute pooled SD
                mask = cell_stats[factor].isin([level_a, level_b])
                if by_cols:
                    for col, val in zip(by_cols, _ensure_tuple(by_key)):
                        mask &= cell_stats[col] == val

                subcell = cell_stats.loc[mask]
                pooled_var, _ = _compute_pooled_variance(subcell, [factor])
                pooled_sd = np.sqrt(pooled_var)

                mean_diff = mean_a - mean_b
                effect_size = _compute_cohens_d(mean_diff, pooled_sd)

                results[comp_key] = (mean_diff, effect_size)

            except Exception:
                results[comp_key] = (np.nan, np.nan)

    except Exception:
        for comp_key in comparisons:
            results[comp_key] = (np.nan, np.nan)

    return results


def _compute_marginal_means(
    df: pd.DataFrame, y_col: str, factor: str, by_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Compute weighted marginal means for a factor, optionally within 'by' levels.

    Averages over all other factors using observed cell counts as weights.

    Parameters
    ----------
    df : DataFrame
        Input data
    y_col : str
        Dependent variable column name
    factor : str
        Factor for which to compute marginal means
    by_cols : list[str] or None
        Additional factors to condition on (for interaction analysis)

    Returns
    -------
    DataFrame
        Marginal means indexed by [by_cols..., factor] with 'mean' and 'n' columns
    """
    # Identify all categorical columns
    cat_cols = [
        col
        for col in df.columns
        if col != y_col
        and (df[col].dtype.name == "category" or df[col].dtype == object)
    ]

    # Set up grouping structure
    if by_cols is None:
        by_cols = []

    grouping_factors = [factor] + by_cols
    other_factors = [col for col in cat_cols if col not in grouping_factors]
    all_factors = grouping_factors + other_factors

    # Get cell statistics
    cell_stats = _compute_cell_stats(
        df, y_col, all_factors if all_factors else [factor]
    )

    # Compute weighted marginal means
    grouped = cell_stats.groupby(grouping_factors, observed=True, sort=False)

    marginal = grouped.apply(
        lambda g: pd.Series(
            {"n": g["n"].sum(), "mean": np.average(g["mean"], weights=g["n"])}
        )
    ).reset_index()

    # Set index stuff
    if by_cols:
        marginal = marginal.set_index(by_cols + [factor])
    else:
        marginal = marginal.set_index(factor)

    return marginal


def _slice_marginal_means(
    mm: pd.DataFrame, factor: str, by_cols: list[str] | None, by_key: tuple | Any
) -> pd.DataFrame:
    """
    Extract a slice of marginal means for a specific 'by' combination.

    Parameters
    ----------
    mm : DataFrame
        Marginal means table
    factor : str
        The factor of interest
    by_cols : list[str] or None
        The 'by' columns
    by_key : tuple or scalar
        The specific values for the 'by' columns

    Returns
    -------
    DataFrame
        Sliced table indexed by factor with 'mean' and 'n' columns
    """
    df_mm = mm.reset_index()

    if not by_cols:
        return df_mm[[factor, "mean", "n"]].set_index(factor)

    by_key = _ensure_tuple(by_key)
    mask = np.ones(len(df_mm), dtype=bool)

    for col, val in zip(by_cols, by_key):
        mask &= df_mm[col] == val

    sliced = df_mm.loc[mask, [factor, "mean", "n"]]
    return sliced.set_index(factor)


### helper to make clean dataframes


### Core ANOVA class


class ANOVAModel:
    """
    Comprehensive ANOVA analysis with effect sizes and bootstrapped CIs.

    This class provides a unified interface for:
    - Standard factorial ANOVA with categorical predictors
    - ANCOVA with continuous covariates (standardized betas)
    - Repeated measures / mixed-effects models
    - Bootstrapped confidence intervals for effect sizes
    - Post-hoc pairwise comparisons with Hedges' g

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing all variables
    formula : str
        R / Patsy-style formula (e.g., 'y ~ C(group) * C(condition)')
    id_col : str, optional
        Subject/ID column for repeated measures analysis.
        If provided, a mixed-effects model will be used.
    n_boot : int, default=1000
        Number of bootstrap replicates for confidence intervals
    ci_level : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI)
    anova_type : int, default=2
        Type of sums of squares (1, 2, or 3)
    strata_cols : list[str], optional
        Columns for stratified bootstrapping. If None, auto-detected
        from categorical predictors.
    random_state : int, optional
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs for bootstrapping.

    Attributes
    ----------
    results_ : ANOVAResults
        Fitted results (available after calling fit())
    is_fitted_ : bool
        Whether the model has been fitted

    """

    def __init__(
        self,
        df: pd.DataFrame,
        formula: str,
        id_col: str | None = None,
        n_boot: int = 1000,
        ci_level: float = 0.95,
        anova_type: int = 3,
        strata_cols: list[str] | None = None,
        random_state: int | None = None,
        n_jobs: int = -1,
    ):
        # Check inputs are valid...
        if not 0 < ci_level < 1:
            raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
        if anova_type not in (1, 2, 3):
            raise ValueError(f"anova_type must be 1, 2, or 3, got {anova_type}")
        if n_boot < 1:
            raise ValueError(f"n_boot must be >= 1, got {n_boot}")

        self.df = df.copy()
        self.formula = formula
        self.id_col = id_col
        self.n_boot = n_boot
        self.ci_level = ci_level
        self.anova_type = anova_type
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Parse formula to extract dependent variable
        self._y_col = formula.split("~")[0].strip()

        # Auto-detect strata columns if not provided
        if strata_cols is None:
            self._strata_cols = [
                col
                for col in df.columns
                if col != self._y_col
                and col != id_col
                and (df[col].dtype.name == "category" or df[col].dtype == object)
            ]
        else:
            self._strata_cols = strata_cols

        # Identify continuous predictors for standardized betas
        self._continuous_cols = self._identify_continuous_predictors()

        # State
        self.is_fitted_ = False
        self.results_: ANOVAResults | None = None
        self._model = None
        self._anova_table = None

    def _identify_continuous_predictors(self) -> list[str]:
        """Identify continuous (non-categorical) predictors from the formula."""
        # Simple parsing: find terms not wrapped in C()

        rhs = self.formula.split("~")[1] if "~" in self.formula else ""

        # Find all variable references
        # This is a simplified parser - handles common cases
        terms = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", rhs)

        # Exclude categorical markers and interactions
        categorical_markers = {"C", "c"}
        continuous = []

        for term in terms:
            if term in categorical_markers:
                continue
            if term in self.df.columns:
                col = self.df[term]
                if col.dtype in ["int64", "float64", "int32", "float32"]:
                    # Check if it's actually used as continuous (not wrapped in C())
                    if (
                        f"C({term})" not in self.formula
                        and f"c({term})" not in self.formula
                    ):
                        continuous.append(term)

        return list(set(continuous))

    def _precompute_strata_indices(self) -> list[np.ndarray]:
        """Precompute index arrays for each stratum for efficient bootstrapping."""
        if not self._strata_cols:
            return []

        grouped = self.df.groupby(self._strata_cols, observed=True, sort=False)
        return [group.index.values for _, group in grouped]

    def _fit_standard_ols(self) -> tuple[Any, pd.DataFrame]:
        """Fit standard OLS model and compute ANOVA table."""
        model = smf.ols(self.formula, data=self.df).fit(cov_type="HC3")
        anova_table = anova_lm(model, typ=self.anova_type)
        return model, anova_table

    def _fit_mixed_model(self) -> tuple[Any, pd.DataFrame]:
        """
        Fit mixed-effects model for repeated measures.

        Uses random intercepts for subjects.
        """
        # Construct mixed model formula with random intercept
        mixed_formula = self.formula

        model = smf.mixedlm(
            mixed_formula, data=self.df, groups=self.df[self.id_col]
        ).fit(reml=True)

        # For mixed models, we need to construct an ANOVA-like table (Type III)
        anova_table = self._construct_mixed_anova_table(model)

        return model, anova_table

    def _construct_mixed_anova_table(self, model) -> pd.DataFrame:
        """
        Construct an ANOVA-like summary table from mixed model results.

        """
        # Get the design matrix and response
        y = model.model.endog
        n = len(y)

        # Get fitted values and compute total SS
        fitted = model.fittedvalues
        resid = y - fitted
        ss_resid = np.sum(resid**2)
        ss_total = np.sum((y - np.mean(y)) ** 2)

        # For mixed models - wald for fixed + aprox SS
        fe_params = model.fe_params

        # fixed effect cov
        try:
            vcov = model.cov_params()
        except:
            vcov = None

        # get effects from formula
        effects_data = []

        # Group parameters by effect (handle categorical dummy coding)
        effect_groups = {}
        for param_name in fe_params.index:
            if param_name == "Intercept":
                continue

            # Extract the base effect name (e.g., "C(time)" from "C(time)[T.post]")
            if "[" in param_name:
                base_effect = param_name.split("[")[0]
                # Handle interactions
                if ":" in base_effect:
                    base_effect = param_name.rsplit("[", 1)[
                        0
                    ]  # Keep interaction notation
            else:
                base_effect = param_name

            if base_effect not in effect_groups:
                effect_groups[base_effect] = []
            effect_groups[base_effect].append(param_name)

        # Compute Wald test and approximate SS for each effect
        for effect_name, param_names in effect_groups.items():
            params = fe_params[param_names].values

            if vcov is not None:
                # Get the submatrix of vcov for these parameters
                try:
                    param_indices = [
                        list(fe_params.index).index(p) for p in param_names
                    ]
                    sub_vcov = vcov.iloc[param_indices, param_indices].values

                    # Wald chi-square statistic
                    wald_chi2 = params @ np.linalg.solve(sub_vcov, params)
                    df_effect = len(param_names)
                    f_stat = wald_chi2 / df_effect

                    # Approximate p-value (using F distribution with residual df)
                    df_resid = n - len(fe_params)
                    p_value = 1 - stats.f.cdf(f_stat, df_effect, df_resid)

                    # Approximate SS for the effect
                    # SS_effect ≈ MSE * df_effect * F
                    mse = ss_resid / df_resid
                    ss_effect = mse * df_effect * f_stat

                except Exception:
                    f_stat = np.nan
                    p_value = np.nan
                    ss_effect = np.nan
                    df_effect = len(param_names)
            else:
                f_stat = np.nan
                p_value = np.nan
                ss_effect = np.nan
                df_effect = len(param_names)

            effects_data.append(
                {
                    "effect": effect_name,
                    "sum_sq": ss_effect,
                    "df": float(df_effect),
                    "F": f_stat,
                    "PR(>F)": p_value,
                }
            )

        # Create DataFrame
        if effects_data:
            anova_table = pd.DataFrame(effects_data).set_index("effect")
        else:
            anova_table = pd.DataFrame(columns=["sum_sq", "df", "F", "PR(>F)"])

        # Add residual row
        df_resid = n - len(fe_params)
        resid_row = pd.DataFrame(
            {
                "sum_sq": [ss_resid],
                "df": [float(df_resid)],
                "F": [np.nan],
                "PR(>F)": [np.nan],
            },
            index=["Residual"],
        )
        anova_table = pd.concat([anova_table, resid_row])

        return anova_table

    def _compute_partial_eta_squared(
        self, anova_table: pd.DataFrame
    ) -> dict[str, float]:
        """
        Compute partial eta-squared for all effects.

        partial η² = SS_effect / (SS_effect + SS_error)
        """
        # Find residual row
        if "Residual" in anova_table.index:
            resid_label = "Residual"
        else:
            # Find row with NaN F value
            resid_mask = anova_table["F"].isna()
            if resid_mask.any():
                resid_label = anova_table.index[resid_mask][0]
            else:
                raise ValueError("Could not identify residual row in ANOVA table")

        ss_resid = float(anova_table.loc[resid_label, "sum_sq"])

        # Compute for each effect
        eta_squared = {}
        for effect in anova_table.index:
            if effect == resid_label:
                continue

            ss_effect = float(anova_table.loc[effect, "sum_sq"])
            total = ss_effect + ss_resid
            eta_squared[effect] = ss_effect / total if total > 0 else np.nan

        return eta_squared

    def _compute_standardized_betas(self, model) -> dict[str, float]:
        """
        Compute standardized regression coefficients for continuous predictors.

        β_standardized = β * (SD_x / SD_y)
        """
        if not self._continuous_cols:
            return {}

        standardized = {}
        sd_y = self.df[self._y_col].std()

        for col in self._continuous_cols:
            if col in model.params.index:
                beta = model.params[col]
                sd_x = self.df[col].std()
                standardized[col] = beta * (sd_x / sd_y) if sd_y > 0 else np.nan

        return standardized

    def _compute_mixed_effect_sizes(
        self, effects: list[str]
    ) -> dict[str, EffectSizeResult]:
        """
        Compute partial eta-squared for mixed model effects from ANOVA table.
        """
        effect_sizes = {}

        # Get residual SS
        ss_resid = float(self._anova_table.loc["Residual", "sum_sq"])

        for effect in effects:
            if effect in self._anova_table.index:
                ss_effect = self._anova_table.loc[effect, "sum_sq"]
                if pd.notna(ss_effect):
                    ss_effect = float(ss_effect)
                    total = ss_effect + ss_resid
                    eta2 = ss_effect / total if total > 0 else np.nan
                else:
                    eta2 = np.nan
            else:
                eta2 = np.nan

            effect_sizes[effect] = EffectSizeResult(
                value=eta2, ci_low=np.nan, ci_high=np.nan, ci_level=self.ci_level
            )

        return effect_sizes

    def _bootstrap_mixed_effect_sizes(
        self, effect_sizes: dict[str, EffectSizeResult], effects: list[str]
    ):
        """
        Add bootstrapped CIs to mixed model effect sizes.

        Uses subject-level resampling (resample entire subjects with all their observations).
        """
        rng = np.random.default_rng(self.random_state)

        # Get unique subject IDs
        subject_ids = self.df[self.id_col].unique()
        n_subjects = len(subject_ids)

        # Storage for bootstrap values
        boot_results = {effect: [] for effect in effects}

        for _ in range(self.n_boot):
            # Resample subjects (with replacement)
            resampled_subjects = rng.choice(subject_ids, size=n_subjects, replace=True)

            # Build resampled dataframe (all observations for each resampled subject)
            dfs = []
            for i, subj in enumerate(resampled_subjects):
                subj_data = self.df[self.df[self.id_col] == subj].copy()
                # Assign new subject ID to handle duplicates
                subj_data[self.id_col] = i
                dfs.append(subj_data)

            df_boot = pd.concat(dfs, ignore_index=True)

            try:
                # Fit mixed model on bootstrap sample
                model_boot = smf.mixedlm(
                    self.formula, data=df_boot, groups=df_boot[self.id_col]
                ).fit(reml=True, disp=False)

                # Compute effect sizes for this bootstrap
                y = model_boot.model.endog
                fitted = model_boot.fittedvalues
                resid = y - fitted
                ss_resid_boot = np.sum(resid**2)
                n = len(y)

                fe_params = model_boot.fe_params
                vcov = model_boot.cov_params()

                # Group parameters by effect
                effect_groups = {}
                for param_name in fe_params.index:
                    if param_name == "Intercept":
                        continue
                    if "[" in param_name:
                        base_effect = param_name.split("[")[0]
                        if ":" in base_effect:
                            base_effect = param_name.rsplit("[", 1)[0]
                    else:
                        base_effect = param_name

                    if base_effect not in effect_groups:
                        effect_groups[base_effect] = []
                    effect_groups[base_effect].append(param_name)

                # Compute eta-squared for each effect
                df_resid = n - len(fe_params)
                mse = ss_resid_boot / df_resid if df_resid > 0 else np.nan

                for effect in effects:
                    if effect in effect_groups:
                        param_names = effect_groups[effect]
                        params = fe_params[param_names].values

                        try:
                            param_indices = [
                                list(fe_params.index).index(p) for p in param_names
                            ]
                            sub_vcov = vcov.iloc[param_indices, param_indices].values

                            wald_chi2 = params @ np.linalg.solve(sub_vcov, params)
                            df_effect = len(param_names)
                            f_stat = wald_chi2 / df_effect

                            ss_effect = mse * df_effect * f_stat
                            eta2 = (
                                ss_effect / (ss_effect + ss_resid_boot)
                                if (ss_effect + ss_resid_boot) > 0
                                else np.nan
                            )

                            boot_results[effect].append(eta2)
                        except Exception:
                            boot_results[effect].append(np.nan)
                    else:
                        boot_results[effect].append(np.nan)

            except Exception:
                # If model fitting fails, append NaN for all effects
                for effect in effects:
                    boot_results[effect].append(np.nan)

        # Compute CIs and update effect_sizes
        alpha = (1 - self.ci_level) / 2

        for effect in effects:
            values = np.array(boot_results[effect])
            values = values[~np.isnan(values)]

            if len(values) > 0:
                ci_low, ci_high = np.quantile(values, [alpha, 1 - alpha])
                effect_sizes[effect] = EffectSizeResult(
                    value=effect_sizes[effect].value,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    ci_level=self.ci_level,
                    boot_distribution=values,
                )
            else:
                warnings.warn(
                    f"All bootstrap replicates failed for mixed model effect '{effect}'. "
                    "CI cannot be computed."
                )

    def _bootstrap_effect_sizes_parallel(
        self, effects: list[str]
    ) -> dict[str, EffectSizeResult]:
        """
        Bootstrap confidence intervals for partial eta-squared
        """
        # Point estimates
        point_estimates = self._compute_partial_eta_squared(self._anova_table)

        if self.n_boot <= 0:
            return {
                effect: EffectSizeResult(
                    value=point_estimates.get(effect, np.nan),
                    ci_low=np.nan,
                    ci_high=np.nan,
                    ci_level=self.ci_level,
                )
                for effect in effects
            }

        # Prepare data for parallel processing
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=self.n_boot)

        df_dict = self.df.to_dict("list")
        df_dtypes = {col: str(self.df[col].dtype) for col in self.df.columns}

        args_list = [
            (
                df_dict,
                df_dtypes,
                self.formula,
                self.anova_type,
                self._strata_cols,
                seed,
                effects,
            )
            for seed in seeds
        ]

        # Initialize storage with explicit effect keys
        boot_results = {effect: [] for effect in effects}

        n_workers = self.n_jobs if self.n_jobs > 0 else None

        if self.n_boot >= 100 and n_workers != 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_single_bootstrap_anova, args) for args in args_list
                ]

                for future in as_completed(futures):
                    result = future.result()
                    # Explicitly match by effect name
                    for effect in effects:
                        if effect in result:
                            boot_results[effect].append(result[effect])
                        else:
                            boot_results[effect].append(np.nan)
        else:
            for args in args_list:
                result = _single_bootstrap_anova(args)
                for effect in effects:
                    if effect in result:
                        boot_results[effect].append(result[effect])
                    else:
                        boot_results[effect].append(np.nan)

        # Compute confidence intervals
        alpha = (1 - self.ci_level) / 2
        effect_size_results = {}

        for effect in effects:
            # Get point estimate for THIS effect
            point_est = point_estimates.get(effect, np.nan)

            values = np.array(boot_results[effect])
            values = values[~np.isnan(values)]

            if len(values) == 0:
                warnings.warn(
                    f"All bootstrap replicates failed for effect '{effect}'. "
                    "CI cannot be computed."
                )
                ci_low, ci_high = np.nan, np.nan
            else:
                ci_low, ci_high = np.quantile(values, [alpha, 1 - alpha])

                # Sanity check: warn if point estimate outside CI
                if point_est < ci_low or point_est > ci_high:
                    warnings.warn(
                        f"Point estimate ({point_est:.4f}) for '{effect}' is outside "
                        f"bootstrap CI [{ci_low:.4f}, {ci_high:.4f}]. This may indicate "
                        "an issue with the bootstrap or high variability."
                    )

            effect_size_results[effect] = EffectSizeResult(
                value=point_est,
                ci_low=ci_low,
                ci_high=ci_high,
                ci_level=self.ci_level,
                boot_distribution=values if len(values) > 0 else None,
            )

        return effect_size_results

    def fit(self, compute_ci: bool = True) -> "ANOVAModel":
        """
        Fit the ANOVA model and compute effect sizes.

        Parameters
        ----------
        compute_ci : bool, default=True
            Whether to compute bootstrapped confidence intervals.
            Set to False for faster fitting without CIs.

        Returns
        -------
        self
            The fitted model instance
        """
        # Determine model type and fit
        if self.id_col is not None:
            self._model, self._anova_table = self._fit_mixed_model()
            is_mixed = True
        else:
            self._model, self._anova_table = self._fit_standard_ols()
            is_mixed = False

        # Get effect names (excluding residual)
        if "Residual" in self._anova_table.index:
            effects = [idx for idx in self._anova_table.index if idx != "Residual"]
        else:
            effects = [
                idx
                for idx in self._anova_table.index
                if pd.notna(self._anova_table.loc[idx, "F"])
            ]

        # Compute effect sizes with CIs
        if compute_ci and not is_mixed:
            effect_sizes = self._bootstrap_effect_sizes_parallel(effects)
        elif is_mixed:
            # For mixed models, compute effect sizes from the ANOVA table
            effect_sizes = self._compute_mixed_effect_sizes(effects)

            # Bootstrap CIs for mixed models if requested
            if compute_ci and self.n_boot > 0:
                self._bootstrap_mixed_effect_sizes(effect_sizes, effects)
        else:
            # Point estimates only
            point_estimates = self._compute_partial_eta_squared(self._anova_table)
            effect_sizes = {
                effect: EffectSizeResult(
                    value=point_estimates.get(effect, np.nan),
                    ci_low=np.nan,
                    ci_high=np.nan,
                    ci_level=self.ci_level,
                )
                for effect in effects
            }

        # Compute standardized betas for continuous predictors
        standardized_betas = None
        if self._continuous_cols and not is_mixed:
            standardized_betas = self._compute_standardized_betas(self._model)

        # Store results
        self.results_ = ANOVAResults(
            anova_table=self._anova_table.copy(),
            model=self._model,
            effect_sizes=effect_sizes,
            formula=self.formula,
            n_observations=len(self.df),
            is_mixed=is_mixed,
            standardized_betas=standardized_betas,
        )

        self.is_fitted_ = True
        return self

    def summary(
        self, show_ci: bool = True, show_betas: bool = True, precision: int = 4
    ) -> pd.DataFrame:
        """
        Display ANOVA summary table.

        Parameters
        ----------
        show_ci : bool, default=True
            Include confidence intervals for effect sizes
        show_betas : bool, default=True
            Include standardized betas for continuous predictors
        precision : int, default=4
            Number of decimal places for numeric values

        Returns
        -------
        pd.DataFrame
            Summary table with ANOVA results and effect sizes
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Start with ANOVA table
        summary_df = self.results_.anova_table.copy()

        # Add partial eta-squared
        eta_sq = pd.Series(
            {k: v.value for k, v in self.results_.effect_sizes.items()},
            name="partial_eta2",
        )
        summary_df["partial_eta2"] = eta_sq

        # Add confidence intervals
        if show_ci:
            ci_low = pd.Series(
                {k: v.ci_low for k, v in self.results_.effect_sizes.items()},
                name="eta2_ci_low",
            )
            ci_high = pd.Series(
                {k: v.ci_high for k, v in self.results_.effect_sizes.items()},
                name="eta2_ci_high",
            )
            summary_df["eta2_ci_low"] = ci_low
            summary_df["eta2_ci_high"] = ci_high

        # Print summary
        print("=" * 70)
        print("ANOVA Summary")
        print("=" * 70)
        print(f"Formula: {self.formula}")
        print(f"N observations: {self.results_.n_observations}")
        print(
            f"Model type: {'Mixed (repeated measures)' if self.results_.is_mixed else 'Standard OLS'}"
        )
        print("-" * 70)
        print()

        # Format and display table
        display_df = summary_df.round(precision)
        print(display_df.to_string())
        print()

        # Show standardized betas if applicable
        if show_betas and self.results_.standardized_betas:
            print("-" * 70)
            print("Standardized Betas (continuous predictors):")
            for var, beta in self.results_.standardized_betas.items():
                print(f"  {var}: {beta:.{precision}f}")
            print()

        print("=" * 70)

        return summary_df

    def pairwise_posthoc(
        self,
        factor: str,
        by: str | list[str] | None = None,
        comparisons: list[tuple[str, str]] | None = None,
        effect_size: Literal["hedges_g", "cohens_d"] = "cohens_d",
        es_report: Literal["signed", "absolute", "both"] = "signed",
        n_boot: int | None = None,
        n_jobs: int | None = None,
    ) -> pd.DataFrame:
        """
        Perform post-hoc pairwise comparisons for a factor.

        Computes mean differences and effect sizes (Hedges' g or Cohen's d)
        with bootstrapped confidence intervals.

        Parameters
        ----------
        factor : str
            The factor for which to compute pairwise comparisons
        by : str or list[str], optional
            Conditioning variable(s) for simple effects analysis.
            If provided, comparisons are made within each level of 'by'.
        comparisons : list[tuple[str, str]], optional
            Specific pairs to compare. If None, all pairs are compared.
        effect_size : {'hedges_g', 'cohens_d'}, default='hedges_g'
            Type of standardized effect size to compute
        es_report : {'signed', 'absolute', 'both'}, default='absolute'
            How to report effect sizes:
            - 'signed': preserves direction (can be negative)
            - 'absolute': reports |ES|
            - 'both': includes both signed and absolute
        n_boot : int, optional
            Number of bootstrap replicates. Defaults to model's n_boot.
        n_jobs : int, optional
            Number of parallel jobs. Defaults to model's n_jobs.

        Returns
        -------
        pd.DataFrame
            Table of pairwise comparisons with columns:
            - by columns (if applicable)
            - level_a, level_b: compared levels
            - diff, ci_low_diff, ci_high_diff: mean difference with CI
            - es, ci_low_es, ci_high_es: effect size with CI
            - n_a, n_b: sample sizes
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        n_boot = n_boot if n_boot is not None else self.n_boot
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs

        # Normalize 'by' to list
        by_cols = None
        if by is not None:
            by_cols = by if isinstance(by, list) else [by]

        # Compute marginal means
        mm = _compute_marginal_means(self.df, self._y_col, factor, by_cols)

        # Identify all categorical columns for cell stats
        all_factors = [
            col
            for col in self.df.columns
            if col != self._y_col
            and col != self.id_col
            and (self.df[col].dtype.name == "category" or self.df[col].dtype == object)
        ]

        cell_stats = _compute_cell_stats(self.df, self._y_col, all_factors)

        # Build comparison list
        comparison_dict = self._build_comparison_dict(mm, factor, by_cols, comparisons)

        # Compute point estimates
        results = self._compute_pairwise_point_estimates(
            mm, cell_stats, factor, by_cols, comparison_dict, effect_size
        )

        # Bootstrap confidence intervals
        if n_boot > 0:
            self._bootstrap_pairwise_ci(
                results, factor, by_cols, comparison_dict, all_factors, n_boot, n_jobs
            )

        # Format output DataFrame
        return self._format_pairwise_results(results, by_cols, es_report)

    def _build_comparison_dict(
        self,
        mm: pd.DataFrame,
        factor: str,
        by_cols: list[str] | None,
        comparisons: list[tuple[str, str]] | None,
    ) -> dict[str, tuple]:
        """Build dictionary of comparisons to make."""

        comparison_dict = {}

        if isinstance(mm.index, pd.MultiIndex):
            by_names = [n for n in mm.index.names if n != factor]

            for by_vals, sub in mm.groupby(level=by_names, observed=True, sort=False):
                by_key = _ensure_tuple(by_vals)
                tbl = _slice_marginal_means(sub, factor, by_names, by_key)
                levels = list(tbl.index.unique())

                pairs = (
                    comparisons
                    if comparisons
                    else list(itertools.combinations(levels, 2))
                )
                for a, b in pairs:
                    key = f"{by_key}_{a}_{b}"
                    comparison_dict[key] = (by_key, a, b)
        else:
            levels = list(mm.index.unique())
            pairs = (
                comparisons if comparisons else list(itertools.combinations(levels, 2))
            )
            for a, b in pairs:
                key = f"()_{a}_{b}"
                comparison_dict[key] = ((), a, b)

        return comparison_dict

    def _compute_pairwise_point_estimates(
        self,
        mm: pd.DataFrame,
        cell_stats: pd.DataFrame,
        factor: str,
        by_cols: list[str] | None,
        comparison_dict: dict,
        effect_size: str,
    ) -> dict[str, dict]:
        """Compute point estimates for all pairwise comparisons."""
        results = {}

        for comp_key, (by_key, level_a, level_b) in comparison_dict.items():
            # Get marginal means
            tbl = _slice_marginal_means(mm, factor, by_cols, by_key)

            mean_a = float(tbl.loc[level_a, "mean"])
            mean_b = float(tbl.loc[level_b, "mean"])
            n_a = int(tbl.loc[level_a, "n"])
            n_b = int(tbl.loc[level_b, "n"])

            # Compute pooled SD
            mask = cell_stats[factor].isin([level_a, level_b])
            if by_cols:
                for col, val in zip(by_cols, _ensure_tuple(by_key)):
                    mask &= cell_stats[col] == val

            subcell = cell_stats.loc[mask]
            pooled_var, _ = _compute_pooled_variance(subcell, [factor])
            pooled_sd = np.sqrt(pooled_var)

            # Compute effect size
            mean_diff = mean_a - mean_b
            if effect_size == "hedges_g":
                es = _compute_cohens_d(mean_diff, pooled_sd)
            else:  # Cohen's d
                es = mean_diff / (pooled_sd + _EPSILON)

            results[comp_key] = {
                "by_key": by_key,
                "level_a": level_a,
                "level_b": level_b,
                "diff": mean_diff,
                "es": es,
                "n_a": n_a,
                "n_b": n_b,
                "boot_diff": [],
                "boot_es": [],
            }

        return results

    def _bootstrap_pairwise_ci(
        self,
        results: dict,
        factor: str,
        by_cols: list[str] | None,
        comparison_dict: dict,
        all_factors: list[str],
        n_boot: int,
        n_jobs: int,
    ):
        """Add bootstrapped CIs to pairwise comparison results."""
        # Use dict representation to preserve dtypes across processes
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=n_boot)

        # Prepare data
        df_dict = self.df.to_dict("list")
        df_dtypes = {col: str(self.df[col].dtype) for col in self.df.columns}

        args_list = [
            (
                df_dict,
                df_dtypes,
                self._y_col,
                factor,
                by_cols,
                self._strata_cols,
                seed,
                comparison_dict,
                all_factors,
            )
            for seed in seeds
        ]

        # Run bootstrap
        n_workers = n_jobs if n_jobs > 0 else None

        if n_boot >= 50 and n_workers != 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_single_bootstrap_pairwise, args)
                    for args in args_list
                ]

                for future in as_completed(futures):
                    boot_result = future.result()
                    for comp_key, (diff, es) in boot_result.items():
                        results[comp_key]["boot_diff"].append(diff)
                        results[comp_key]["boot_es"].append(es)
        else:
            for args in args_list:
                boot_result = _single_bootstrap_pairwise(args)
                for comp_key, (diff, es) in boot_result.items():
                    results[comp_key]["boot_diff"].append(diff)
                    results[comp_key]["boot_es"].append(es)

        # Compute CIs
        alpha = (1 - self.ci_level) / 2

        for comp_key, res in results.items():
            diff_vals = np.array(res["boot_diff"])
            diff_vals = diff_vals[~np.isnan(diff_vals)]

            es_vals = np.array(res["boot_es"])
            es_vals = es_vals[~np.isnan(es_vals)]

            if len(diff_vals) > 0:
                res["ci_low_diff"], res["ci_high_diff"] = np.quantile(
                    diff_vals, [alpha, 1 - alpha]
                )
            else:
                res["ci_low_diff"] = res["ci_high_diff"] = np.nan

            if len(es_vals) > 0:
                # Signed CI
                res["ci_low_es_signed"], res["ci_high_es_signed"] = np.quantile(
                    es_vals, [alpha, 1 - alpha]
                )
                # Absolute CI
                es_abs = np.abs(es_vals)
                res["ci_low_es_abs"], res["ci_high_es_abs"] = np.quantile(
                    es_abs, [alpha, 1 - alpha]
                )
            else:
                res["ci_low_es_signed"] = res["ci_high_es_signed"] = np.nan
                res["ci_low_es_abs"] = res["ci_high_es_abs"] = np.nan

    def _format_pairwise_results(
        self, results: dict, by_cols: list[str] | None, es_report: str
    ) -> pd.DataFrame:
        """Format pairwise results into a DataFrame."""
        rows = []

        for comp_key, res in results.items():
            row = {
                "level_a": res["level_a"],
                "level_b": res["level_b"],
                "diff": res["diff"],
                "ci_low_diff": res.get("ci_low_diff", np.nan),
                "ci_high_diff": res.get("ci_high_diff", np.nan),
                "n_a": res["n_a"],
                "n_b": res["n_b"],
            }

            # Add 'by' columns
            if by_cols:
                for i, col in enumerate(by_cols):
                    row[col] = res["by_key"][i] if i < len(res["by_key"]) else None

            # Add effect size columns based on report type
            if es_report == "signed":
                row["es"] = res["es"]
                row["ci_low_es"] = res.get("ci_low_es_signed", np.nan)
                row["ci_high_es"] = res.get("ci_high_es_signed", np.nan)
            elif es_report == "absolute":
                row["es"] = abs(res["es"])
                row["ci_low_es"] = res.get("ci_low_es_abs", np.nan)
                row["ci_high_es"] = res.get("ci_high_es_abs", np.nan)
            else:  # both
                row["es_signed"] = res["es"]
                row["ci_low_es_signed"] = res.get("ci_low_es_signed", np.nan)
                row["ci_high_es_signed"] = res.get("ci_high_es_signed", np.nan)
                row["es_abs"] = abs(res["es"])
                row["ci_low_es_abs"] = res.get("ci_low_es_abs", np.nan)
                row["ci_high_es_abs"] = res.get("ci_high_es_abs", np.nan)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Reorder columns
        col_order = []
        if by_cols:
            col_order.extend(by_cols)
        col_order.extend(["level_a", "level_b"])

        if es_report == "both":
            col_order.extend(
                [
                    "es_signed",
                    "ci_low_es_signed",
                    "ci_high_es_signed",
                    "es_abs",
                    "ci_low_es_abs",
                    "ci_high_es_abs",
                ]
            )
        else:
            col_order.extend(["es", "ci_low_es", "ci_high_es"])

        col_order.extend(["diff", "ci_low_diff", "ci_high_diff", "n_a", "n_b"])

        # Only include columns that exist
        col_order = [c for c in col_order if c in df.columns]

        return (
            df[col_order]
            .sort_values(
                by_cols[:2] if by_cols and len(by_cols) >= 2 else ["level_a", "level_b"]
            )
            .reset_index(drop=True)
        )

    def simple_slopes(
        self,
        continuous_var: str,
        categorical_vars: str | list[str],
        ci_level: float | None = None,
    ) -> pd.DataFrame:
        """
        Compute simple slopes for a continuous variable at each level of categorical variable(s).

        This is useful for interpreting interactions between continuous and categorical
        predictors. Reports the effect of the continuous variable separately within
        each combination of categorical factor levels.

        Parameters
        ----------
        continuous_var : str
            Name of the continuous predictor variable
        categorical_vars : str or list[str]
            Categorical variable(s) to condition on
        ci_level : float, optional
            Confidence level for intervals. Defaults to model's ci_level.

        Returns
        -------
        pd.DataFrame
            Table with columns for categorical levels, slope estimate, SE, CI, t, and p-value

        Example
        -------
        >>> model = ANOVAModel(df, 'DV ~ C(Type) * C(Subtype) * IV')
        >>> model.fit()
        >>> model.simple_slopes('IV', ['Type', 'Subtype'])
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if self.results_.is_mixed:
            raise NotImplementedError(
                "Simple slopes for mixed models not yet implemented."
            )

        if ci_level is None:
            ci_level = self.ci_level

        # Normalize categorical_vars to list
        if isinstance(categorical_vars, str):
            categorical_vars = [categorical_vars]

        # Get unique combinations of categorical levels
        level_combinations = (
            self.df.groupby(categorical_vars, observed=True).size().index
        )

        results = []

        for levels in level_combinations:
            levels = levels if isinstance(levels, tuple) else (levels,)

            # Subset data for this combination
            mask = pd.Series(True, index=self.df.index)
            for var, level in zip(categorical_vars, levels):
                mask &= self.df[var] == level

            subset = self.df[mask]

            if len(subset) < 3:
                continue

            # Fit simple regression of DV on continuous_var for this subset
            X = sm.add_constant(subset[continuous_var])
            y = subset[self._y_col]

            try:
                simple_model = sm.OLS(y, X).fit(cov_type="HC3")

                slope = simple_model.params[continuous_var]
                se = simple_model.bse[continuous_var]
                t_stat = simple_model.tvalues[continuous_var]
                p_value = simple_model.pvalues[continuous_var]

                # Confidence interval
                alpha = 1 - ci_level
                t_crit = stats.t.ppf(1 - alpha / 2, simple_model.df_resid)
                ci_low = slope - t_crit * se
                ci_high = slope + t_crit * se

                # Standardized slope (within this subset)
                sd_x = subset[continuous_var].std()
                sd_y = subset[self._y_col].std()
                slope_standardized = slope * (sd_x / sd_y) if sd_y > 0 else np.nan

                row = {
                    "slope": slope,
                    "slope_std": slope_standardized,
                    "se": se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "t": t_stat,
                    "p": p_value,
                    "n": len(subset),
                }

                # Add categorical level columns
                for var, level in zip(categorical_vars, levels):
                    row[var] = level

                results.append(row)

            except Exception:
                continue

        # Build output DataFrame
        df_results = pd.DataFrame(results)

        # Reorder columns
        col_order = categorical_vars + [
            "slope",
            "slope_std",
            "se",
            "ci_low",
            "ci_high",
            "t",
            "p",
            "n",
        ]
        df_results = df_results[col_order]

        return df_results.sort_values(categorical_vars).reset_index(drop=True)

    def simple_slopes_pairwise(
        self,
        continuous_var: str,
        categorical_vars: str | list[str],
        ci_level: float | None = None,
        p_adjust: str | None = "bonferroni",
        n_boot: int | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute pairwise differences in simple slopes with bootstrapped confidence intervals.

        Compares the effect of a continuous variable across all pairs of
        categorical factor level combinations.

        Parameters
        ----------
        continuous_var : str
            Name of the continuous predictor variable
        categorical_vars : str or list[str]
            Categorical variable(s) to condition on
        ci_level : float, optional
            Confidence level for intervals. Defaults to model's ci_level.
        p_adjust : str or None, default='bonferroni'
            Method for p-value adjustment. Options: 'bonferroni', 'holm', 'fdr_bh', None
        n_boot : int, optional
            Number of bootstrap replicates. Defaults to model's n_boot.
        random_state : int, optional
            Random seed for reproducibility. Defaults to model's random_state.

        Returns
        -------
        pd.DataFrame
            Table with pairwise slope differences, bootstrapped CIs, and p-values

        Example
        -------
        >>> model = ANOVAModel(df, 'DV ~ C(Type) * C(Subtype) * IV')
        >>> model.fit()
        >>> model.simple_slopes_pairwise('IV', 'Type', n_boot=1000)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if ci_level is None:
            ci_level = self.ci_level
        if n_boot is None:
            n_boot = self.n_boot
        if random_state is None:
            random_state = self.random_state

        # Normalize categorical_vars to list
        if isinstance(categorical_vars, str):
            categorical_vars = [categorical_vars]

        rng = np.random.default_rng(random_state)

        # Get unique combinations of categorical levels
        level_combinations = list(
            self.df.groupby(categorical_vars, observed=True).size().index
        )

        # Normalize to tuples
        level_combinations = [
            lvl if isinstance(lvl, tuple) else (lvl,) for lvl in level_combinations
        ]

        # Create subsets for each combination
        subsets = {}
        for levels_tuple in level_combinations:
            mask = pd.Series(True, index=self.df.index)
            for var, level in zip(categorical_vars, levels_tuple):
                mask &= self.df[var] == level

            subset = self.df[mask].copy()
            if len(subset) >= 3:
                subsets[levels_tuple] = subset

        # Function to compute slopes for all groups from a (possibly resampled) dataset
        def compute_all_slopes(data_dict):
            slopes = {}
            for levels_tuple, subset in data_dict.items():
                try:
                    X = sm.add_constant(subset[continuous_var])
                    y = subset[self._y_col]
                    model = sm.OLS(y, X).fit(cov_type="HC3")
                    slopes[levels_tuple] = model.params[continuous_var]
                except Exception:
                    slopes[levels_tuple] = np.nan
            return slopes

        # Compute point estimates
        point_slopes = compute_all_slopes(subsets)

        # Generate all pairs
        pairs = list(itertools.combinations(subsets.keys(), 2))

        # Compute point estimate differences
        point_diffs = {}
        for levels_a, levels_b in pairs:
            if levels_a in point_slopes and levels_b in point_slopes:
                point_diffs[(levels_a, levels_b)] = (
                    point_slopes[levels_a] - point_slopes[levels_b]
                )

        # Bootstrap
        boot_diffs = {pair: [] for pair in pairs}

        for _ in range(n_boot):
            # Resample within each subset (stratified bootstrap)
            resampled_subsets = {}
            for levels_tuple, subset in subsets.items():
                idx = rng.choice(len(subset), size=len(subset), replace=True)
                resampled_subsets[levels_tuple] = subset.iloc[idx]

            # Compute slopes on resampled data
            boot_slopes = compute_all_slopes(resampled_subsets)

            # Compute differences
            for levels_a, levels_b in pairs:
                if levels_a in boot_slopes and levels_b in boot_slopes:
                    diff = boot_slopes[levels_a] - boot_slopes[levels_b]
                    boot_diffs[(levels_a, levels_b)].append(diff)
                else:
                    boot_diffs[(levels_a, levels_b)].append(np.nan)

        # Compute CIs and build results
        alpha = (1 - ci_level) / 2
        results = []

        for levels_a, levels_b in pairs:
            boot_vals = np.array(boot_diffs[(levels_a, levels_b)])
            boot_vals = boot_vals[~np.isnan(boot_vals)]

            if len(boot_vals) == 0:
                continue

            diff = point_diffs.get((levels_a, levels_b), np.nan)
            slope_a = point_slopes.get(levels_a, np.nan)
            slope_b = point_slopes.get(levels_b, np.nan)

            # Bootstrap CI (percentile method)
            ci_low, ci_high = np.quantile(boot_vals, [alpha, 1 - alpha])

            # Bootstrap SE
            se_boot = np.std(boot_vals, ddof=1)

            # Bootstrap p-value (proportion of bootstrap distribution crossing zero)
            # Two-tailed: 2 * min(proportion >= 0, proportion <= 0)
            prop_positive = np.mean(boot_vals >= 0)
            prop_negative = np.mean(boot_vals <= 0)
            p_boot = 2 * min(prop_positive, prop_negative)
            p_boot = min(p_boot, 1.0)  # Cap at 1

            # Also compute parametric t-test for comparison
            # Using bootstrap SE and Welch-Satterthwaite df
            n_a = len(subsets[levels_a])
            n_b = len(subsets[levels_b])

            row = {
                "slope_a": slope_a,
                "slope_b": slope_b,
                "diff": diff,
                "se_boot": se_boot,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_boot": p_boot,
                "n_a": n_a,
                "n_b": n_b,
            }

            # Add level identifiers
            for i, var in enumerate(categorical_vars):
                row[f"{var}_a"] = levels_a[i]
                row[f"{var}_b"] = levels_b[i]

            results.append(row)

        df_results = pd.DataFrame(results)

        if len(df_results) == 0:
            return df_results

        # Apply p-value adjustment
        if p_adjust is not None and len(df_results) > 1:
            df_results["p_adj"] = self._adjust_pvalues(
                df_results["p_boot"].values, method=p_adjust
            )

        # Reorder columns
        level_cols = []
        for var in categorical_vars:
            level_cols.extend([f"{var}_a", f"{var}_b"])

        other_cols = [
            "slope_a",
            "slope_b",
            "diff",
            "se_boot",
            "ci_low",
            "ci_high",
            "p_boot",
        ]
        if "p_adj" in df_results.columns:
            other_cols.append("p_adj")
        other_cols.extend(["n_a", "n_b"])

        return df_results[level_cols + other_cols].reset_index(drop=True)


def stats_dataframe(
    df, DV_col, DV_dtype=float, DV_scale=1.0, IV_col=None, IV_dtype=float, IV_scale=1
) -> pd.DataFrame:
    """Simple function to generate a clean dataframe for ANOVA / ANCOVA fitting

    Parameters
    ----------
    df : pd.DataFrame
        data
    DV_col : str
        Column name of the Dv column
    DV_dtype : type, optional
        dastatype of DV column, by default float
    DV_scale : float, optional
        value to scale DV column by, by default 1
    IV_col : str, optional
        string for additional covariate column, if needed, by default None
    IV_dtype : type, optional
        IV datatype, by default float
    IV_scale : int, optional
        take a guess, by default 1

    Returns
    -------
    pd.DataFrame
        data frame with ['DV','Type','Subtype'] columns, and optionaly, 'IV'
    """
    if IV_col != None:
        df = pd.DataFrame(
            {
                "DV": df[DV_col].values.astype(DV_dtype) * DV_scale,
                "Type": df["Type"].to_list(),
                "Subtype": df["Subtype"].str[-1],
                "IV": df[IV_col].values.astype(IV_dtype) * IV_scale,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "DV": df[DV_col].values.astype(DV_dtype) * DV_scale,
                "Type": df["Type"].to_list(),
                "Subtype": df["Subtype"].str[-1],
            }
        )
    return df
