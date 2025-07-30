import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lightweight_mmm import lightweight_mmm, optimize_media
from lightweight_mmm import media_transforms, preprocessing
from scipy import optimize, stats
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Compatibility patch for JAX 0.4+ where `jnp.where` no longer accepts keyword
# arguments for `x` and `y`. Older versions of `lightweight_mmm` still call the
# function using keywords, so we monkey patch the helper used during model fit
# to avoid a ``TypeError``.
# ---------------------------------------------------------------------------
def _apply_exponent_safe(data, exponent):
    """Replicates media_transforms.apply_exponent_safe without kw args."""
    exponent_safe = jnp.where(data == 0, 1, data) ** exponent
    return jnp.where(data == 0, 0, exponent_safe - 1)


def _hill(data, half_max_effective_concentration, slope):
    save_transform = _apply_exponent_safe(
        data=data / half_max_effective_concentration,
        exponent=-slope,
    )
    return jnp.where(save_transform == 0, 0, 1.0 / (1 + save_transform))


# Replace the library's implementation if present
if hasattr(media_transforms, "apply_exponent_safe"):
    media_transforms.apply_exponent_safe = _apply_exponent_safe
if hasattr(media_transforms, "hill"):
    media_transforms.hill = _hill

# ---------------------------------------------------------------------------
# Additional compatibility fix for ``jnp.reshape`` on older versions of
# ``lightweight_mmm``. In JAX 0.4+, ``jnp.reshape`` no longer accepts the
# ``newshape`` keyword argument. The library uses this keyword in its internal
# optimization helper. We monkey patch the function to ensure compatibility.
# ---------------------------------------------------------------------------
import functools
import jax


def _sync_from_slider(slider_key: str, input_key: str) -> None:
    """Update the number input when a slider changes."""
    st.session_state[input_key] = st.session_state[slider_key]


def _sync_from_input(slider_key: str, input_key: str) -> None:
    """Update the slider when a number input changes."""
    st.session_state[slider_key] = st.session_state[input_key]


@functools.partial(
    jax.jit,
    static_argnames=(
        "media_mix_model",
        "media_input_shape",
        "target_scaler",
        "media_scaler",
    ),
)
def _objective_function(
    extra_features,
    media_mix_model,
    media_input_shape,
    media_gap,
    target_scaler,
    media_scaler,
    geo_ratio,
    seed,
    media_values,
):
    if hasattr(media_mix_model, "n_geos") and media_mix_model.n_geos > 1:
        media_values = geo_ratio * jnp.expand_dims(media_values, axis=-1)
    media_values = jnp.tile(
        media_values / media_input_shape[0], reps=media_input_shape[0]
    )
    media_values = jnp.reshape(media_values, media_input_shape)
    media_values = media_scaler.transform(media_values)
    return -jnp.sum(
        media_mix_model.predict(
            media=media_values.reshape(media_input_shape),
            extra_features=extra_features,
            media_gap=media_gap,
            target_scaler=target_scaler,
            seed=seed,
        ).mean(axis=0)
    )


if hasattr(optimize_media, "_objective_function"):
    optimize_media._objective_function = _objective_function


def find_optimal_budgets_with_locks(
    model,
    budget,
    prices,
    locked_spend,
    bounds_lower_pct=0.2,
    bounds_upper_pct=0.2,
    max_iterations=200,
    solver_func_tolerance=1e-6,
    solver_step_size=1.4901161193847656e-08,
    progress_callback=None,
):
    """Run the allocation optimization used by the manual tool.

    This helper wraps ``optimize_media.find_optimal_budgets`` so that the
    Streamlit interface can lock specific channels when users experiment with
    manual spend values. It builds per-channel bounds, fixes locked channels to
    their chosen spend, and then maximizes the predicted KPI using SciPy's SLSQP
    solver. Progress updates are forwarded to ``progress_callback`` so the UI can
    display a progress bar.

    Parameters
    ----------
    model : LightweightMMM
        Trained media mix model.
    budget : float
        Total budget to distribute.
    prices : array-like
        Channel spend multipliers.
    locked_spend : dict
        Mapping of channel index to locked spend value.
    progress_callback : callable, optional
        Function called with a float between 0 and 1 after each iteration to
        update a progress indicator in the UI.
    """
    n_time_periods = 1
    jax.config.update("jax_enable_x64", True)
    if isinstance(bounds_lower_pct, float):
        bounds_lower_pct = jnp.repeat(bounds_lower_pct, len(prices))
    else:
        bounds_lower_pct = jnp.array(bounds_lower_pct)
    if isinstance(bounds_upper_pct, float):
        bounds_upper_pct = jnp.repeat(bounds_upper_pct, len(prices))
    else:
        bounds_upper_pct = jnp.array(bounds_upper_pct)

    base_bounds = optimize_media._get_lower_and_upper_bounds(
        media=model.media,
        n_time_periods=n_time_periods,
        lower_pct=bounds_lower_pct,
        upper_pct=bounds_upper_pct,
        media_scaler=None,
    )

    lb = np.array(base_bounds.lb)
    ub = np.array(base_bounds.ub)
    starting_values = optimize_media._generate_starting_values(
        n_time_periods=n_time_periods,
        media=model.media,
        media_scaler=None,
        budget=budget,
        prices=prices,
    )

    for idx, spend in locked_spend.items():
        lb[idx] = spend
        ub[idx] = spend
        starting_values[idx] = spend

    bounds = optimize.Bounds(lb=lb, ub=ub)
    media_scaler = preprocessing.CustomScaler(multiply_by=1, divide_by=1)
    if model.n_geos == 1:
        geo_ratio = 1.0
    else:
        average_per_time = model.media.mean(axis=0)
        geo_ratio = average_per_time / jnp.expand_dims(
            average_per_time.sum(axis=-1), axis=-1
        )
    media_input_shape = (n_time_periods, *model.media.shape[1:])
    partial_obj = functools.partial(
        optimize_media._objective_function,
        None,
        model,
        media_input_shape,
        None,
        None,
        media_scaler,
        geo_ratio,
        None,
    )

    iteration = 0

    def _cb(_x):
        nonlocal iteration
        iteration += 1
        if progress_callback is not None:
            progress_callback(min(iteration / max_iterations, 1.0))

    solution = optimize.minimize(
        fun=partial_obj,
        x0=starting_values,
        bounds=bounds,
        method="SLSQP",
        jac="3-point",
        callback=_cb,
        options={
            "maxiter": max_iterations,
            "disp": True,
            "ftol": solver_func_tolerance,
            "eps": solver_step_size,
        },
        constraints={
            "type": "eq",
            "fun": optimize_media._budget_constraint,
            "args": (prices, budget),
        },
    )
    if progress_callback is not None:
        progress_callback(1.0)
    kpi_without_optim = optimize_media._objective_function(
        extra_features=None,
        media_mix_model=model,
        media_input_shape=media_input_shape,
        media_gap=None,
        target_scaler=None,
        media_scaler=media_scaler,
        seed=None,
        geo_ratio=geo_ratio,
        media_values=starting_values,
    )

    jax.config.update("jax_enable_x64", False)
    return solution, kpi_without_optim, starting_values


@st.cache_data(show_spinner=False)
def _load_dataframe(uploaded_file) -> pd.DataFrame:
    """Load the uploaded CSV or Excel file and sort by date.

    Zero values in numeric columns are treated as missing data and
    replaced using forward and backward fill across the date sequence.
    """
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = (
        df[numeric_cols]
        .replace(0, np.nan)
        .fillna(method="ffill")
        .fillna(method="bfill")
    )
    return df


@st.cache_resource(show_spinner=False)
def _train_model(
    media_data: np.ndarray, target: np.ndarray, n_channels: int
) -> lightweight_mmm.LightweightMMM:
    """Fit and return a cached media mix model."""
    model = lightweight_mmm.LightweightMMM()
    model.fit(media=media_data, target=target, media_prior=np.ones(n_channels))
    return model


def _calculate_statistics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return basic evaluation statistics for predictions."""
    residuals = actual - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    if len(actual) > 1:
        _t, p_value = stats.ttest_rel(actual, predicted)
    else:
        p_value = np.nan
    r2 = float(np.nan_to_num(r2))
    rmse = float(np.nan_to_num(rmse))
    p_value = float(p_value) if not np.isnan(p_value) else np.nan
    return {"r2": r2, "rmse": rmse, "p_value": p_value}


def _validation_metrics(
    media_data: np.ndarray, target: np.ndarray, n_channels: int, test_size: float = 0.2
) -> dict:
    """Train on a holdout split and compute validation statistics."""
    n = len(target)
    if n < 2:
        return {"r2": 0.0, "rmse": 0.0, "p_value": np.nan}
    split = max(1, int(n * (1 - test_size)))
    train_media, test_media = media_data[:split], media_data[split:]
    train_target, test_target = target[:split], target[split:]
    model = _train_model(train_media, train_target, n_channels)
    if len(test_target) == 0:
        return {"r2": 0.0, "rmse": 0.0, "p_value": np.nan}
    preds = model.predict(media=test_media).mean(axis=0)
    metrics = _calculate_statistics(test_target, np.asarray(preds))
    return {
        "r2": float(np.nan_to_num(metrics["r2"])),
        "rmse": float(np.nan_to_num(metrics["rmse"])),
        "p_value": (
            float(metrics["p_value"]) if not np.isnan(metrics["p_value"]) else np.nan
        ),
    }


# Page configuration
st.set_page_config(page_title="Marketing Spend Optimization")

st.title("Marketing Spend Optimization")

# Brief summary of the underlying optimization approach
with st.expander("How the Model Works"):
    st.markdown(
        "The app first trains a Bayesian media mix model on your historical data"
        " to learn how each marketing channel contributes to the target"
        " conversion metric. Once the model is fit it can predict conversions for"
        " any hypothetical spend scenario.\n\n"
        "When you click **Optimize**, a modified version of "
        "`lightweight_mmm.optimize_media.find_optimal_budgets` searches for the"
        " spend allocation that maximizes predicted conversions. SciPy's SLSQP"
        " solver redistributes the total budget across all unlocked channels while"
        " keeping locked channels fixed. Bounds based on each channel's historical"
        " minimum and maximum spend ensure the solution remains realistic. Progress"
        " updates are shown in the status bar until convergence."
    )

# Sidebar instructions and budget input
st.sidebar.header("Simulation Settings")
st.sidebar.write(
    "Upload a CSV or Excel file with columns: Date, conversion, search_cost, video_cost, meta_cost."
)

total_budget = st.sidebar.number_input(
    "Total Budget", min_value=0.0, value=1000.0, step=100.0
)

# File uploader
uploaded_file = st.file_uploader("Upload marketing data", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    progress = st.progress(0.0)
    df = _load_dataframe(uploaded_file)
    progress.progress(0.4)

    # Reset widget defaults when a new file is uploaded
    file_id = uploaded_file.name
    if st.session_state.get("current_file") != file_id:
        st.session_state["current_file"] = file_id
        for col in ["search_cost", "video_cost", "meta_cost"]:
            st.session_state[f"{col}_slider"] = 0.0
            st.session_state[f"{col}_input"] = 0.0
            st.session_state[f"{col}_lock"] = False
        st.session_state.pop("optimized_results", None)
        st.session_state["apply_optimized_to_widgets"] = False

    st.subheader("Raw Data")
    st.write(df.head())

    media_cols = ["search_cost", "video_cost", "meta_cost"]
    target = df["conversion"].astype(float).values
    media_data = df[media_cols].astype(float).values

    with st.spinner("Fitting model..."):
        model = _train_model(media_data, target, len(media_cols))
    progress.progress(0.8)

    predictions = model.predict(media=media_data).mean(axis=0)
    df["predicted_conversions"] = predictions

    train_stats = _calculate_statistics(target, np.asarray(predictions))
    val_stats = _validation_metrics(media_data, target, len(media_cols))

    progress.progress(1.0)
    progress.empty()

    spend_ranges = {col: (df[col].min(), df[col].max()) for col in media_cols}
    diminish_points = {col: df[col].quantile(0.9) for col in media_cols}

    future_spend = {}
    st.session_state.setdefault("apply_optimized_to_widgets", False)

    # If we have optimized spend allocations from the previous run, we
    # update the slider defaults once on the next rerun. Without this
    # guard the widgets would overwrite any user adjustments because the
    # code executes on every rerun triggered by Streamlit when a widget
    # value changes.
    if "optimized_results" in st.session_state and st.session_state.get(
        "apply_optimized_to_widgets"
    ):
        for col in media_cols:
            alloc_val = max(0.0, st.session_state["optimized_results"]["alloc"][col])
            st.session_state[f"{col}_slider"] = alloc_val
            st.session_state[f"{col}_input"] = alloc_val
        st.session_state["apply_optimized_to_widgets"] = False

    for col in media_cols:
        col_title = col.replace("_cost", "").title()
        slider_key = f"{col}_slider"
        input_key = f"{col}_input"
        lock_key = f"{col}_lock"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = float(df[col].iloc[-1])
        if input_key not in st.session_state:
            st.session_state[input_key] = float(df[col].iloc[-1])
        if lock_key not in st.session_state:
            st.session_state[lock_key] = False

        max_val = float(max(total_budget, spend_ranges[col][1], df[col].iloc[-1]))
        st.slider(
            f"{col_title} Spend",
            min_value=0.0,
            max_value=max_val,
            key=slider_key,
            on_change=_sync_from_slider,
            args=(slider_key, input_key),
            disabled=st.session_state[lock_key],
        )
        st.number_input(
            f"{col_title} Spend Value",
            min_value=0.0,
            max_value=max_val,
            key=input_key,
            on_change=_sync_from_input,
            args=(slider_key, input_key),
            disabled=st.session_state[lock_key],
        )
        st.checkbox(f"Lock {col_title}", key=lock_key)

        val = st.session_state[slider_key]
        if val < spend_ranges[col][0] or val > spend_ranges[col][1]:
            st.warning(
                f"{col_title} spend outside historical range "
                f"({spend_ranges[col][0]:.2f} - {spend_ranges[col][1]:.2f})"
            )
        elif val > diminish_points[col]:
            st.info(
                f"{col_title} spend exceeds estimated diminishing return "
                f"point ({diminish_points[col]:.2f})"
            )
        future_spend[col] = val

    optimize_clicked = st.button("Optimize")

    if optimize_clicked:
        locked = {
            i: future_spend[col]
            for i, col in enumerate(media_cols)
            if st.session_state.get(f"{col}_lock")
        }
        locked_total = sum(locked.values())
        if locked_total > total_budget:
            st.error("Locked spend exceeds total budget")
        else:
            progress_bar = st.progress(0.0)

            def _update(pct):
                progress_bar.progress(pct)

            solution, _, _ = find_optimal_budgets_with_locks(
                model,
                budget=total_budget,
                prices=np.ones(len(media_cols)),
                locked_spend=locked,
                progress_callback=_update,
            )
            progress_bar.empty()

            optimized_spend = np.maximum(solution.x.reshape(-1), 0)
            final_alloc = {
                col: float(locked.get(i, optimized_spend[i]))
                for i, col in enumerate(media_cols)
            }

            # Ensure spend sums to the total budget before rounding
            current_total = sum(final_alloc.values())
            diff = total_budget - current_total
            if abs(diff) > 1e-6:
                for i, col in enumerate(media_cols):
                    if i not in locked:
                        final_alloc[col] += diff
                        break

            # If adjustment pushed a channel negative, clip and redistribute
            negatives = [c for c, v in final_alloc.items() if v < 0 and c not in locked]
            if negatives:
                for c in negatives:
                    final_alloc[c] = 0.0
                diff2 = total_budget - sum(final_alloc.values())
                if abs(diff2) > 1e-6:
                    for i, col in enumerate(media_cols):
                        if i not in locked and final_alloc[col] + diff2 >= 0:
                            final_alloc[col] += diff2
                            break

            display_alloc = {c: round(v, 2) for c, v in final_alloc.items()}
            diff = round(total_budget - sum(display_alloc.values()), 2)
            if abs(diff) >= 0.01:
                for i, col in enumerate(media_cols):
                    if i not in locked and display_alloc[col] + diff >= 0:
                        display_alloc[col] = round(display_alloc[col] + diff, 2)
                        break
            future_media = np.array([final_alloc[c] for c in media_cols]).reshape(1, -1)
            # ``model.predict`` returns a JAX array which cannot be directly cast
            # to a Python ``float`` when it has a non-scalar shape. ``mean(axis=0)``
            # yields an array with a single value, so we explicitly convert using
            # ``np.asarray`` and ``item()`` to extract the scalar value.
            future_pred = np.asarray(
                model.predict(media=future_media).mean(axis=0)
            ).item()

            # Metrics based on the optimized spend applied across all periods
            optimized_media_full = np.tile(future_media, (len(media_data), 1))
            opt_predictions = model.predict(media=optimized_media_full).mean(axis=0)
            opt_train_stats = _calculate_statistics(target, np.asarray(opt_predictions))
            opt_val_stats = _validation_metrics(
                optimized_media_full, target, len(media_cols)
            )

            st.session_state["optimized_results"] = {
                "alloc": display_alloc,
                "future_pred": future_pred,
                "train_stats": opt_train_stats,
                "val_stats": opt_val_stats,
            }
            st.session_state["apply_optimized_to_widgets"] = True

            # ``st.experimental_rerun`` was deprecated in Streamlit 1.27 in
            # favor of ``st.rerun``. Use whichever attribute is available so
            # the app works across versions. ``getattr`` must not evaluate a
            # missing attribute, so check for each explicitly.
            rerun = getattr(st, "rerun", None)
            if rerun is None:
                rerun = getattr(st, "experimental_rerun", None)
            if rerun is None:
                raise RuntimeError(
                    "Streamlit does not provide a rerun function in this version"
                )
            rerun()

    if "optimized_results" in st.session_state:
        st.write(
            "Optimized Spend Allocation",
            st.session_state["optimized_results"]["alloc"],
        )
        st.metric(
            "Predicted Future Conversions",
            st.session_state["optimized_results"]["future_pred"],
        )

    metrics_train = train_stats
    metrics_val = val_stats
    if "optimized_results" in st.session_state:
        metrics_train = st.session_state["optimized_results"].get(
            "train_stats", metrics_train
        )
        metrics_val = st.session_state["optimized_results"].get(
            "val_stats", metrics_val
        )

    st.subheader("Model Metrics")
    st.write(f"R\u00b2: {metrics_train['r2']:.4f}")
    st.write(f"RMSE: {metrics_train['rmse']:.2f}")
    st.write(f"Paired t-test p-value: {metrics_train['p_value']:.4f}")
    st.write(f"Validation R\u00b2: {metrics_val['r2']:.4f}")
    st.write(f"Validation RMSE: {metrics_val['rmse']:.2f}")
    st.write(f"Validation p-value: {metrics_val['p_value']:.4f}")

    # Plot predicted conversions over time
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["predicted_conversions"], label="Predicted")
    ax.plot(df["Date"], df["conversion"], label="Actual", alpha=0.5)
    ax.set_title("Predicted vs Actual Conversions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Conversions")
    ax.legend()
    st.pyplot(fig)

    # Contribution chart
    base_media = np.zeros_like(media_data)
    base_pred = model.predict(media=base_media).mean(axis=0)
    contribution = np.zeros_like(media_data, dtype=float)
    for i in range(len(media_cols)):
        temp = np.zeros_like(media_data)
        temp[:, i] = media_data[:, i]
        pred_i = model.predict(media=temp).mean(axis=0)
        contribution[:, i] = pred_i - base_pred

    fig2, ax2 = plt.subplots()
    for i, col in enumerate(media_cols):
        ax2.plot(df["Date"], contribution[:, i], label=col.replace("_cost", ""))
    ax2.set_title("Media Channel Contribution to Conversions")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Estimated Contribution")
    ax2.legend()
    st.pyplot(fig2)
