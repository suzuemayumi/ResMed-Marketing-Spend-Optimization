import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lightweight_mmm import lightweight_mmm, optimize_media
from lightweight_mmm import media_transforms, preprocessing
from scipy import optimize
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
    """Wrapper around optimize_media.find_optimal_budgets that supports locks.

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
    replaced using linear interpolation across the date sequence.
    """
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = (
        df[numeric_cols].replace(0, np.nan).interpolate(method="linear").bfill().ffill()
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


# Page configuration
st.set_page_config(page_title="Marketing Spend Optimization")

st.title("Marketing Spend Optimization")

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

            unlocked_idxs = [i for i in range(len(media_cols)) if i not in locked]
            if unlocked_idxs:
                available = total_budget - sum(locked.values())
                unlocked_sum = sum(final_alloc[media_cols[i]] for i in unlocked_idxs)
                if unlocked_sum > 0:
                    scale = available / unlocked_sum
                    for i in unlocked_idxs:
                        final_alloc[media_cols[i]] *= scale
                else:
                    equal_alloc = available / len(unlocked_idxs)
                    for i in unlocked_idxs:
                        final_alloc[media_cols[i]] = equal_alloc

            final_alloc = {c: max(0.0, v) for c, v in final_alloc.items()}

            display_alloc = {c: max(0.0, round(v, 2)) for c, v in final_alloc.items()}
            diff = round(total_budget - sum(display_alloc.values()), 2)
            if abs(diff) >= 0.01 and unlocked_idxs:
                col = media_cols[unlocked_idxs[0]]
                display_alloc[col] = round(max(0.0, display_alloc[col] + diff), 2)
            future_media = np.array([final_alloc[c] for c in media_cols]).reshape(1, -1)
            # ``model.predict`` returns a JAX array which cannot be directly cast
            # to a Python ``float`` when it has a non-scalar shape. ``mean(axis=0)``
            # yields an array with a single value, so we explicitly convert using
            # ``np.asarray`` and ``item()`` to extract the scalar value.
            future_pred = np.asarray(
                model.predict(media=future_media).mean(axis=0)
            ).item()

            st.session_state["optimized_results"] = {
                "alloc": display_alloc,
                "future_pred": future_pred,
            }
            st.session_state["apply_optimized_to_widgets"] = True

            # ``st.experimental_rerun`` was deprecated in Streamlit 1.27 in
            # favor of ``st.rerun``. Use whichever attribute is available so
            # the app works across versions.
            rerun = getattr(st, "experimental_rerun", st.rerun)
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
