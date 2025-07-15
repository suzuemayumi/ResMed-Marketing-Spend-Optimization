import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lightweight_mmm import lightweight_mmm, optimize_media
from lightweight_mmm import media_transforms
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
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    st.subheader("Raw Data")
    st.write(df.head())

    media_cols = ["search_cost", "video_cost", "meta_cost"]
    target = df["conversion"].astype(float).values
    media_data = df[media_cols].astype(float).values

    # Fit LightweightMMM model
    model = lightweight_mmm.LightweightMMM()
    model.fit(media=media_data, target=target, media_prior=np.ones(len(media_cols)))

    st.success("Model fitted")

    # Predict historical conversions
    predictions = model.predict(media=media_data).mean(axis=0)
    df["predicted_conversions"] = predictions

    # Future spend sliders
    st.subheader("Adjust Future Spend")
    spend_ranges = {col: (df[col].min(), df[col].max()) for col in media_cols}
    diminish_points = {col: df[col].quantile(0.9) for col in media_cols}
    future_spend = {}
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

        st.slider(
            f"{col_title} Spend",
            min_value=0.0,
            max_value=float(total_budget),
            value=st.session_state[slider_key],
            key=slider_key,
        )
        st.number_input(
            f"{col_title} Spend Value",
            min_value=0.0,
            value=st.session_state[slider_key],
            key=input_key,
        )
        st.checkbox("Lock", key=lock_key)

        if st.session_state[input_key] != st.session_state[slider_key]:
            st.session_state[slider_key] = st.session_state[input_key]

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

    if st.button("Run", key="run_prediction"):
        future_media = np.array([future_spend[c] for c in media_cols]).reshape(1, -1)
        st.session_state["future_pred"] = model.predict(media=future_media)[0]

    if "future_pred" in st.session_state:
        st.metric("Predicted Future Conversions", st.session_state["future_pred"])

    # Optimize button
    if st.button("Optimize"):
        solution, _, _ = optimize_media.find_optimal_budgets(
            n_time_periods=1,
            media_mix_model=model,
            budget=total_budget,
            prices=np.ones(len(media_cols)),
        )
        optimized_spend = np.round(solution.x.reshape(-1), 2)

        locked_channels = [c for c in media_cols if st.session_state.get(f"{c}_lock")]
        locked_budget = sum(future_spend[c] for c in locked_channels)
        unlocked_channels = [c for c in media_cols if c not in locked_channels]
        remaining_budget = max(total_budget - locked_budget, 0)
        unlocked_sum = sum(
            optimized_spend[media_cols.index(c)] for c in unlocked_channels
        )
        scale = remaining_budget / unlocked_sum if unlocked_sum else 0

        final_spend = {}
        for i, col in enumerate(media_cols):
            if col in locked_channels:
                final_spend[col] = future_spend[col]
            else:
                final_spend[col] = optimized_spend[i] * scale

        st.write("Optimized Spend Allocation", final_spend)

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
