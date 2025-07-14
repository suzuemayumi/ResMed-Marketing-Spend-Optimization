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
    static_argnames=("media_mix_model", "media_input_shape", "target_scaler", "media_scaler"),
)
def _objective_function(extra_features, media_mix_model, media_input_shape, media_gap,
                        target_scaler, media_scaler, geo_ratio, seed, media_values):
    if hasattr(media_mix_model, "n_geos") and media_mix_model.n_geos > 1:
        media_values = geo_ratio * jnp.expand_dims(media_values, axis=-1)
    media_values = jnp.tile(media_values / media_input_shape[0], reps=media_input_shape[0])
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
    future_spend = {
        col: st.slider(
            f"{col.replace('_cost', '').title()} Spend",
            min_value=0.0,
            max_value=float(total_budget),
            value=float(df[col].iloc[-1]),
        )
        for col in media_cols
    }

    future_media = np.array([future_spend[c] for c in media_cols]).reshape(1, -1)
    future_pred = model.predict(media=future_media)[0]
    st.metric("Predicted Future Conversions", future_pred)

    # Optimize button
    if st.button("Optimize"):
        solution, _, _ = optimize_media.find_optimal_budgets(
            n_time_periods=1,
            media_mix_model=model,
            budget=total_budget,
            prices=np.ones(len(media_cols)),
        )
        optimized_spend = np.round(solution.x.reshape(-1), 2)
        st.write(
            "Optimized Spend Allocation",
            dict(zip(media_cols, optimized_spend)),
        )

    # Plot predicted conversions over time
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["predicted_conversions"], label="Predicted")
    ax.plot(df["Date"], df["conversion"], label="Actual", alpha=0.5)
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
    ax2.legend()
    st.pyplot(fig2)
