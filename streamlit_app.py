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

# Replace the library's implementation if present
if hasattr(media_transforms, "apply_exponent_safe"):
    media_transforms.apply_exponent_safe = _apply_exponent_safe
# Page configuration
st.set_page_config(page_title="Marketing Spend Optimization")

st.title("Marketing Spend Optimization")

# Sidebar instructions and budget input
st.sidebar.header("Simulation Settings")
st.sidebar.write(
    "Upload a CSV or Excel file with columns: Date, conversion, search_cost, video_cost, meta_cost."
)

total_budget = st.sidebar.number_input("Total Budget", min_value=0.0, value=1000.0, step=100.0)

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
    model.fit(media=media_data,
              target=target,
              media_prior=np.ones(len(media_cols)))

    st.success("Model fitted")

    # Predict historical conversions
    predictions = model.predict(media=media_data)
    df["predicted_conversions"] = predictions

    # Future spend sliders
    st.subheader("Adjust Future Spend")
    future_spend = {
        col: st.slider(
            f"{col.replace('_cost', '').title()} Spend", min_value=0.0,
            max_value=float(total_budget), value=float(df[col].iloc[-1])
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
    base_pred = model.predict(media=base_media)
    contribution = np.zeros_like(media_data, dtype=float)
    for i in range(len(media_cols)):
        temp = np.zeros_like(media_data)
        temp[:, i] = media_data[:, i]
        pred_i = model.predict(media=temp)
        contribution[:, i] = pred_i - base_pred

    fig2, ax2 = plt.subplots()
    for i, col in enumerate(media_cols):
        ax2.plot(df["Date"], contribution[:, i], label=col.replace("_cost", ""))
    ax2.legend()
    st.pyplot(fig2)
