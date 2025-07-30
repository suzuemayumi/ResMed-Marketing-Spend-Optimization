

# ResMed Marketing Spend Optimization

This repository contains a lightweight Streamlit application for exploring marketing channel performance and building media mix models with the `lightweight-mmm` library.

## Prerequisites

- Python 3.11 (confirmed compatible with `lightweight-mmm`)
- JAX and jaxlib pinned to **0.4.19** to avoid compatibility issues. Older
   releases of `lightweight-mmm` call `jnp.where` with keyword arguments, which
   is incompatible with JAX 0.4+. The Streamlit app includes a small monkey patch
   so you do not need to downgrade JAX.
## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch the Streamlit app with:

```bash
streamlit run streamlit_app.py
```

## Usage Notes

When the application loads your dataset, you can adjust future marketing spend
for each channel using interactive sliders **or** by typing an exact value into
the accompanying number input. Model predictions no longer update while you
adjust spend. Instead, the prediction and optimized allocation are calculated
when you click **Optimize**. If the entered amount falls outside the historical
range of your data, the app will display a warning. Values above the estimated
90th percentile for a channel trigger an informational message that spend may
suffer from diminishing returns.

Each slider also includes a **Lock** checkbox. Locked channels keep their
specified spend during optimization, allocating any remaining budget across the
other unlocked channels.

Progress bars appear while your data is loaded and when optimization runs so
you can track the current status.

## Allocation Model Behind the Manual Tool

The manual optimization tool relies on the same approach as
`lightweight_mmm.optimize_media.find_optimal_budgets`. It formulates an
optimization problem that distributes the total budget across all unlocked
channels while keeping any locked channels fixed to their specified spend. The
objective maximizes the KPI predicted by the trained media mix model using
SciPy's SLSQP solver. Bounds derived from historical spend values prevent the
optimizer from allocating unrealistic amounts to a channel.

## Data Requirements

The app expects a CSV or Excel file containing the following columns:

- `Date`
- `conversion`
- `search_cost`
- `video_cost`
- `meta_cost`

Make sure your data file includes these columns so that the application can load and visualize the marketing spend correctly. Any zero values in numeric columns are treated as missing data and filled using linear interpolation across the date sequence. Remaining gaps fall back to the column mean so all inputs stay positive.

## Testing

There are currently no unit tests in this repository. However, you can run `pytest` to verify that the Python environment is correctly configured.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
