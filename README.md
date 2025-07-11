

# ResMed Marketing Spend Optimization

This project uses Streamlit to visualize and analyze marketing spend data. Follow the steps below to get the application running locally.

## Setup

1. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app**:

   ```bash
   streamlit run streamlit_app.py
   ```

## Data Requirements

The app expects a CSV or Excel file containing the following columns:

- `Date`
- `conversion`
- `search_cost`
- `video_cost`
- `meta_cost`

Make sure your data file includes these columns so that the application can load and visualize the marketing spend correctly.

## Testing

There are currently no unit tests in this repository. However, you can run `pytest` to verify that the Python environment is correctly configured.
