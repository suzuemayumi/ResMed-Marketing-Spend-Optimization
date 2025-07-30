import io
import os
import sys
import types
import pandas as pd

# Ensure the application module is importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Stub out heavy dependencies used by streamlit_app at import time
class _DummyModule(types.ModuleType):
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

def _identity_decorator(*args, **kwargs):
    def _decorator(func):
        return func
    return _decorator

streamlit_stub = _DummyModule("streamlit")
streamlit_stub.cache_data = _identity_decorator
streamlit_stub.cache_resource = _identity_decorator
streamlit_stub.sidebar = types.SimpleNamespace(header=lambda *a, **k: None,
                                              write=lambda *a, **k: None,
                                              number_input=lambda *a, **k: 0.0)
streamlit_stub.session_state = {}
streamlit_stub.progress = lambda *a, **k: types.SimpleNamespace(progress=lambda *a, **k: None, empty=lambda: None)
streamlit_stub.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None)
streamlit_stub.button = lambda *a, **k: False
streamlit_stub.checkbox = lambda *a, **k: False
streamlit_stub.slider = lambda *a, **k: 0.0
streamlit_stub.number_input = lambda *a, **k: 0.0
streamlit_stub.write = lambda *a, **k: None
streamlit_stub.set_page_config = lambda *a, **k: None
streamlit_stub.title = lambda *a, **k: None
streamlit_stub.subheader = lambda *a, **k: None
streamlit_stub.pyplot = lambda *a, **k: None
streamlit_stub.metric = lambda *a, **k: None
streamlit_stub.rerun = lambda *a, **k: None
streamlit_stub.experimental_rerun = lambda *a, **k: None
streamlit_stub.file_uploader = lambda *a, **k: None
class _DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False

streamlit_stub.expander = lambda *a, **k: _DummyContext()
sys.modules.setdefault("streamlit", streamlit_stub)

# lightweight_mmm stubs
lw_stub = _DummyModule("lightweight_mmm")
lw_stub.lightweight_mmm = _DummyModule("lightweight_mmm.lightweight_mmm")
lw_stub.optimize_media = _DummyModule("lightweight_mmm.optimize_media")
lw_stub.media_transforms = _DummyModule("lightweight_mmm.media_transforms")
lw_stub.preprocessing = _DummyModule("lightweight_mmm.preprocessing")
lw_stub.lightweight_mmm.LightweightMMM = object
lw_stub.optimize_media._objective_function = lambda *a, **k: None
lw_stub.media_transforms.apply_exponent_safe = lambda *a, **k: None
lw_stub.media_transforms.hill = lambda *a, **k: None
lw_stub.preprocessing.CustomScaler = object
sys.modules.setdefault("lightweight_mmm", lw_stub)
sys.modules.setdefault("lightweight_mmm.lightweight_mmm", lw_stub.lightweight_mmm)
sys.modules.setdefault("lightweight_mmm.optimize_media", lw_stub.optimize_media)
sys.modules.setdefault("lightweight_mmm.media_transforms", lw_stub.media_transforms)
sys.modules.setdefault("lightweight_mmm.preprocessing", lw_stub.preprocessing)

# scipy and matplotlib stubs
sys.modules.setdefault("scipy", _DummyModule("scipy"))
sys.modules.setdefault("scipy.optimize", _DummyModule("scipy.optimize"))
sys.modules.setdefault("matplotlib", _DummyModule("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", _DummyModule("matplotlib.pyplot"))

# jax stubs
jax_stub = _DummyModule("jax")
jax_stub.jit = lambda *a, **k: (lambda f: f)
jax_stub.numpy = _DummyModule("jax.numpy")
jax_stub.numpy.where = lambda *a, **k: None
jax_stub.numpy.tile = lambda *a, **k: None
jax_stub.numpy.reshape = lambda *a, **k: None
jax_stub.numpy.repeat = lambda *a, **k: None
jax_stub.config = types.SimpleNamespace(update=lambda *a, **k: None)
sys.modules.setdefault("jax", jax_stub)
sys.modules.setdefault("jax.numpy", jax_stub.numpy)
sys.modules.setdefault("jnp", jax_stub.numpy)

import streamlit_app


def test_load_dataframe_fills_zero_values():
    data = {
        "Date": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
        ],
        "conversion": [0, 10, 0, 20, 0],
        "search_cost": [0, 100, 0, 200, 0],
    }
    df = pd.DataFrame(data)
    buffer = io.BytesIO(df.to_csv(index=False).encode())
    buffer.name = "test.csv"

    result = streamlit_app._load_dataframe(buffer)

    numeric_cols = result.select_dtypes(include="number").columns
    assert (result[numeric_cols] == 0).sum().sum() == 0
