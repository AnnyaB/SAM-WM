import numpy as np

from coolworld.ml.baselines import linear_trend_forecast, persistence_forecast


def test_persistence_repeats_last_real_input_value() -> None:
    context = np.array([[[[10.0]], [[11.0]], [[12.0]]]], dtype=np.float32)
    prediction = persistence_forecast(context, temperature_index=0, pred_len=2)
    np.testing.assert_allclose(prediction, np.array([[[12.0], [12.0]]]))


def test_linear_trend_extrapolates_deterministic_unit_sequence() -> None:
    context = np.array([[[[10.0]], [[11.0]], [[12.0]]]], dtype=np.float32)
    prediction = linear_trend_forecast(context, temperature_index=0, pred_len=2)
    np.testing.assert_allclose(prediction, np.array([[[13.0], [14.0]]]), atol=1e-6)
