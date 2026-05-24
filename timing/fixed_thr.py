import timing.pseudo_t
import numpy as np

def fixed_thr(signal_window, valid, max_idx, values_max, **kwargs):
  globals().update(kwargs)

  rise_valid = signal_window[valid, signal_samples_pre_peak - rise_samples_pre_peak:signal_samples_pre_peak + rise_samples_post_peak]
  thresholds = np.ones_like(values_max)*timing_thr

  idx_valid = np.where(valid)
  thr_valid  = thresholds[idx_valid]        # (N_valid,)

  return {"time": timing.pseudo_t(rise_valid, valid, thr_valid, sampling_rate, interpolation_factor, max_idx, rise_interp_left_samples, rise_interp_right_samples, rise_samples_pre_peak, thr_tol=timing_thr_tol)}

