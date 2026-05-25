from timing.pseudo_t import get_rise_interp
from registry import register_routine
import numpy as np

@register_routine("zero_crossing")
def zero_crossing(signal_window, valid, max_idx, values_max, **kwargs):
    globals().update(kwargs)

    rise_valid = signal_window[valid, signal_samples_pre_peak - rise_samples_pre_peak:signal_samples_pre_peak + rise_samples_post_peak]
    thresholds = np.ones_like(values_max)*timing_thr

    idx_valid = np.where(valid)
    thr_valid  = thresholds[idx_valid]

    prelim_pseudo_t, rise_interp = get_rise_interp(rise_valid, valid, thr_valid, interpolation_factor, max_idx, rise_interp_left_samples, rise_interp_right_samples, thr_tol=timing_thr_tol)


    print(rise_interp[0])

    n_wf, n_samples = rise_interp.shape

    # ============================================================
    # Precompute scalar fit quantities
    # ============================================================

    n = np.float32(n_samples)

    Sx = np.float32((n_samples - 1) * n_samples / 2)

    Sxx = np.float32(
        (n_samples - 1)
        * n_samples
        * (2 * n_samples - 1)
        / 6
    )

    denom = n * Sxx - Sx * Sx

    # ============================================================
    # Waveform-dependent reductions
    # ============================================================

    # Sy = sum(y)
    Sy = np.sum(rise_interp, axis=1, dtype=np.float32)

    # Sxy = sum(x*y)
    #
    # Avoid broadcasted x matrix
    #
    x = np.arange(n_samples, dtype=np.float32)

    Sxy = np.sum(
        rise_interp * x[None, :],
        axis=1,
        dtype=np.float32
    )

    # ============================================================
    # Linear fit coefficients
    # ============================================================

    a = (n * Sxy - Sx * Sy) / denom

    b = (Sy - a * Sx) / n

    # ============================================================
    # Zero crossing
    # ============================================================

    x0 = -b / a

    # Optional sub-sample dithering
    x0 += np.random.uniform(
        -0.5,
	0.5,
	size=x0.shape
    ).astype(np.float32)

    # ============================================================
    # Convert to original coordinates
    # ============================================================

    pseudo_t_valid = (
        x0 / np.float32(interpolation_factor)
        + prelim_pseudo_t
        - np.float32(rise_interp_left_samples)
        + max_idx[idx_valid].astype(np.float32)
        - np.float32(rise_samples_pre_peak)
    )

    pseudo_t_valid /= np.float32(sampling_rate)

    # ============================================================
    # Scatter back
    # ============================================================

    pseudo_t = np.zeros(valid.shape, dtype=np.float32)

    pseudo_t[idx_valid] = pseudo_t_valid

    return {"time": pseudo_t}

