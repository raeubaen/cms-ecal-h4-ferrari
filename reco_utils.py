import numpy as np

def split(waveforms, threshold=None, pre=5, post=10, baseline_samples=10):

    # Assume waveforms is shape (E, C, S)
    E, C, S = waveforms.shape

    if threshold is not None:
      argmax_idx = np.argmax(waveforms > threshold, axis=2)  # shape (E, C)
    else:
      argmax_idx = np.argmax(waveforms, axis=2)  # shape (E, C)

    # Step 2: Build offsets
    window_offsets = np.arange(-int(pre), int(post)).reshape(1, 1, -1)         # shape (1,1,pre+post)
    baseline_offsets = np.arange(-int(pre)-int(baseline_samples), -int(pre)).reshape(1, 1, -1)      # shape (1,1,bs_samples)

    # Expand argmax index for broadcasting
    argmax_exp = argmax_idx[:, :, np.newaxis]  # shape (E, C, None)

    # Add offsets and wrap with modulo S to stay in bounds
    window_indices   = (argmax_exp + window_offsets) % S        # shape (E, C, pre+post)
    baseline_indices = (argmax_exp + baseline_offsets) % S      # shape (E, C, bs_samples)

    # Build broadcasted event/channel indices
    event_idx = np.arange(E)[:, None, None]
    chan_idx  = np.arange(C)[None, :, None]

    baseline_waveforms = waveforms[event_idx, chan_idx, baseline_indices]    # (E, C, bs_samples)

    # Step 3: Compute baseline mean
    baseline = np.mean(baseline_waveforms, axis=2)	 # shape (E, C)
    baseline_std = np.std(baseline_waveforms, axis=2)    # shape (E, C)
    baseline_integral = np.sum(baseline_waveforms, axis=2)  # shape (E, C)

    return argmax_idx, baseline, baseline_std, baseline_integral, (event_idx, chan_idx, window_indices)


def find_central_region(charge_mean, ix, iy, central_region_width, fixed_central_region=None, min_outer_over_seed_ratio=None):
    fake_mask = np.full(ix.shape, True)
    mask_central_region = np.full(ix.shape, True)


    while True:
      if fixed_central_region:
          seed_ch = np.argmax(np.logical_and(ix==fixed_central_region[0],iy==fixed_central_region[1]))
          print(f"Seed channel: {seed_ch}", flush=True)
      else:
        charge_mean[~fake_mask] = 0
        seed_ch = np.argmax(charge_mean)

      if int(central_region_width) % 2 == 0: raise ValueError("Central region width MUST be odd")

      all_minus_seed = central_region_width**2 - 1
      cut = (central_region_width+1)/2.

      ix_seed, iy_seed = ix[seed_ch], iy[seed_ch]
      mask_central_region = np.logical_and(np.abs(ix - ix_seed) < cut, np.abs(iy - iy_seed) < cut)
      mask_central_region[seed_ch] = False
      seed_central_region_ratio = np.sum(charge_mean[mask_central_region]) * all_minus_seed / np.sum(mask_central_region) / charge_mean[seed_ch]
      mask_central_region[seed_ch] = True
      if fixed_central_region: return mask_central_region, seed_ch
      if seed_central_region_ratio < min_outer_over_seed_ratio:
        fake_mask[seed_ch] = False
        continue
      else:
        break
    print(f"Seed channel: {seed_ch}", flush=True)
    return mask_central_region, seed_ch

