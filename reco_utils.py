import numpy as np

def decode_ecal_waves(waves, gain_list):
    bit13_mask = 1 << 13 #validity bit
    bit12_mask = 1 << 12 #gain bit
    amp_mask   = 0x0FFF #amplitude mask
    is_valid = (waves & bit13_mask) != 0
    gain_is_high = (waves & bit12_mask) != 0
    amplitudes = (waves & amp_mask).astype(np.float32)

    gains = gain_list[None, :, None]

    amplitudes *= np.where(gain_is_high, gains, 1)
    #amplitudes[~is_valid] = 0
    return amplitudes, is_valid, gain_is_high


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


def find_5x5(charge_mean, ieta, iphi, fixed_5x5=None):
    fake_mask = np.full(ieta.shape, True)
    mask_5x5 = np.full(ieta.shape, True)


    while True:
      if fixed_5x5:
          seed_ch = np.argmax(np.logical_and(ieta==fixed_5x5[0],iphi==fixed_5x5[1]))
          print(f"Seed channel: {seed_ch}", flush=True)
      else:
        charge_mean[~fake_mask] = 0
        seed_ch = np.argmax(charge_mean)
      ieta_seed, iphi_seed = ieta[seed_ch], iphi[seed_ch]
      mask_5x5 = np.logical_and(np.abs(ieta - ieta_seed) < 3, np.abs(iphi - iphi_seed) < 3)
      mask_5x5[seed_ch] = False
      seed_5x5_ratio = np.sum(charge_mean[mask_5x5]) * 24 / np.sum(mask_5x5) / charge_mean[seed_ch]
      mask_5x5[seed_ch] = True
      if fixed_5x5: return mask_5x5, seed_ch
      if seed_5x5_ratio < 0.2:
        fake_mask[seed_ch] = False
        continue
      else:
        break
    print(f"Seed channel: {seed_ch}", flush=True)
    return mask_5x5, seed_ch

