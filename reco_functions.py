import time
import numpy as np
import multiprocessing as mp

import reco_utils
import timing as timing_module


def generic_reco(waves, detector_name, **kwargs):

  globals().update(kwargs)

  t0 = time.time()


  max_idx, baselines, baselines_std, baseline_integral, signal_window_3d_indices = reco_utils.split(waves, pre=signal_samples_pre_peak, post=signal_samples_post_peak, threshold=raw_threshold_before_peak_finding)

  print(f"baselines evaluation took: {time.time() - t0}")
  t0 = time.time()

  values_mean = np.mean(waves, axis=2) # mean of all values
  values_std = np.std(waves, axis=2)   # std of all values

  event_idx = np.arange(waves.shape[0])[:, None]        # shape (E, 1)
  chan_idx  = np.arange(waves.shape[1])[None, :]        # shape (1, C)

  if baseline_subtract:
    values_max = waves[event_idx, chan_idx, max_idx] - baselines
  else:
    values_max = waves[event_idx, chan_idx, max_idx]

  mask_under_thr = values_max < charge_zerosup_peak_threshold
  waves[mask_under_thr, :] = 0


  if baseline_subtract:
    waves[~mask_under_thr, :] = waves[~mask_under_thr, :] - baselines[~mask_under_thr, None]

  signal_window = waves[*signal_window_3d_indices]

  charge = np.zeros_like(values_max)
  charge[~mask_under_thr] = np.sum(signal_window[~mask_under_thr, :], axis=-1)

  if charge_to_peak_conversion:
     charge = charge * charge_to_peak_slope

  ich = np.repeat(np.arange(0, waves.shape[1])[np.newaxis, :], charge.shape[0], axis=0)

  return_dict = {}
  mask_selected_events = np.ones((charge.shape[0],), dtype=bool)
  det = detector_name

  print(f"baseline subtraction, charge integration things took: {time.time() - t0}")
  t0 = time.time()

  if geo_dict is not None:
    ix, iy = (geo_dict[key] for key in coords_2d_list)
    if coord_z is not None: iz = (geo_dict[key] for key in coord_z)
    else: iz = None

    if do_central_region:
      charge_mean = np.mean(charge, axis=0)
      seed_ch = -999

      mask_central_region, seed_ch = reco_utils.find_central_region(charge_mean, ix, iy, central_region_width, fixed_central_region=fixed_central_region, min_outer_over_seed_ratio=min_outer_over_seed_ratio)
      print(f"find central_region took: {time.time() - t0}")
      t0 = time.time()


      peak_seed = values_max[:, seed_ch]

      charge_seed = charge[:, seed_ch]
      charge_sum_central_region = np.sum(charge[:, mask_central_region], axis=1)
      charge_sum_central_region = np.clip(charge_sum_central_region, seed_charge_threshold, None)

      mask_low_charge_seed = charge_seed > seed_charge_threshold

      # amplitude_map of the central_region matrix
      charge_fraction_central_region = np.zeros(charge.shape)
      charge_fraction_central_region[:, mask_central_region] = charge[:, mask_central_region] / charge_sum_central_region[:, np.newaxis]

      print(f"seed/central_region/fractions took: {time.time() - t0}")
      t0 = time.time()

      w_log = np.maximum(0.0,w0_centroid + np.log(np.clip(charge_fraction_central_region, 1e-8, None)))
      w_log /= (np.sum(w_log, axis=1, keepdims=True))

      ix_centroid = w_log[:, mask_central_region] @ ix[mask_central_region]
      iy_centroid = w_log[:, mask_central_region] @ iy[mask_central_region]

      print(f"centrois took: {time.time() - t0}")


      iy_within_central_region = iy - iy[seed_ch]
      ix_within_central_region = ix - ix[seed_ch]

      ix_within_central_region = np.repeat(ix_within_central_region[np.newaxis, :], charge.shape[0], axis=0)
      iy_within_central_region = np.repeat(iy_within_central_region[np.newaxis, :], charge.shape[0], axis=0)

      t0 = time.time()

      seed_ch_app = seed_ch
      seed_ch = np.repeat(np.ones(1,)*seed_ch, charge_sum_central_region.shape[0], axis=0)

      highest_ch = np.argmax(charge * mask_central_region, axis=1)
      highest_charge = np.take_along_axis(charge, highest_ch[:, None], axis=1).squeeze()
      highest_peak = np.take_along_axis(values_max, highest_ch[:, None], axis=1).squeeze()

      print(f"highest ch took: {time.time() - t0}")
      t0 = time.time()

      print(f"tau took: {time.time() - t0}")
      t0 = time.time()

      kxk = f"{central_region_width}x{central_region_width}"
      return_dict.update({
        f"{det}_charge_sum_{kxk}": charge_sum_central_region, f"{det}_charge_seed": charge_seed, f"{det}_peak_seed": peak_seed, f"{det}_seed_over_{kxk}": charge_fraction_central_region[:, seed_ch_app],
        f"{det}_highest_charge_over_{kxk}": highest_charge/charge_sum_central_region,
        f"{det}_{coords_2d_list[1]}_within_{kxk}": iy_within_central_region, f"{det}_{coords_2d_list[0]}_within_{kxk}": ix_within_central_region,
        f"{det}_charge_divided_{kxk}": charge_fraction_central_region, f"{det}_seed_ch": seed_ch,
        f"{det}_{coords_2d_list[0]}_centroid": ix_centroid, f"{det}_{coords_2d_list[1]}_centroid": iy_centroid,
        f"{det}_highest_ch": highest_ch, f"{det}_highest_charge": highest_charge, f"{det}_highest_peak": highest_peak,
      })

      #mask_selected_events = mask_low_charge_seed

    ix = np.repeat(ix[np.newaxis, :], charge.shape[0], axis=0)
    iy = np.repeat(iy[np.newaxis, :], charge.shape[0], axis=0)

  if do_tau:
      if do_central_region: tau_mask = mask_central_region
      else: tau_mask = np.full((signal_window.shape[1],), True)

      descent = signal_window[:, tau_mask, signal_samples_pre_peak+1:signal_samples_pre_peak+tau_descent_samples+1]
      log_w = np.log(np.clip(descent, 1, None))
      log_slopes = np.diff(log_w, axis=2) / descent.shape[2]
      tau = np.zeros(charge.shape)
      tau[:, tau_mask] = -1.0 / (1e-12 + np.median(log_slopes, axis=2) * sampling_rate)
      return_dict.update({f"{det}_tau": tau})

  if do_timing:
    if do_central_region: timing_mask = mask_central_region
    else: timing_mask = np.full((signal_window.shape[1],), True)

    valid = ~mask_under_thr & timing_mask[None, :]

    for timing_method in timing_methods:
      timing_function = getattr(timing_module, timing_method)

      timing_function_result = timing_function(signal_window, valid, max_idx, values_max, **kwargs)

      for key in timing_function_result:
        return_dict.update({f"{det}_{timing_method}_{key}": timing_function_result[key]})

  per_ch_info = {
    f"{det}_peak_pos": max_idx, f"{det}_peak_time": max_idx/sampling_rate,
    f"{det}_charge": charge, f"{det}_peak": values_max, f"{det}_baseline_mean": baselines,
    f"{det}_baseline_std": baselines_std, f"{det}_baseline_integral": baseline_integral/baseline_samples*signal_window.shape[2],
  }

  if save_mean_rms_all_samples:
    per_ch_info.update({f"{det}_samples_mean": values_mean, f"{det}_samples_std": values_std})
  if geo_dict is not None:
    per_ch_info.update({f"{det}_{coords_2d_list[0]}": ix, f"{det}_{coords_2d_list[1]}": iy})
  if id is not None:
    for var in id:
      per_ch_info.update({f"{det}_{var}": np.repeat(id[var][np.newaxis, :], waves.shape[0], axis=0)})

  if do_central_region and save_only_central_region_info:
    for key in per_ch_info:
      return_dict[key] = np.zeros(per_ch_info[key].shape)
      return_dict[key][:, mask_central_region] = per_ch_info[key][:, mask_central_region]
  else:
    return_dict.update(per_ch_info)

  return mask_selected_events, return_dict



#not implemented
def generic_reco_chunk(args):
    try:
        waves, det, kwargs = args
        return generic_reco(waves, det, **kwargs)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, flush=True)


#not implemented
def generic_reco_parallel(waves, detector_name, n_cpus=2, **kwargs):
    E = waves.shape[0]
    chunk_size = (E + n_cpus - 1) // n_cpus  # ceil division
    chunks = [(waves[i*chunk_size:(i+1)*chunk_size], detector_name, kwargs)
              for i in range(n_cpus)]

    print("opening pool")
    results = [generic_reco_chunk(chunk) for chunk in chunks]

    # Combine results
    masks_list, dicts_list = zip(*results)
    combined_mask = np.concatenate(masks_list, axis=0)

    combined_dict = {}
    for key in dicts_list[0].keys():
        combined_dict[key] = np.concatenate([d[key] for d in dicts_list], axis=0)

    return combined_mask, combined_dict
