import time
import numpy as np
import awkward as ak
from scipy import ndimage
import multiprocessing as mp
from multifit import run_fit
import reco_utils

def generic_reco(waves, detector_name, **kwargs):

  globals().update(kwargs)

  t0 = time.time()

  with np.printoptions(threshold=np.inf):
    print("waves 1st event", waves[0])


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

  print("baselines: ", baselines[~mask_under_thr, None][0])

  if baseline_subtract:
    waves[~mask_under_thr, :] = waves[~mask_under_thr, :] - baselines[~mask_under_thr, None]

  with np.printoptions(threshold=np.inf):
    print("waves 1st event", waves[0])

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
    ieta, iphi = geo_dict["ieta"], geo_dict["iphi"]

    if do_5x5:
      charge_mean = np.mean(charge, axis=0)
      seed_ch = -999

      mask_5x5, seed_ch = reco_utils.find_5x5(charge_mean, ieta, iphi, fixed_5x5=fixed_5x5)
      print(f"find 5x5 took: {time.time() - t0}")
      t0 = time.time()


      peak_seed = values_max[:, seed_ch]

      charge_seed = charge[:, seed_ch]
      charge_sum_5x5 = np.sum(charge[:, mask_5x5], axis=1)
      charge_sum_5x5 = np.clip(charge_sum_5x5, seed_charge_threshold, None)

      mask_low_charge_seed = charge_seed > seed_charge_threshold

      # amplitude_map of the 5x5 matrix
      charge_fraction_5x5 = np.zeros(charge.shape)
      charge_fraction_5x5[:, mask_5x5] = charge[:, mask_5x5] / charge_sum_5x5[:, np.newaxis]

      print(f"seed/5x5/fractions took: {time.time() - t0}")
      t0 = time.time()

      w_log = np.maximum(0.0,w0_centroid + np.log(np.clip(charge_fraction_5x5, 1e-8, None)))
      w_log /= (np.sum(w_log, axis=1, keepdims=True))

      ieta_centroid = w_log[:, mask_5x5] @ ieta[mask_5x5]
      iphi_centroid = w_log[:, mask_5x5] @ iphi[mask_5x5]

      print(f"centrois took: {time.time() - t0}")
      t0 = time.time()


      iphi_within_5x5 = iphi - iphi[seed_ch]
      ieta_within_5x5 = ieta - ieta[seed_ch]

      ieta_within_5x5 = np.repeat(ieta_within_5x5[np.newaxis, :], charge.shape[0], axis=0)
      iphi_within_5x5 = np.repeat(iphi_within_5x5[np.newaxis, :], charge.shape[0], axis=0)

      print(f"ieta within 5x5 (+iphi) took: {time.time() - t0}")
      t0 = time.time()


      seed_ch_app = seed_ch
      seed_ch = np.repeat(np.ones(1,)*seed_ch, charge_sum_5x5.shape[0], axis=0)

      highest_ch = np.argmax(charge * mask_5x5, axis=1)
      highest_charge = np.take_along_axis(charge, highest_ch[:, None], axis=1).squeeze()
      highest_peak = np.take_along_axis(values_max, highest_ch[:, None], axis=1).squeeze()

      print(f"highest ch took: {time.time() - t0}")
      t0 = time.time()

      print(f"tau took: {time.time() - t0}")
      t0 = time.time()

      return_dict.update({
        f"{det}_charge_sum_5x5": charge_sum_5x5, f"{det}_charge_seed": charge_seed, f"{det}_peak_seed": peak_seed, f"{det}_seed_over_5x5": charge_fraction_5x5[:, seed_ch_app],
        f"{det}_highest_charge_over_5x5": highest_charge/charge_sum_5x5,
        f"{det}_iphi_within_5x5": iphi_within_5x5, f"{det}_ieta_within_5x5": ieta_within_5x5,
        f"{det}_charge_divided_5x5": charge_fraction_5x5, f"{det}_seed_ch": seed_ch,
        f"{det}_ieta_centroid": ieta_centroid, f"{det}_iphi_centroid": iphi_centroid,
        f"{det}_highest_ch": highest_ch, f"{det}_highest_charge": highest_charge, f"{det}_highest_peak": highest_peak,
      })

      #mask_selected_events = mask_low_charge_seed

    ieta = np.repeat(ieta[np.newaxis, :], charge.shape[0], axis=0)
    iphi = np.repeat(iphi[np.newaxis, :], charge.shape[0], axis=0)

  if do_tau:
      if do_5x5: tau_mask = mask_5x5
      else: tau_mask = np.full((signal_window.shape[1],), True)

      descent = signal_window[:, tau_mask, signal_samples_pre_peak+1:signal_samples_pre_peak+tau_descent_samples+1]
      log_w = np.log(np.clip(descent, 1, None))
      log_slopes = np.diff(log_w, axis=2) / descent.shape[2]
      tau = np.zeros(charge.shape)
      tau[:, tau_mask] = -1.0 / (1e-12 + np.median(log_slopes, axis=2) * sampling_rate)
      return_dict.update({f"{det}_tau": tau})

  if do_timing:
    if do_5x5: timing_mask = mask_5x5
    else: timing_mask = np.full((signal_window.shape[1],), True)

    timing_nch = int(np.sum(timing_mask))

    if timing_method == "cf" or timing_method == "fixed_thr":
      rise = signal_window[:, timing_mask, signal_samples_pre_peak - rise_samples_pre_peak:signal_samples_pre_peak + rise_samples_post_peak]
      print(rise.shape)
      rise_interp = ndimage.zoom(rise, [1, 1, interpolation_factor])

      with np.printoptions(threshold=np.inf):
        print(rise_interp[0])

      if timing_method == "cf":
        peak_interp = rise_interp.max(axis=2) #shape: (Events, Channel) - on y axis
        thresholds = peak_interp*cf

      if timing_method == "fixed_thr":
        thresholds = np.ones((rise.shape[0], rise.shape[1]))*timing_thr

      pseudo_t = np.zeros((signal_window.shape[0], signal_window.shape[1]))

      pseudo_t[:, timing_mask] = np.argmax(rise_interp > np.repeat((thresholds)[:, :, np.newaxis], rise_interp.shape[2], axis=2), axis=2).astype(float)
      print(pseudo_t[0])
      pseudo_t[:, timing_mask] += np.random.uniform(low=-0.5, high=0.5, size=(pseudo_t.shape[0], timing_nch))
      pseudo_t[:, timing_mask] /= float(sampling_rate*interpolation_factor)
      pseudo_t[:, timing_mask] += ((max_idx[:, timing_mask] - rise_samples_pre_peak) / sampling_rate)
      return_dict.update({f"{det}_{timing_method}_time": pseudo_t})

    elif timing_method == "lsfit":

      fit_amp, fit_time = run_fit(signal_window, mask_under_thr, max_idx, spline_file, sampling_rate, signal_samples_pre_peak)
      return_dict.update({f"{det}_{timing_method}_time": fit_time, f"{det}_{timing_method}_amp": fit_amp})

    else:
      raise NotImplemented(f"method: {timing_method} not implemented")

  per_ch_info = {
    f"{det}_peak_pos": max_idx, f"{det}_peak_time": max_idx/sampling_rate,
    f"{det}_charge": charge, f"{det}_peak": values_max, f"{det}_baseline_mean": baselines,
    f"{det}_baseline_std": baselines_std, f"{det}_baseline_integral": baseline_integral/baseline_samples*signal_window.shape[2],
  }

  if save_mean_rms_all_samples:
    per_ch_info.update({f"{det}_samples_mean": values_mean, f"{det}_samples_std": values_std})
  if geo_dict is not None:
    per_ch_info.update({f"{det}_ieta": ieta, f"{det}_iphi": iphi})
  if id is not None:
    for var in id:
      per_ch_info.update({f"{det}_{var}": np.repeat(id[var][np.newaxis, :], waves.shape[0], axis=0)})

  if do_5x5 and save_only_5x5_info:
    for key in per_ch_info:
      return_dict[key] = np.zeros(per_ch_info[key].shape)
      return_dict[key][:, mask_5x5] = per_ch_info[key][:, mask_5x5]
  else:
    return_dict.update(per_ch_info)

  return mask_selected_events, return_dict


def hodo_reco(tree, detector_name):
  det = detector_name
  reco_dict = {}
  coords_list = ["x1", "x2", "y1", "y2"]
  branches = tree.arrays(
    [f"{det}_{coord}_nclusters" for coord in coords_list] +
    [f"{det}_{coord}_pos" for coord in coords_list],
    library="ak"
  )
  mask_dict = np.ones(len(branches[f"{det}_{coords_list[0]}_nclusters"]), dtype=bool)
  for coord in coords_list:
    clus = branches[f"{det}_{coord}_nclusters"]
    pos = branches[f"{det}_{coord}_pos"]
    mask = (clus > 0)
    pos_first_cluster = ak.to_numpy(ak.where(mask, ak.firsts(pos), -999))
    mask_single_cluster = ak.to_numpy(clus == 1)

    safe_clus = ak.where(clus > 0, clus, 1)
    average_all_clusters = ak.to_numpy(
        ak.where(clus > 0, ak.sum(pos, axis=1) / safe_clus, -999.0)
    )

    reco_dict.update({
      f"{det}_{coord}_cl0_pos": pos_first_cluster,
      f"{det}_{coord}_single_cl_flag": mask_single_cluster,
      f"{det}_{coord}_avg_pos": average_all_clusters,
    })

  return mask_dict, reco_dict


def bcp_reco(bcp_clk, detector_name):
  det = detector_name
  reco_dict = {}
  mask = np.ones((bcp_clk.shape[0],), dtype=bool)
  bcp_clk = bcp_clk.astype(np.int64)
  bcp1_clk = bcp_clk[:, 0, :]
  bcp2_clk = bcp_clk[:, 1, :]
  bcp1_clk_mean = np.tile(np.mean(bcp1_clk, axis=0), (bcp1_clk.shape[0], 1))
  bcp2_clk_mean = np.tile(np.mean(bcp2_clk, axis=0), (bcp2_clk.shape[0], 1))
  reco_dict.update({f"{det}1_clk": bcp1_clk, f"{det}2_clk": bcp2_clk,
    f"{det}1_clk_mean": bcp1_clk_mean.astype(int), f"{det}2_clk_mean": bcp2_clk_mean.astype(int)
  })

  return mask, reco_dict


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
