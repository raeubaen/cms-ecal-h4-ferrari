from ferrari_core.registry import register_routine
import numpy as np

@register_routine("bcp")
def bcp_reco(tree, detector_name, detector_dict_piece):

  bcp_clk = tree["bcp_clk"].array(library="np")

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




