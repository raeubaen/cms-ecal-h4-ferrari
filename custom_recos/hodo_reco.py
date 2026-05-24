from registry import register_reco
import numpy as np
import awkward as ak

@register_reco("hodo")
def hodo_reco(tree, detector_name, detector_dict_piece):
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


