from registry import register_reco
import numpy as np


@register_reco("decode_mem")
def decode_mem(waves, gain_list):
    bit13_mask = 1 << 13 #validity bit
    bit12_mask = 1 << 12 #gain bit
    amp_mask   = 0x0FFF #amplitude mask
    is_valid = (waves & bit13_mask) != 0
    gain_is_high = (waves & bit12_mask) != 0
    amplitudes = (waves & amp_mask).astype(np.float32)


    if gain_list is not None:
        gains = gain_list[None, :, None]
        amplitudes *= np.where(gain_is_high, gains, 1)
    #amplitudes[~is_valid] = 0
    return amplitudes
