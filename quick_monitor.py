import json, numpy as np
pred = json.load(open(r"F:\OpenTAD\work_dirs\e2e_pku_mmd_videomae_s_768x1_160_adapter\gpu1_id0\result_detection_fixed.json"))
vid, first = next(iter(pred["results"].items()))
print("pred seg:", first[0]["segment"][:2])
print("pred label idx:", first[0]["label"])