"""
    Combines all features together from all modalities 
"""

import numpy as np 

modals = ["/share/workhorse3/mahmoudi/courses/11776/11776_Intervening_before_its_too_late/speech/speech_features/audio_feats.npz",
          "/share/workhorse3/mahmoudi/courses/11776/11776_Intervening_before_its_too_late/language/language_data/all_language_features.npz",
          "/share/workhorse3/mahmoudi/courses/11776/11776_Intervening_before_its_too_late/vision/visual_data/pose.npz", 
          "/share/workhorse3/mahmoudi/temp/openface.npz"]

labels = {75: 0, 77: 0, 12: 1, 49: 0, 7: 1, 50: 0, 39: 1, 20: 1, 23: 0, 76: 1, 58: 0, 64: 0, 34: 0, 8: 0, 71: 0, 37: 1, 29: 0, 74: 1, 3: 0, 45: 1, 40: 1, 48: 1, 2: 0, 62: 1, 68: 0, 81: 0, 82: 0, 54: 0, 69: 0, 67: 1, 5: 1, 30: 1, 46: 0, 59: 1, 70: 0, 72: 1, 10: 0, 87: 0, 52: 0, 32: 0, 33: 1, 41: 0, 90: 0, 55: 0, 85: 0, 11: 0, 21: 0, 44: 0, 89: 0, 79: 1, 31: 0, 14: 0, 27: 1, 4: 0, 57: 0, 25: 1, 51: 1, 47: 0, 88: 0, 43: 0, 9: 0, 73: 1, 84: 1, 18: 0, 38: 1, 35: 0, 78: 1, 13: 0, 28: 0, 80: 0, 42: 1, 36: 1, 56: 0, 16: 0, 6: 0, 1: 0, 83: 1, 66: 0, 17: 0, 60: 0, 26: 0, 61: 0, 53: 0, 22: 0, 63: 0, 65: 1, 15: 0, 19: 1, 24: 0, 86: 0}

ALL_FEATURES = {}

for path in modals: 
    print("Reading: {}".format(path))
    x = np.load(path).items()[0][1].item()
    for key in x.keys():
        # cheap tricks because pose is not clean 
        try:
            int(key)
        except:
            continue

        if int(key) not in ALL_FEATURES:
            ALL_FEATURES[int(key)] = x[key]
        else:
            if path.split("/")[-1] == "openface.npz":
                ALL_FEATURES[int(key)]["openface"] = x[key]
            elif path.split("/")[-1] == "pose.npz":
                vec = list(map(float, x[key].strip().split(",")))
                ALL_FEATURES[int(key)]["pose"] = vec
            else:
                ALL_FEATURES[int(key)].update(x[key])

    # print(ALL_FEATURES)
     
# add the labels 
for key in ALL_FEATURES.keys():
    ALL_FEATURES[key]["label"] = labels[key]
 
print(ALL_FEATURES.keys()) 
np.savez("all_features.npz", ALL_FEATURES)
            
