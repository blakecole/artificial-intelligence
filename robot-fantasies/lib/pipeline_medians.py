# ********************************************************** #
#    NAME: Blake Cole + Gabe Schamberg                       #
#    ORGN: MIT                                               #
#    FILE: pipeline_medians.py                               #
#    DATE: 06 DEC 2020                                       #
# ********************************************************** #

# Revision History
# Date         |  User  | Comment
# =========================================================
# 2020 DEC 06  | blerk  | created.

import numpy as np
import pickle
import scipy.io
from preprocess import median_points

athletes = pickle.load(open('../data/pickles/athletes.p','rb'))
names = list(athletes.keys())  ## list of names [strings]
nicknames = ["pipeline masters", "pipe masters"]

pipeline_medians = {}
pipeline_means = {}
pipeline_max = {}
for name in names:
    pipeline_points = []
    athlete = athletes[name]
    years = athlete.keys()

    for year in years:
        events = athlete[year]['events']

        idx = 0
        for event in events:
            # print('EVENT = %s' %event.lower())

            if any(nickname in event.lower() for nickname in nicknames):
                p = athlete[year]['points'][idx]
                # print(p)
                if ((type(p)==int) and (p!="TBD")):
                    pipeline_points.append(p)
                    # print('POINTS = %s' %pipeline_points)
            idx += 1

    # handle newbies
    if not pipeline_points:
        pipeline_points = -1
            
    pipeline_medians[name] = np.median(pipeline_points)
    pipeline_means[name] = np.round(np.mean(pipeline_points),decimals=2)
    pipeline_max[name] = np.max(pipeline_points)

print('\n\n MEDIAN PIPELINE PERFORMANCE:\n')
[print('%s : %s' %(key, value)) for key, value in pipeline_medians.items()]

print('\n\n MEAN PIPELINE PERFORMANCE:\n')
[print('%s : %s' %(key, value)) for key, value in pipeline_means.items()]

print('\n\n BEST PIPELINE PERFORMANCE:\n')
[print('%s : %s' %(key, value)) for key, value in pipeline_max.items()]


# grab season stats from gabe
season_names,season_medians = median_points(experience_multiplier=0)

# export for use in MATLAB
scipy.io.savemat('../data/pipe_medians.mat', dict(x=list(pipeline_medians.keys()), y=list(pipeline_medians.values())))

scipy.io.savemat('../data/pipe_means.mat', dict(x=list(pipeline_means.keys()), y=list(pipeline_means.values())))

scipy.io.savemat('../data/pipe_max.mat', dict(x=list(pipeline_max.keys()), y=list(pipeline_max.values())))

scipy.io.savemat('../data/season_medians.mat', dict(x=season_names, y=season_medians))
