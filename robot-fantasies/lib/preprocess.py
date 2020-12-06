import numpy as np
import pickle

# go through each surfer's stats to find their median event points
def median_points(experience_multiplier=0):
	athletes = pickle.load(open('../data/pickles/athletes.p','rb'))
	names = list(athletes.keys())
	med_points = []
	for name in names:
		points = []
		athlete = athletes[name]
		years = athlete.keys()
		for year in years:
			point = athlete[year]['points']
			points += point
		points = [ p for p in points if type(p)==int ]
		if len(points) == 0 : points = [1]
		med_points.append(np.median(points)*(1+experience_multiplier*len(points)))
	return(names,med_points)