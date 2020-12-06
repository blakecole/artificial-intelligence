import numpy as np
from preprocess import median_points

def gamma_slide(names,v,c,max_budget=50,max_team=8,step=0.001,verbose=True):
	last_val = 0
	val = 0
	max_val = 0
	gamma = 0
	while val >= last_val:
		last_val = val
		val = 0
		g = [(v[i]**gamma)/c[i] for i in range(len(names))]
		i_sort = np.argsort(g)[-max_team:]
		budget = max_budget
		team = []
		for i in i_sort: 
			if budget - c[i] < 0:
				break
			else:
				val += v[i]
				budget -= c[i]
				team.append(names[i])
		if val > max_val:
			best_team = team
			max_val = val
			if verbose:
				print("------ NEW BEST TEAM ------")
				print("Team: " + str(team))
				print("Predicted Point Total: " + str(val))
		gamma += step
	return team