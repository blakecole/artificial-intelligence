from preprocess import median_points
from optimize import gamma_slide

# get the names and predicted values for each surfer
names,v = median_points(experience_multiplier=0)

# (for now) enter prices by hand
costs = {
	'adrian-buchan':6,
	'adriano-de-souza':4.5,
	'alex-ribeiro':3,
	'caio-ibelli':6.5,
	'conner-coffin':6,
	'connor-oleary':3,
	'deivid-silva':3,
	'ethan-ewing':3,
	'filipe-toledo':11,
	'frederico-morais':5,
	'gabriel-medina':12,
	'griffin-colapinto':7,
	'italo-ferreira':12.5,
	'jack-freestone':7,
	'jack-robinson':3.5,
	'jadson-andre':4.5,
	'jeremy-flores':8.5,
	'john-john-florence':9.5,
	'jordy-smith':11.5,
	'julian-wilson':8,
	'kanoa-igarashi':10,
	'kelly-slater':9,
	'kolohe-andino':10.5,
	'leonardo-fioravanti':500000,  # not currently available on fs, take out of running
	'matthew-mcgillivray':4,
	'michel-bourez':7.5,
	'miguel-pupo':3,
	'mikey-wright':50000000, # not currently available on fs, take out of running
	'morgan-cibilic':3,
	'owen-wright':8.5,
	'peterson-crisanto':5.5,
	'ryan-callinan':7.5,
	'seth-moniz':8,
	'wade-carmichael':6.5,
	'yago-dora':5.5
}

# create properly ordered cost vector
c = [costs[name] for name in names]

# use sliding gamma heuristic
gamma_slide(names,v,c,verbose=True)