# robot-fantasies
Ongoing project to automate choosing fantasy surfing team (http://fantasy.surfermag.com)

Scraped data (pickled in a `dict`) can be found in `data/pickles/athletes.p`. In `lib/` you'll find `scrape.py` for generating a dict, `preprocess.py` for processing scraped data to predict each surfers value, `optimize.py` for picking surfers based on their predicted values, cost, and FS budget, and `pick_team.py` for running the preprocessing and optimization steps.
