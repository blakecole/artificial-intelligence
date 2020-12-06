import urllib.request
from html.parser import HTMLParser
import os
import sys
import pprint
import pickle


# generates list of athlete urls from main wsl athlete site
class RosterParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.roster = []

    def handle_starttag(self, tag, attrs):
        if (tag == 'a'):
            for (name,val) in attrs:
                if name == 'href':
                    self.roster.append(val)

# finds the years that each athlete competed from the athlete urls
class YearParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.years = []

    def handle_starttag(self, tag, attrs):
        if (tag == 'option'):
            for (name,val) in attrs:
                if name == 'value' and all(char.isdigit() for char in val):
                    self.years.append(val)

# gets the contests and point totals for a given year/athlete pair
class StatParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.roster = []
        self.get_event = False
        self.get_point = False
        self.events = []
        self.points = []

    def handle_starttag(self, tag, attrs):
        if (tag == 'a'):
            for (name,val) in attrs:
                if (name == 'href') and ('eventId' in val):
                    self.get_event = True
        elif (tag == 'td'):
            for (name,val) in attrs:
                if (name == 'class') and (val == 'event-athlete-points stat'):
                    self.get_point = True

    def handle_data(self,data):
        if self.get_event:
            self.events.append(data)
            self.get_event = False
        if self.get_point:
            # be ready for missing values
            try: self.points.append(int(data.replace(',','')))
            except: self.points.append('TBD')
            self.get_point = False


# create a RosterParser and feeds it
def get_roster(verbose=True):
    roster_url = 'https://www.worldsurfleague.com/athletes?tourIds[]=1'
    openurl = urllib.request.urlopen(roster_url)
    roster_html = str(openurl.read())
    roster_parser = RosterParser()
    roster_parser.feed(roster_html)
    roster = roster_parser.roster
    # do a little bit of clean up of the roster
    r = []
    for url in roster:
        if ('athlete' in url) and (url not in r) and (any(char.isdigit() for char in url)):
            r.append(url)
    roster = r
    pp = pprint.PrettyPrinter()
    if verbose: pp.pprint(roster)
    return roster

# loop through an athletes years to get their stats
def get_stats(athlete_url,verbose=True):
    openurl = urllib.request.urlopen(athlete_url)
    athlete_html = str(openurl.read())
    year_parser = YearParser()
    year_parser.feed(athlete_html)
    years = year_parser.years
    # dictionary to hold results by year
    stats = {}
       
    for year in years:
        if verbose: print('YEAR: %s'%year)
        openurl = urllib.request.urlopen(athlete_url+'?yearResultsTourCode=mct&yearResultsYear=%s'%year)
        athlete_html = str(openurl.read())
        stat_parser = StatParser()
        stat_parser.feed(athlete_html)
        events = stat_parser.events
        points = stat_parser.points
        if verbose: print(events)
        if verbose: print(points)
        stats[year] = {'events':events,'points':points}

    return stats

#########
# MAIN
#########

# get the list of athlete urls
roster = get_roster()
# create empty dict
athletes = {} 
for url in roster:
    # get the athlete name
    name = url.split('/')[-1]
    print('----------')
    print(name)
    stats = get_stats(url)
    athletes[name] = stats
pickle.dump( athletes, open( "../data/pickles/athletes.p", "wb" ) )

