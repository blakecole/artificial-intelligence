# ********************************************************** #
#    NAME: Blake Cole + Gabe Schamberg                       #
#    ORGN: MIT                                               #
#    FILE: scrape.py                                         #
#    DATE: 03 DEC 2020                                       #
# ********************************************************** #

# Revision History
# Date         |  User  | Comment
# =========================================================
# 2020 DEC 05  | grabe  | gen-0 robot fully functional.
# 2020 DEC 07  | blerk  | scrape added: fantasy surfer cost


import urllib.request
from html.parser import HTMLParser
import os
import sys
import pprint
import pickle
import pandas as pd

import requests
import re

# General notes on HTML processing methodology:
#  (1) a URL is provided by the user.
#  (2) a Parser subclass inherits the HTMLParser superclass.
#  (3) each Parser class is designed to identify specific
#      features in the HTML code found at the URL.
#  (4) the parser loops through EVERY DANG HTML tag, each of
#      which has a set of attributes, each of which has a
#      (name,attribute) pair.
#  (5) thus, the user can specify
#       --> a tag (e.g. 'a' for links, 'td' for tables...)
#           --> an attriute name belonging to the tag
#       and finally, extract the attribute 'value'

#  for more info, see:
#  https://docs.python.org/3/library/html.parser.html

# ---------------------------------------------------------
# DEFINE CLASSES ------------------------------------------
# ---------------------------------------------------------
# generates list of athlete urls from main wsl athlete site
class RosterParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.roster = []

    def handle_starttag(self, tag, attrs):
        # the <a> tag defines a hyperlink
        if (tag == 'a'):
            
            # 'href' is an attribute of <a>
            # href defines the link destination
            for (name,val) in attrs:
                if name == 'href':
                    self.roster.append(val)

                    
# finds the years that each athlete competed from the athlete urls
class YearParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.years = []

    def handle_starttag(self, tag, attrs):
        # the <option> tag defines an option in a dropdown list
        if (tag == 'option'):

            # 'value' is an attribute of <option>
            # value defines a selectable option in a dropdown list
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
        # the <a> tag defines a hyperlink
        if (tag == 'a'):
            for (name,val) in attrs:

                # this time, we are not interested in all hyperlinks
                # we only want those which correspond to an event
                if (name == 'href') and ('eventId' in val):
                    self.get_event = True

        # the <td> tag defines a standard data cell in an HTML table
        elif (tag == 'td'):
            for (name,val) in attrs:

                # WSL gets to define each 'class' in the table
                # in this case, we are interested in points at each event
                # this class is named: 'event-athlete-points stat'
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

            
class CostParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.cost = []
        self.get_data = False
        self.get_cost = False

    def handle_starttag(self, tag, attrs):
        if (tag == 'script'):
            for (name, val) in attrs:
                if(name == 'type') and (val == 'text/javascript'):
                    self.get_data = True
            

    def handle_data(self,data):
        if self.get_data:
            try:
                clean = re.search("var athletes = '(.+?)';", data).group(1)
                clean_dic = eval(clean)
                self.cost = clean_dic
            except AttributeError:
                pass # do nothing: var athletes not found
            self.get_data = False
            
# ---------------------------------------------------------
# DEFINE FUNCTIONS  ---------------------------------------
# ---------------------------------------------------------
# create a RosterParser and feeds it
def get_roster(verbose=True):
    roster_url = 'https://www.worldsurfleague.com/athletes?tourIds[]=1'
    openurl = urllib.request.urlopen(roster_url)    # open althletes webpage
    roster_html = str(openurl.read())               # read webpage html as string
    roster_parser = RosterParser()                  # initialize RosterParser obj
    roster_parser.feed(roster_html)                 # feed html to parser
    roster = roster_parser.roster                   # string list of all WSL urls
    
    # do a little bit of clean up of the roster (eliminate all non-roster urls)
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
    openurl = urllib.request.urlopen(athlete_url)    # open inividual athlete page
    athlete_html = str(openurl.read())               # read webpage html as string
    year_parser = YearParser()                       # initialize YearParser obj
    year_parser.feed(athlete_html)                   # scan years in dropdown menu
    years = year_parser.years                        # string list of athlete comp years
    # dictionary to hold results by year
    stats = {}
       
    for year in years:
        if verbose: print('YEAR: %s'%year)
        openurl = urllib.request.urlopen(athlete_url+'?yearResultsTourCode=mct&yearResultsYear=%s'%year)
        athlete_html = str(openurl.read())    # open indvidual athelete-year page
        stat_parser = StatParser()
        stat_parser.feed(athlete_html)
        events = stat_parser.events
        points = stat_parser.points
        if verbose: print(events)
        if verbose: print(points)
        stats[year] = {'events':events,'points':points}

    return stats


# get surfer cost data from fantasy surfer
def get_cost(verbose=True):
    # define login info
    username = "robot_fantasies"
    password = "FSbluntmasters69"
    base_url = "https://fantasy.surfer.com/"
    login_url = base_url + "/login/"
    target_url = base_url + "team/mens"

    # start session
    session = requests.Session()
    agent = {"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '\
                            +'(KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    login_data = {"password" : "1166256fdeb254e2df7fb196964d6c323ba45f1d",
                  "username" : username,
                  "legacy_password" : password,
                  "persistent" : "on",
                  "submit" : "Login"
    }

    # log in and scrape
    s = session.post(login_url, data=login_data)
    s = session.get(target_url, headers=agent, allow_redirects=False)
    cost_html = s.text

    # parse
    cost_parser = CostParser()
    cost_parser.feed(cost_html)
    cost = pd.DataFrame(cost_parser.cost)

    # delete the thumbnail URL
    cost.drop("thumbnail", axis=1, inplace=True)
    print(cost)

    return(cost)
    

#########
# MAIN
#########
"""
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
"""
cost = get_cost(verbose=True)

# no need to save while experimenting
# pickle.dump( athletes, open( "../data/pickles/athletes.p", "wb" ) )

