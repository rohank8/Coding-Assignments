#!/usr/bin/env python
# coding: utf-8

# In[24]:


#first import packages we need
import matplotlib.pyplot as plt
import numpy as np

#making sure to import the data file that we need

with open('planets.csv') as f:
    planets = f.readlines()
    
#part (a)

planets = [string.strip('\n') for string in planets if string.startswith('#') == False]
planets = [string.split(',') for string in planets]

#using pythons in built string methods and doing the same as what we did in the first lab by exluding lines starting with
#now we take the max and min years for and each item in the lists, the max is the oldest and the min is the youngest

yrs = {int(entry[2]) for entry in planets[1:]}
yrs = list(yrs) 

#the same code as in lab 1 to yield list of planets 

min_yr = int(min(yrs))
max_yr = int(max(yrs))

for a in range(min_yr, max_yr):

    if a not in yrs:
        yrs.append(int(a))
        
yrs.sort(key = lambda x: x)
nod = len(yrs) 

#use the simple sort function on python to sort the elements of the list in chronological order as we did in lab 1
#nod is the number of years that planets have been discovered, we 'return' the length of the years using the len function

Nyr = [0]*nod

for entry in planets[1:]:
    ind = yrs.index(int(entry[2])) 
    Nyr[ind] += 1
    
#same code as in lab 1 to count and increment the count

plt.plot(yrs, Nyr, 'm.-', markersize = 9)

plt.title('Number of planets discovered each year', fontweight = 'bold', size = 14)

plt.xlabel('Year of discovery', size = 14)

plt.ylabel('Number of planets discovered', size = 14)

plt.grid()

plt.show()


# In[25]:


mass_of_planet = [float(entry[5]) for entry in planets[1:] if entry[5] != '' and entry[6] != '']

stellar_mass = [float(entry[6]) for entry in planets[1:] if entry[5] != '' and entry[6] != '']

# similar code used in lab 1

plt.scatter(mass_of_planet, stellar_mass, color = 'purple', s=2)

plt.title('Mass of Planet vs Stellar mass', size = 13, fontweight = 'bold')

plt.xscale('log')

plt.yscale('log')

plt.xlabel('Planet mass', size = 13)

plt.ylabel('Stellar mass', size = 13)

plt.grid()

plt.show()


# In[26]:


#part c
facilities = {entry[3] for entry in planets[1:] if entry[3] != ''}
facilities = list(facilities) 
#using same code as part (b) to list out different facilities
numberOf_facilities = len(facilities) 
observations = [0]*numberOf_facilities
#simple code to get number of different facilities and 'length'
first_discovery = [3000]*numberOf_facilities
for entry in planets[1:]:
    ind = facilities.index(entry[3])
    observations[ind] += 1
    year = int(entry[2])
    if int(year) < int(first_discovery[ind]):
        first_discovery[ind] = year
#if statement describing what we want
ind = observations.index(max(observations))
most_discovery = observations[ind] 
most_facility = facilities[ind] 
#getting facilities with most discoveries
mainlist = [(facilities[a], observations[a], first_discovery[a]) for a in range(len(facilities))]
mainlist.sort(key = lambda x: x[1], reverse=True) 
#creating list we actually want
Top10 = [main_list[a] for a in range(10)]
Top10.sort(key = lambda x: x[2])
#organising the list now
print('%-50s %-30s %-20s' %('Name of facility', 'Number of discoveries made', 'Year of discovery'))
for entry in Top10: 
    print('%-50s %-30i %-20i' %entry)
print('\nThe facility with the most planet observations is %s, with %i discovered planets' %(most_facility, most_discovery))
mainlist.sort(key = lambda x: x[2]) 


# In[ ]:




