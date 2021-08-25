import numpy as np

"""
Observation: We can use the modulo operation to find a match.
We start at our current time, check all bus ids, then increase time by 1, until we find the first zero-remainder.
"""

#Original example
#timestamp = 939
#ids = np.array([7,13,59,31,19])
solution_found = False

#Exercise 1
timestamp = 1000677
string = "29,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,41,x,x,x,x,x,x,x,x,x,661,x,x,x,x,x,x,x,x,x,x,x,x,13,17,x,x,x,x,x,x,x,x,23,x,x,x,x,x,x,x,521,x,x,x,x,x,37,x,x,x,x,x,x,x,x,x,x,x,x,19"
string = string.replace(",x","")
ids = np.fromstring(string,dtype=int, sep=",")
iterated_timestamp = timestamp

while( not solution_found):
    for id in ids:
        if (iterated_timestamp % id) == 0:
            print("Bus id: ", id, ", is the most efficient leaving at: ", iterated_timestamp)
            print("\nSolution: ", id*(iterated_timestamp-timestamp))
            solution_found = True
        else:
            continue
    iterated_timestamp += 1
        
