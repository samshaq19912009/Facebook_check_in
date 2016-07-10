import pandas as pd
import datetime
import sys
import collections

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
file4 = sys.argv[4]
file5 = sys.argv[5]

sol1 = pd.read_csv(file1)
sol2 = pd.read_csv(file2)
sol3 = pd.read_csv(file3)
sol4 = pd.read_csv(file4)
sol5 = pd.read_csv(file5)

final = sol1

final["place_id_1"] = sol1['place_id']
final["place_id_2"] = sol2['place_id']
final["place_id_3"] = sol3['place_id']
final["place_id_4"] = sol4['place_id']
final["place_id_5"] = sol4['place_id']

def ensemble(x):
    # each place_id is a 5 top place_id guess
    # yes, I am using map@5 instead of map@3.
    place_id_1 = x.place_id_1.split(" ")
    place_id_2 = x.place_id_2.split(" ")
    place_id_3 = x.place_id_3.split(" ")
    place_id_4 = x.place_id_4.split(" ")
    place_id_5 = x.place_id_5.split(" ")

    # place_id_1 has the best prediction, place_id_5 has the worst prediction
    output =  [(place_id_1[i]+" ")*(6-i) for i in range(len(place_id_1))] 
    output += [(place_id_2[i]+" ")*(5-i) for i in range(len(place_id_2))]
    output += [(place_id_3[i]+" ")*(5-i) for i in range(len(place_id_3))]
    output += [(place_id_4[i]+" ")*(4-i) for i in range(len(place_id_4))]
    output += [(place_id_5[i]+" ")*(3-i) for i in range(len(place_id_5))]
    output = "".join(output).split()

    m = collections.Counter(output) # this Counter method is to count word frequency.
    #return output
    return " ".join([x[0] for x in m.most_common(3)])

final['place_id'] = final.apply(lambda x: ensemble(x), axis=1)


sub = final.drop(["place_id_1","place_id_2","place_id_3", "place_id_4","place_id_5"], axis=1)

print sub.head(5)

now = datetime.datetime.now()

file_to_sub = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'



sub.to_csv(file_to_sub, index=False)

print "Done"
