import pandas as pd
import sys

input_file = sys.argv[-1]


result = pd.read_csv(input_file, index_col=None)

#result = result.head(5)

ans = result.apply(lambda row : ' '.join(row["place_id"].split(' ')[0:3]), axis=1)

result["place_id"] = ans

result.to_csv("0623.csv", index=False)
