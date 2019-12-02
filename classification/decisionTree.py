import pandas
from math import log


def entopy_cal(no_outcomes):
    entropy = 0
    for i in no_outcomes:
        temp = -(i / sum(no_outcomes) * log(i / sum(no_outcomes), 2))
        entropy += temp

    return entropy


# df = pandas.DataFrame.from_csv(r"/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/sample.csv")
# print(sum([9, 5]))
# print(-9/14*log(9/14,2)-5/14*log(5/14, 2))


print(entopy_cal([9, 5, 5]))
