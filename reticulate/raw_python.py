import pandas
flpy = pandas.read_csv("flights.csv")
flpy = flpy[flpy['dest'] == "ORD"]
flpy = flpy[['carrier', 'arr_delay', 'dep_delay']]
print(flpy.head())
