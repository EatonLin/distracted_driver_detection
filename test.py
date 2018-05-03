import pandas as pd

try:
    data = pd.read_csv("driver_imgs_list.csv")
    print("Load dataset:{} {}".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")

subject_dict = {}
subject_temp = ""
classname_temp = ""
driver_count = 0
driver_classname_count = 0
drivers_count = 0


def add_driver_subject_information(dict, subject, classname, count):
    if subject in dict:
        dict[subject].append(count)
    else:
        dict[subject] = [count]


for index, row in data.iterrows():
    if subject_temp is not row["subject"]:
        if driver_classname_count > 1:
            add_driver_subject_information(subject_dict, subject_temp, classname_temp, driver_classname_count)
            driver_classname_count = 0

        # if driver_count != 0:
        #     add_driver_subject_information(subject_dict, subject_temp, classname_temp, driver_classname_count)
        subject_temp = row["subject"]
        driver_count = 1
        drivers_count += 1
    else:
        driver_count += 1
        if classname_temp is not row["classname"]:
            if driver_classname_count != 0:
                add_driver_subject_information(subject_dict, subject_temp, classname_temp, driver_classname_count)
            classname_temp = row["classname"]
            driver_classname_count = 1
        else:
            driver_classname_count += 1

subject_dict[subject_temp].append(driver_classname_count)
new_dict = [(key, subject_dict[key]) for key in sorted(subject_dict.keys())]

import numpy as np
import matplotlib.pyplot as plt

n = 10
x = np.arange(n)
fig, ax = plt.subplots()
bar_width = 0.3
opacity = 0.4
index = 0

color_list = ['r', 'b', 'c', 'g', 'k', 'm', 'y', 'r', 'b', 'c']

for item in new_dict:
    plt.bar(x + index * bar_width, item[1], bar_width, alpha=opacity, color='r', label=item[0])

plt.legend()
plt.show()
