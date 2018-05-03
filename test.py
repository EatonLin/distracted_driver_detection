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


def add_driver_classname_count(classname, count):
    dict = {}
    dict[classname] = count
    return dict


def add_driver_subject_information(subject_dict, subject, class_dict):
    subject_dict[subject] = class_dict

for index, row in data.iterrows():
    if subject_temp is not row["subject"]:
        if driver_classname_count > 1:
            print("subject:{} classname:{} count:{}".format(subject_temp, classname_temp, driver_classname_count))
            classname_dict = add_driver_classname_count(classname_temp, driver_classname_count)
            add_driver_subject_information(subject_dict, subject_temp, classname_dict)
            driver_classname_count = 0

        if driver_count != 0:
            print("subject:{} count:{}".format(subject_temp, driver_count))
            # add_driver_subject_information(subject_dict, subject_temp, classname_dict)
        subject_temp = row["subject"]
        driver_count = 1
        drivers_count += 1
    else:
        driver_count += 1
        if classname_temp is not row["classname"]:
            if driver_classname_count != 0:
                print("subject:{} classname:{} count:{}".format(subject_temp, classname_temp, driver_classname_count))
            classname_temp = row["classname"]
            driver_classname_count = 1
        else:
            driver_classname_count += 1





print("Drivers dict:", subject_dict)