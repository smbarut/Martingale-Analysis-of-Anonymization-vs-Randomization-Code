import pandas as pd
import numpy as np


def load_data(file_path):

    data = pd.read_csv(file_path)
    return data

def see_labels(data):

    label_column = 'labels' 
    set_labels = set(data[label_column])
    print(f"Unique labels in the dataset: {set_labels}")
    print("Column names in the dataset:")
    print(data.columns.tolist())

def names(data):
    names = []
    for i in range(len(data)):
        names.append(data.loc[i, "name"])
    names = list(set(names))
    return names

def phones(data):
    phones = []
    for i in range(len(data)):
        phones.append(data.loc[i, "phone"])
    phones = list(set(phones))
    return phones

def emails(data):
    emails = []
    for i in range(len(data)):
        emails.append(data.loc[i, "email"])
    emails = list(set(emails))
    return emails

def data_preparer(i, random_data):
    original_text = data.loc[i, "text"]
    name = data.loc[i, "name"]
    phone = data.loc[i, "phone"]
    email = data.loc[i, "email"]

    name_anonymized = original_text.replace(name, "[NAME]")
    phone_anonymized = name_anonymized.replace(phone, "[PHONE]")
    email_anonymized = phone_anonymized.replace(email, "[EMAIL]")

    random_name = random_data[0]
    random_phone = random_data[1]
    random_email = random_data[2]

    name_randomized = original_text.replace(name, random_name)
    phone_randomized = name_randomized.replace(phone, random_phone)
    email_randomized = phone_randomized.replace(email, random_email)

    return name_anonymized, phone_anonymized, email_anonymized, name_randomized, phone_randomized, email_randomized





if __name__ == "__main__":
    file_path = 'data.csv'
    data = load_data(file_path)
    print(names(data))




