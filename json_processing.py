import pandas as pd
import json


# converting JSON file to a pandas dataframe
def json_to_df(file):
    data = pd.DataFrame()
    with open(file) as train_file:
        dict_train = json.load(train_file)

    # converting json dataset from dictionary to dataframe
    for i in range(len(dict_train)):
        data = data.append(pd.DataFrame.from_dict(
            dict_train[i]), ignore_index=True)
    column_mask = data.isnull().mean(axis=0) < 1
    return data.loc[:, column_mask]


def remove_rows_based_on(data, factor):
    data = data[~data.link.str.contains(factor)]
    return data


if __name__ == '__main__':
    df = json_to_df('sublinks.json')
    df = remove_rows_based_on(df, 'https')
    df = remove_rows_based_on(df, '#')
    df.to_csv('sublinks.csv', index=False)
