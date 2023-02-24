import pandas as pd

image_data = pd.read_csv('unprocessed_data/image_data.csv').to_numpy()
oh_groups = pd.read_csv('unprocessed_data/oh_groups.csv').to_numpy()

dataset = pd.merge(pd.DataFrame(image_data), pd.DataFrame(oh_groups), on=0, how='outer')
dataset = dataset.iloc[:, 1:]
dataset.to_csv('dataset.csv', index=False, header=False)

print(image_data.shape)
print(oh_groups.shape)

print(dataset.shape)
