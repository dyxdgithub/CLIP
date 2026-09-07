import fiftyone as fo

# 查看当前所有的数据集名称
print(fo.list_datasets())

# 删除指定的数据集（替换为你的数据集名称）
# fo.delete_dataset("your_dataset_name")
for dataset in fo.list_datasets():
    fo.delete_dataset(dataset)