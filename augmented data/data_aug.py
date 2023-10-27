import pandas as pd

# 读取数据
data = pd.read_csv('train.csv')

# 创建一个新的DataFrame存储训练数据
train_data = pd.DataFrame()
train_data['Argument ID'] = ["A" + str(i).zfill(5) for i in range(1, len(data) + 1)]
train_data['Conclusion'] = ['UNKNOWN' for _ in range(len(data))]  # 由于原始数据中没有提供Conclusion，所以此处设为UNKNOWN
train_data['Stance'] = data['label'].map({1: 'in favor of', -1: 'against'})
train_data['Premise'] = data['scenario']

# 创建一个新的DataFrame存储训练标签
value_list = ["Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
              "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal",
              "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility",
              "Benevolence: caring", "Benevolence: dependability", "Universalism: concern",
              "Universalism: nature", "Universalism: tolerance", "Universalism: objectivity"]

train_label = pd.DataFrame()
train_label['Argument ID'] = train_data['Argument ID']

# 初始化所有值为0
for value in value_list:
    train_label[value] = 0

# 设置相应的标签为1
for idx, row in data.iterrows():
    label = row['scenario'].split(' ')[0].replace('[', '').replace(']', '').strip().lower()
    # 使用if-elif逻辑来确定哪个列需要设置为1
    # 这里只为Power和Benevolence添加了例子，你可以按需扩展这部分逻辑
    if 'power' in label:
        train_label.at[idx, "Power: dominance"] = 1
    if 'benevolence' in label:
        train_label.at[idx, "Benevolence: caring"] = 1

# 保存到CSV文件中
train_data.to_csv('train_data.csv', index=False)
train_label.to_csv('train_label.csv', index=False)
