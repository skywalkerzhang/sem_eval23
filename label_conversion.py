import pandas as pd

# 映射关系
mapping = {
    "Openness to change": ["Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism"],
    "Self-enhancement": ["Achievement", "Power: dominance", "Power: resources", "Face"],
    "Conservation": ["Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility"],
    "Self-transcendence": ["Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance", "Universalism: objectivity"]
}

# 读取labels.csv文件
df = pd.read_csv("data/labels-training.tsv", sep='\t')

# 为每个父类标签创建一个新列，并使用映射关系为这些列赋值
for parent_label, sub_labels in mapping.items():
    df[parent_label] = df[sub_labels].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# 只保留"Argument ID"和父类标签列
df = df[["Argument ID"] + list(mapping.keys())]

# 保存新的父类标签到csv文件
df.to_csv("data/level2-processed-labels-training.tsv", index=False, sep='\t')
