import os
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import pandas as pd
from datasets import Dataset

# 本地数据集路径
GLUE_DIR = r"D:\AboutCode\Python\Pycharm\Code\ALBERT\glue_data"
model_path = r"D:\AboutCode\Python\Pycharm\Code\ALBERT\albert_base"
task_name = "STS-B"
task_path = os.path.join(GLUE_DIR, task_name)

# 检查数据路径
print(f"加载本地数据集: {task_path}")

# 加载 ALBERT 分词器和模型
tokenizer = AlbertTokenizer.from_pretrained(model_path)
model = TFAlbertForSequenceClassification.from_pretrained(model_path, from_pt=True, num_labels=1)

# 加载 STS-B 数据集
def load_local_glue_data(task_path):
    # STS-B 的训练文件和验证文件路径
    train_file = os.path.join(task_path, "train.tsv")
    dev_file = os.path.join(task_path, "dev.tsv")

    # 明确列名
    column_names = ["index", "sentence1", "sentence2", "score"]

    # 加载数据，跳过标题行
    train_data = pd.read_csv(train_file, sep="\t", header=0, names=column_names, on_bad_lines="skip")
    dev_data = pd.read_csv(dev_file, sep="\t", header=0, names=column_names, on_bad_lines="skip")

    # 打印列名检查
    print("训练集列名:", train_data.columns)
    print("验证集列名:", dev_data.columns)

    # 删除空值行
    train_data = train_data.dropna(subset=["sentence1", "sentence2", "score"])
    dev_data = dev_data.dropna(subset=["sentence1", "sentence2", "score"])

    # 强制将文本列转换为字符串
    train_data["sentence1"] = train_data["sentence1"].astype(str)
    train_data["sentence2"] = train_data["sentence2"].astype(str)
    dev_data["sentence1"] = dev_data["sentence1"].astype(str)
    dev_data["sentence2"] = dev_data["sentence2"].astype(str)

    # 确保分数列为浮点数
    train_data["score"] = train_data["score"].astype(float)
    dev_data["score"] = dev_data["score"].astype(float)

    # 打印样本数
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(dev_data)}")

    # 转换为 Hugging Face 数据集格式
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)

    return train_dataset, dev_dataset

train_dataset, dev_dataset = load_local_glue_data(task_path)

# 数据预处理：分词和编码（双句任务）
def preprocess_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'],
                     truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)

# 检查预处理后的数据集是否为空
print(f"预处理后训练集样本数: {len(train_dataset)}")
print(f"预处理后验证集样本数: {len(dev_dataset)}")

# 转换为 TensorFlow 数据集
if len(train_dataset) > 0 and len(dev_dataset) > 0:
    train_tf_dataset = train_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="score",
        shuffle=True,
        batch_size=32
    )

    dev_tf_dataset = dev_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="score",
        shuffle=False,
        batch_size=32
    )

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=["mse"])

    # 训练模型
    history = model.fit(train_tf_dataset, validation_data=dev_tf_dataset, epochs=3)

    # 评估模型
    results = model.evaluate(dev_tf_dataset)
    print("验证集结果:")
    print(f"Loss: {results[0]}, MSE: {results[1]}")
else:
    print("训练集或验证集为空，请检查数据预处理流程！")
