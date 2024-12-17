import os
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import pandas as pd
from datasets import Dataset

# 本地数据集路径
GLUE_DIR = r"D:\AboutCode\Python\Pycharm\Code\ALBERT\glue_data"
model_path = r"D:\AboutCode\Python\Pycharm\Code\ALBERT\albert_base"
# 选择任务
task_name = "MRPC"  # 任务名称可以是 'MNLI', 'QQP', 'SST-2', 'MRPC' 等
task_path = os.path.join(GLUE_DIR, task_name)

# 检查数据路径
print(f"加载本地数据集: {task_path}")

# 加载 ALBERT 分词器和模型
model_name = "albert-base-v2"
tokenizer = AlbertTokenizer.from_pretrained(model_path)
model = TFAlbertForSequenceClassification.from_pretrained(model_path, num_labels=2)  # 假设 MRPC 是二分类

# 加载数据
def load_local_glue_data(task_path):
    # 数据集文件路径
    train_file = os.path.join(task_path, "train.tsv")
    dev_file = os.path.join(task_path, "dev.tsv")

    # 读取 TSV 文件
    train_data = pd.read_csv(train_file, sep="\t", on_bad_lines="skip")
    dev_data = pd.read_csv(dev_file, sep="\t", on_bad_lines="skip")

    # 转换为 Hugging Face 数据集格式
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)

    return train_dataset, dev_dataset

train_dataset, dev_dataset = load_local_glue_data(task_path)


# 数据预处理：分词和编码
def preprocess_function(examples):
    return tokenizer(examples['#1 String'], examples['#2 String'],
                     truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)


# 转换为 TensorFlow 数据集
train_tf_dataset = train_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols="Quality",  # 修改为 MRPC 数据集中的标签列名
    shuffle=True,
    batch_size=32
)

dev_tf_dataset = dev_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols="Quality",  # 修改为 MRPC 数据集中的标签列名
    shuffle=False,
    batch_size=32
)


# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# 训练模型
history = model.fit(train_tf_dataset, validation_data=dev_tf_dataset, epochs=3)

# 评估模型
results = model.evaluate(dev_tf_dataset)
print("验证集结果:")
print(f"Loss: {results[0]}, Accuracy: {results[1]}")