import os
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import pandas as pd
from datasets import Dataset

# 本地数据集路径
GLUE_DIR = r"D:\AboutCode\Python\Pycharm\Code\ALBERT\glue_data"
model_path = r"D:\AboutCode\Python\Pycharm\Code\ALBERT\albert_base"
task_name = "QNLI"
task_path = os.path.join(GLUE_DIR, task_name)

# 检查数据路径
print(f"加载本地数据集: {task_path}")

# 加载 ALBERT 分词器和模型
tokenizer = AlbertTokenizer.from_pretrained(model_path)
model = TFAlbertForSequenceClassification.from_pretrained(model_path, from_pt=True, num_labels=2)

# 加载 QNLI 数据集
def load_local_glue_data(task_path):
    # QNLI 的训练文件和验证文件路径
    train_file = os.path.join(task_path, "train.tsv")
    dev_file = os.path.join(task_path, "dev.tsv")

    # 明确列名
    column_names = ["index", "question", "sentence", "label"]

    # 加载数据
    train_data = pd.read_csv(
        train_file, sep="\t", names=column_names, header=None, on_bad_lines="skip", low_memory=False
    )
    dev_data = pd.read_csv(
        dev_file, sep="\t", names=column_names, header=None, on_bad_lines="skip", low_memory=False
    )

    # 打印列名检查
    print("训练集列名:", train_data.columns)
    print("验证集列名:", dev_data.columns)

    # 删除空值行
    train_data = train_data.dropna(subset=["question", "sentence", "label"])
    dev_data = dev_data.dropna(subset=["question", "sentence", "label"])

    # 强制将文本列转换为字符串
    train_data["question"] = train_data["question"].astype(str)
    train_data["sentence"] = train_data["sentence"].astype(str)
    dev_data["question"] = dev_data["question"].astype(str)
    dev_data["sentence"] = dev_data["sentence"].astype(str)

    # 映射标签为整数
    label_mapping = {"entailment": 0, "not_entailment": 1}
    train_data["label"] = train_data["label"].map(label_mapping)
    dev_data["label"] = dev_data["label"].map(label_mapping)

    # 转换为 Hugging Face 数据集格式
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)

    return train_dataset, dev_dataset

train_dataset, dev_dataset = load_local_glue_data(task_path)

# 数据预处理：分词和编码（双句任务）
def preprocess_function(examples):
    return tokenizer(examples['question'], examples['sentence'],
                     truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)

# 转换为 TensorFlow 数据集
train_tf_dataset = train_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols="label",
    shuffle=True,
    batch_size=32
)

dev_tf_dataset = dev_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols="label",
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
