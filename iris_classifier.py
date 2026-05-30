"""鸢尾花分类模型 —— 使用 Logistic Regression"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 50)
print("鸢尾花分类模型")
print("=" * 50)
print(f"样本数量: {X.shape[0]}")
print(f"特征数量: {X.shape[1]}")
print(f"特征名称: {feature_names}")
print(f"类别: {target_names}")
print()

# 2. 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape[0]} 条")
print(f"测试集: {X_test.shape[0]} 条")
print()

# 3. 训练模型
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("模型训练完成 ✓")
print()

# 4. 评估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {acc:.2%}")
print()

print("分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print()

# 5. 用新样本做预测
print("=" * 50)
print("对新样本进行预测:")
print("=" * 50)

samples = [
    [5.1, 3.5, 1.4, 0.2],   # 明显是 setosa
    [6.5, 3.0, 5.2, 2.0],   # 明显是 virginica
    [5.9, 3.0, 4.2, 1.5],   # 明显是 versicolor
]

for s in samples:
    pred = model.predict([s])[0]
    proba = model.predict_proba([s])[0]
    name = target_names[pred]
    confidence = max(proba)
    print(f"  特征: {s}")
    print(f"  预测: {name} (置信度: {confidence:.2%})")
    print()
