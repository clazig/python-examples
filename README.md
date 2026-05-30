# Python 示例项目

本项目包含两个简单的 Python 示例程序。

## 📄 项目结构

```
├── hello.py              # Hello Python 入门程序
├── iris_classifier.py    # 鸢尾花分类模型
└── .gitignore
```

---

## 👋 Hello Python

最简单的 Python 入门程序。

```bash
python hello.py
```

输出：`Hello Python!`

---

## 🌸 鸢尾花分类模型

使用 **Logistic Regression** 对 Iris 数据集进行分类。

### 依赖

```bash
pip install scikit-learn
```

### 运行

```bash
python iris_classifier.py
```

### 结果

| 项目 | 结果 |
|------|------|
| 算法 | Logistic Regression |
| 数据集 | Iris（150 条, 4 特征） |
| 训练/测试 | 105 / 45 条 |
| 测试准确率 | **93.33%** |

对新样本的预测示例：

```text
[5.1, 3.5, 1.4, 0.2]  → setosa     (97.68%)
[6.5, 3.0, 5.2, 2.0]  → virginica  (81.79%)
[5.9, 3.0, 4.2, 1.5]  → versicolor (87.05%)
```

---

## 📝 许可证

MIT
