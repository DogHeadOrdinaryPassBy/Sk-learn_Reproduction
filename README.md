# Sk-learn_Reproduction
这是一个sk-learn库的复刻项目

##项目结构
Re_sklearn/
├── __init__.py
├── base/                    # 基础类和通用工具模块
│   ├── __init__.py
│   ├── base_model.py        # 所有模型的基类定义
│   └── metrics.py           # 通用的评估指标
├── preprocessing/           # 数据预处理模块
│   ├── __init__.py
│   ├── standard_scaler.py   # 标准化
│   ├── min_max_scaler.py    # 归一化
│   └── data_split.py        # 数据集划分（如 train_test_split）
├── model_selection/         # 模型选择与交叉验证
│   ├── __init__.py
│   ├── cross_validation.py  # 交叉验证工具
│   ├── grid_search.py       # 网格搜索
│   └── random_search.py     # 随机搜索
├── linear_model/            # 线性模型模块
│   ├── __init__.py
│   ├── linear_regression.py # 线性回归
│   └── logistic_regression.py # 逻辑回归
├── tree/                    # 决策树模型模块
│   ├── __init__.py
│   ├── decision_tree_classifier.py # 决策树分类
│   └── decision_tree_regressor.py  # 决策树回归
├── neighbors/               # K近邻算法模块
│   ├── __init__.py
│   └── knn_classifier.py    # K近邻分类
├── cluster/                 # 无监督学习算法
│   ├── __init__.py
│   └── kmeans.py            # KMeans 聚类
├── utils/                   # 通用工具函数
│   ├── __init__.py
│   └── utils.py             # 通用的工具函数（如数据加载）
├── examples/                # 示例代码处
│   ├── linear_regression_example.py
│   └── decision_tree_example.py
└── tests/                   # 单元测试
    ├── test_linear_model.py
    ├── test_preprocessing.py
    ├── test_model_selection.py
    └── test_tree.py