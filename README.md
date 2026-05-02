# 二手车价格预测基线

这是一个可直接扩展的本地基线工程，面向天池二手车价格预测这类表格回归赛题。

## 目录

- `data/raw/`：原始数据，默认放 `train.csv` 和 `testA.csv`
- `src/features.py`：特征工程
- `src/train.py`：训练、交叉验证、导出结果
- `outputs/`：模型输出目录

## 预期输入

默认假设训练集包含目标列 `price`，测试集不包含 `price`。如果存在以下列，会自动做额外处理：

原始赛题文件必须按单空格读取：`pd.read_csv(path, sep=" ")`。不要使用 `sep=r"\s+"`，否则连续空格里的空字段会被吞掉，导致整行列错位。

- `SaleID`
- `regDate`
- `creatDate`
- `notRepairedDamage`
- `v_0` 到 `v_14`

## 运行

```powershell
python .\src\train.py
```

如果文件名不是默认值，可以显式指定：

```powershell
python .\src\train.py --train-path .\data\raw\train.csv --test-path .\data\raw\testA.csv
```

## 输出

- `outputs\cv_metrics.json`：交叉验证指标
- `outputs\oof_predictions.csv`：训练集 OOF 预测
- `outputs\submission.csv`：测试集提交文件

## 后续优化方向

1. 分品牌、车型做目标编码与统计特征
2. 加入更稳的异常值裁剪和分桶特征
3. 安装 `lightgbm` 或 `catboost` 后替换当前基线模型
4. 对 `power`、`kilometer`、日期差值做更多交叉特征
