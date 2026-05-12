# 数据字典

## 1. 原始字段说明

| 字段 | 含义 | 类型 | 缺失与校验说明 |
|---|---|---|---|
| `SaleID` | 样本唯一 ID | int | 不应缺失；提交文件必须保留。 |
| `name` | 汽车交易名称编码 | int | 不应缺失；可用于频次特征 `name_count`。 |
| `regDate` | 汽车注册日期，格式近似 `YYYYMMDD` | int/date | 用于构造注册年月日、星期、车龄；非法日期会解析为缺失。 |
| `model` | 车型编码 | float/category | 允许少量缺失；核心类别特征，可用于 target encoding 和 model 频次特征。 |
| `brand` | 品牌编码 | int/category | 不应大面积缺失；用于品牌统计和品牌相关特征。 |
| `bodyType` | 车身类型编码 | float/category | 原始数据存在缺失；校验范围约为 0-7。 |
| `fuelType` | 燃油类型编码 | float/category | 原始数据存在缺失；校验范围约为 0-6。 |
| `gearbox` | 变速箱类型编码 | float/category | 原始数据存在缺失；校验范围约为 0-1。 |
| `power` | 发动机功率 | int/float | 可能有异常值；当前特征流程裁剪到 `[0, 600]` 并生成异常标记。 |
| `kilometer` | 行驶里程，单位为万公里区间编码 | float | 校验范围约为 0-15。 |
| `notRepairedDamage` | 是否有尚未修复损坏 | object/category | 原始 `-` 表示缺失；清洗后归一为缺失 token。 |
| `regionCode` | 地区编码 | int/category | 校验范围约为 0-100000；可用于类别和频次特征。 |
| `seller` | 卖家类型 | int/category | 通常区分个人/商家；当前训练特征中会被丢弃。 |
| `offerType` | 报价类型 | int/category | 当前训练特征中会被丢弃。 |
| `creatDate` | 广告创建日期，格式近似 `YYYYMMDD` | int/date | 用于构造创建日期特征和车龄。 |
| `price` | 二手车成交价格，训练目标 | int/float | 仅训练集存在；必须非负；MAE 直接基于该字段计算。 |
| `v_0`-`v_14` | 匿名连续特征 | float | 比赛提供的脱敏特征；含义未知，但通常有强预测价值。 |

## 2. 缺失值处理

当前保守清洗逻辑：

- 文本缺失 token：`""`、`" "`、`"nan"`、`"NaN"`、`"None"`、`"NULL"` 会归一为缺失。
- `notRepairedDamage` 中的 `-` 和空字符串视为缺失。
- 数值字段使用 `pd.to_numeric(errors="coerce")` 转换，非法值转为缺失。
- 模型训练阶段由预处理器处理缺失：
  - 数值列：中位数填充。
  - 类别列：填充为 `__MISSING__`，再做 ordinal encoding。

## 3. 原始数据校验

关键校验包括：

- 必需列是否存在。
- 训练集是否存在目标列 `price`。
- `price` 是否非负。
- 关键字段是否在合理范围内：
  - `kilometer`: 0-15
  - `bodyType`: 0-7
  - `fuelType`: 0-6
  - `gearbox`: 0-1
  - `regionCode`: 0-100000

这些校验的主要目的，是防止原始数据因错误分隔符读取而发生列错位。

## 4. 派生特征字段

常见派生字段包括：

- 日期与车龄：
  - `reg_year`, `reg_month`, `reg_day`, `reg_weekday`
  - `create_year`, `create_month`, `create_day`, `create_weekday`
  - `car_age_days`, `car_age_years`
- 频次特征：
  - `name_count`, `brand_count`, `regionCode_count`, `model_count`
- 非目标 group stats：
  - `brand_power_mean`, `brand_power_median`
  - `brand_kilometer_mean`
  - `model_power_mean`
- 功率和车龄交叉：
  - `power_per_age_year`
  - `power_per_age_year_sqrt`
  - `power_age_product`
  - `log_power_per_age`
  - `kilometer_per_age_year`
  - `age_kilometer_product`

## 5. Proxy 与 Target Encoding 特别说明

`brand_price_mean`、`model_price_mean`、`model_target_mean` 等价格 proxy 或 target encoding 特征必须使用 out-of-fold 或 fold-safe 方式构造。

禁止做法：

```text
用全量训练集 price 直接聚合出 model_price_mean，再作为训练特征参与 CV。
```

这种做法会造成 target leakage，使本地 MAE 虚高可信度下降。

当前标准预测文件：

- 验证预测：`outputs/predictions/valid_predictions.csv`
- 测试预测：`outputs/predictions/test_predictions.csv`

当前标准报告：

- 特征报告：`outputs/reports/feature_report.md`
