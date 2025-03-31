# 自动合成数据生成Pipeline

基于大语言模型的高质量合成数据自动生成工具，可以根据指定主题生成指定数量的数据。

## 功能特点

- 支持任意主题的合成数据生成
- 自动将主题分解为子主题，生成更多样化的数据
- 提供基础版和网页增强版两种pipeline
- 支持输出为标准JSON格式
- 可控制数据生成的数量和质量参数

## 安装依赖

```bash
pip install camel-ai tqdm
```

此外，还需要设置DeepSeek API密钥：

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

或者在运行时通过`--api_key`参数传递。

## 使用方法

### 快速开始

运行示例脚本了解基本功能：

```bash
python input_gen.py
```

### 基本使用

使用主脚本生成合成数据：

```bash
python generate_synthetic_data.py --topic "基础数学问题" --num_examples 10
```

### 高级选项

使用网页增强型pipeline（会爬取相关网页内容增强数据质量）：

```bash
python generate_synthetic_data.py --topic "机器学习基础" --num_examples 20 --num_subtopics 4 --web_enabled
```

### 完整参数说明

```
--topic            主题，必需参数，例如："编程基础"、"数学问题"等
--num_examples     要生成的示例数量，默认为10
--num_subtopics    将主题分解为几个子主题，默认为5
--temperature      模型温度参数，控制生成的随机性，默认为0.2
--output           输出文件路径，默认为synthetic_data.json
--api_key          DeepSeek API密钥
--web_enabled      启用网页内容增强
--verbose          显示详细日志
```

## 输出格式

生成的数据以JSON格式保存，格式如下：

```json
{
    "问题1": "答案1",
    "问题2": "答案2",
    "..."
}
```

同时会生成一个元数据文件，记录生成数据的统计信息。

## 工作原理

1. 将指定的主题分解为多个子主题
2. 为每个子主题生成指定数量的合成数据
3. 如果启用了网页增强，会爬取相关网页内容辅助生成
4. 整合所有子主题的数据，输出为统一格式

## 示例

### 基础数学问题

```bash
python generate_synthetic_data.py --topic "小学数学应用题" --num_examples 10
```

生成的数据示例：

```json
{
  "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？": "8个苹果",
  "一个长方形长8厘米，宽4厘米，它的面积是多少平方厘米？": "32平方厘米"
}
```

### LaTeX格式数学问题

支持LaTeX格式，适合生成高级数学问题：

```bash
python generate_synthetic_data.py --topic "微积分问题" --num_examples 5
```

生成的数据示例：

```json
{
  "求函数 $f(x) = x^2 - 4x + 3$ 的导数。": "$f'(x) = 2x - 4$",
  "计算积分 $\\int x \\cdot \\sin(x) dx$": "$\\sin(x) - x\\cos(x) + C$"
}
```

## 自定义开发

如需扩展功能，可参考以下文件：

- `synthetic_data_pipeline.py`: 基础pipeline实现
- `web_enhanced_data_pipeline.py`: 网页增强pipeline实现
- `generate_synthetic_data.py`: 主脚本，整合了两种pipeline

## 注意事项

- 需要有效的DeepSeek API密钥
- 网页增强功能依赖互联网连接
- 生成大量数据可能会消耗较多API配额 