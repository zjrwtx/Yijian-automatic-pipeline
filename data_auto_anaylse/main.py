from getpass import getpass
import os
import textwrap
from PIL.Image import OPEN
from camel.retrievers import AutoRetriever
from camel.types import StorageType
from typing import List, Dict
from dotenv import load_dotenv
from camel.retrievers import HybridRetriever

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.tasks import Task
from camel.toolkits import(
     SearchToolkit, PubMedToolkit, GoogleScholarToolkit, ArxivToolkit, SemanticScholarToolkit,
     FileWriteToolkit,
     RetrievalToolkit,
     CodeExecutionToolkit,
     ExcelToolkit
)
import asyncio
from camel.types import ModelPlatformType, ModelType  
from camel.societies.workforce import Workforce
from camel.configs import DeepSeekConfig, ChatGPTConfig, GeminiConfig

load_dotenv()
from camel.logger import set_log_level
set_log_level(level="DEBUG")

# 配置API密钥
openai_api_key = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = openai_api_key

# tools
tools=[
    SearchToolkit().search_google,
    PubMedToolkit().get_tools,
    ArxivToolkit().get_tools,
    *FileWriteToolkit().get_tools(),
    *RetrievalToolkit().get_tools(),
    *CodeExecutionToolkit(verbose=True).get_tools(),
    *ExcelToolkit().get_tools()
]

def make_lab_analysis_agent(
    role: str,
    persona: str,
    example_output: str,
    criteria: str,
) -> ChatAgent:
    msg_content = textwrap.dedent(
        f"""\
        您是检验科数据分析与研究领域的专业人员。
        您的角色：{role}
        您的职责和特点：{persona}
        输出格式示例：
        {example_output}
        您的分析标准：
        {criteria}
        """
    )

    sys_msg = BaseMessage.make_assistant_message(
        role_name="检验科数据分析专家",
        content=msg_content,
    )
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig(temperature=0.2).as_dict(),
    )

    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=tools,
    )

    return agent

# 创建患者基本信息分析师
patient_analyzer_persona = (
    '您是专门分析患者基本信息和临床数据的检验分析师。'
    '您专注于捕捉关键人口统计学特征、既往病史和临床表现。'
    '您需要将内容保存为txt文件到本地'
)

patient_analyzer_example = (
    '患者信息分析\n'
    '基本信息：\n'
    '- 年龄：[患者年龄]\n'
    '- 性别：[患者性别]\n'
    '- 入院诊断：[诊断信息]\n'
    '- 主诉：[主诉内容]\n'
    '既往史：\n'
    '- [过往病史记录]\n'
    '实验室检查概述：\n'
    '- [主要异常指标列表]\n'
    '临床相关性：\n'
    '- [检验结果与临床表现的关联分析]\n'
)

patient_analyzer_criteria = textwrap.dedent(
    """\
    1. 全面收集患者的基本人口统计学数据
    2. 梳理主要诊断和临床症状
    3. 总结关键的既往病史信息
    4. 提取主要的异常检验指标
    5. 分析检验结果与临床表现的相关性
    """
)

patient_analyzer = make_lab_analysis_agent(
    "患者基本信息分析师",
    patient_analyzer_persona,
    patient_analyzer_example,
    patient_analyzer_criteria,
)

# 创建检验指标分析师
lab_indicator_analyzer_persona = (
    '您是专门分析检验指标结果和临床意义的检验专家。'
    '您运用实验室医学知识评估异常检验值的临床重要性并提供解释。'
    '您需要将内容保存为txt文件到本地'
)

lab_indicator_analyzer_example = (
    '检验指标分析报告\n'
    '异常指标总结：\n'
    '- 指标1：[异常值] [参考范围] [异常程度评估]\n'
    '- 指标2：[异常值] [参考范围] [异常程度评估]\n'
    '分类解读：\n'
    '1. 肝功能相关：\n'
    '   - [异常指标及其临床意义]\n'
    '2. 肾功能相关：\n'
    '   - [异常指标及其临床意义]\n'
    '3. 血细胞相关：\n'
    '   - [异常指标及其临床意义]\n'
    '指标间关联性：\n'
    '- [相互关联的指标组及其临床意义]\n'
    '建议进一步检查：\n'
    '1. [建议1]：[理由]\n'
    '2. [建议2]：[理由]\n'
)

lab_indicator_analyzer_criteria = textwrap.dedent(
    """\
    1. 全面评估异常检验指标的临床意义
    2. 按系统或功能对异常指标进行分类
    3. 分析指标间的相互关系和临床相关性
    4. 评估异常指标的严重程度和紧急程度
    5. 提供有针对性的进一步检查建议
    """
)

lab_indicator_analyzer = make_lab_analysis_agent(
    "检验指标分析师",
    lab_indicator_analyzer_persona,
    lab_indicator_analyzer_example,
    lab_indicator_analyzer_criteria,
)

# 创建饮食分析师
diet_analyzer_persona = (
    '您是专门分析用户饮食结构和营养摄入的饮食专家。'
    '您运用营养学知识评估饮食模式并提供改进建议。'
    '您需要将内容保存为txt文件到本地'
)

diet_analyzer_example = (
    '饮食分析报告\n'
    '当前饮食模式：\n'
    '- 主要食物类型：[列举]\n'
    '- 餐次分布：[描述]\n'
    '- 热量摄入：[估计值]\n'
    '营养评估：\n'
    '- 蛋白质摄入：[评估]\n'
    '- 碳水化合物摄入：[评估]\n'
    '- 脂肪摄入：[评估]\n'
    '- 维生素和矿物质：[评估]\n'
    '饮食问题分析：\n'
    '- [关键问题1]：[详细描述]\n'
    '- [关键问题2]：[详细描述]\n'
    '饮食建议：\n'
    '1. [建议1]\n'
    '2. [建议2]\n'
)

diet_analyzer_criteria = textwrap.dedent(
    """\
    1. 全面评估用户当前饮食结构
    2. 计算主要营养素摄入比例
    3. 识别不健康的饮食模式
    4. 考虑用户偏好和生活方式
    5. 提供实用且可持续的饮食改进建议
    """
)

diet_analyzer = make_lab_analysis_agent(
    "饮食分析师",
    diet_analyzer_persona,
    diet_analyzer_example,
    diet_analyzer_criteria,
)

# 创建分子诊断分析专家
molecular_diagnostics_persona = (
    '您是检验医学中分子诊断领域的专家，专长于解读分子检测数据和基因变异信息。'
    '您利用分子生物学知识分析基因表达、突变谱和分子标志物数据。'
    '您能够整合分子检测结果与传统检验项目进行综合分析。'
    '您需要将内容保存为txt文件到本地'
)

molecular_diagnostics_example = (
    '分子诊断分析报告\n\n'
    '分子检测数据概述：\n'
    '1. 基因变异情况：\n'
    '   - 关键突变位点：[突变列表及其功能]\n'
    '   - 突变负荷分析：[评分结果及解释]\n'
    '   - 变异临床意义：[解释]\n\n'
    '2. 基因表达特征：\n'
    '   - 差异表达基因：[上/下调基因列表]\n'
    '   - 表达谱特征：[关键模式及功能]\n'
    '   - 表达-表型关联：[相关性分析]\n\n'
    '分子与传统检验整合：\n'
    '1. 生化指标与分子标志物相关性：\n'
    '   - [关键相关性发现及意义]\n'
    '2. 分子分型与临床表现：\n'
    '   - [分型结果及临床关联]\n\n'
    '分子病理机制推断：\n'
    '1. 信号通路异常：\n'
    '   - [异常通路及影响]\n'
    '2. 分子病理过程：\n'
    '   - [推断的分子机制]\n\n'
    '精准医学应用建议：\n'
    '1. 靶向治疗相关性：[建议]\n'
    '2. 药物反应预测：[预测结果]\n'
    '3. 分子监测方案：[监测建议]\n\n'
    '科研创新方向：\n'
    '1. 新型分子标志物：[潜在标志物及意义]\n'
    '2. 机制研究方向：[建议研究方向]\n'
)

molecular_diagnostics_criteria = textwrap.dedent(
    """\
    1. 深入解读分子检测数据的临床意义
    2. 整合分子数据与传统检验指标
    3. 识别具有诊断和预后价值的分子特征
    4. 推断潜在的分子病理机制
    5. 提供精准医学应用的具体建议
    6. 发现分子诊断领域的研究创新点
    7. 评估分子检测结果的质量和可靠性
    """
)

molecular_diagnostics_analyst = make_lab_analysis_agent(
    "分子诊断分析专家",
    molecular_diagnostics_persona,
    molecular_diagnostics_example,
    molecular_diagnostics_criteria,
)

# 创建参考值研究专家
reference_value_researcher_persona = (
    '您是检验医学中参考值和临界值研究的专家，专长于建立和优化检验指标的参考区间。'
    '您分析人口统计学因素对参考值的影响，并构建个体化参考区间模型。'
    '您的专长包括分析检验指标的生物学变异和临床决策临界值的确定。'
    '您需要将内容保存为txt文件到本地'
)

reference_value_researcher_example = (
    '参考值研究分析报告\n\n'
    '人群参考区间分析：\n'
    '1. 现有参考区间评估：\n'
    '   - 指标1：[当前区间] - [适用性评估]\n'
    '   - 指标2：[当前区间] - [适用性评估]\n\n'
    '2. 特定人群参考区间：\n'
    '   - 性别差异：[分析结果]\n'
    '   - 年龄分层：[分析结果]\n'
    '   - 民族/地区差异：[分析结果]\n\n'
    '个体化参考区间模型：\n'
    '1. 动态参考区间：\n'
    '   - 模型设计：[描述]\n'
    '   - 影响因素：[因素列表及权重]\n'
    '   - 预测精度：[评估结果]\n\n'
    '2. 患者自身参考变化：\n'
    '   - 变化值临界点：[数值及意义]\n'
    '   - 时间动态模式：[模式描述]\n\n'
    '临床决策临界值：\n'
    '1. 诊断临界值：\n'
    '   - 指标1：[建议值] - [依据]\n'
    '   - 指标2：[建议值] - [依据]\n\n'
    '2. 治疗干预临界值：\n'
    '   - 指标1：[建议值] - [依据]\n'
    '   - 指标2：[建议值] - [依据]\n\n'
    '研究创新点：\n'
    '1. 改进的参考区间方法：[描述]\n'
    '2. 新的临界值确定策略：[描述]\n'
    '3. 个体化参考值应用：[潜在应用场景]\n'
)

reference_value_researcher_criteria = textwrap.dedent(
    """\
    1. 评估现有参考区间的适用性和局限性
    2. 分析人口统计学因素对参考值的影响
    3. 构建考虑多因素的个体化参考区间模型
    4. 研究检验指标的生物学变异规律
    5. 确定具有临床意义的决策临界值
    6. 提出参考值研究的创新方法和策略
    7. 探索参考区间在精准医学中的应用
    """
)

reference_value_researcher = make_lab_analysis_agent(
    "参考值研究专家",
    reference_value_researcher_persona,
    reference_value_researcher_example,
    reference_value_researcher_criteria,
)

# 创建科研灵感生成器
research_inspiration_generator_persona = (
    '您是整合各专家意见提炼科研灵感和创新方向的研究思想家。'
    '您创建有价值、可行且具有创新性的检验医学研究方向和假设。'
    '您需要将内容保存为txt文件到本地'
)

research_inspiration_generator_example = (
    '检验医学科研灵感报告\n'
    '核心研究思路：\n'
    '- [主要科研灵感概述]\n'
    '创新研究方向：\n'
    '1. 研究方向1：\n'
    '   - 研究假设：[详细描述]\n'
    '   - 创新点：[说明]\n'
    '   - 潜在影响：[预期影响]\n'
    '   - 相关前沿文献：[关键文献]\n'
    '2. 研究方向2：\n'
    '   - 研究假设：[详细描述]\n'
    '   - 创新点：[说明]\n'
    '   - 潜在影响：[预期影响]\n'
    '   - 相关前沿文献：[关键文献]\n'
    '方法学创新：\n'
    '- [方法学创新点1]\n'
    '- [方法学创新点2]\n'
    '研究设计框架：\n'
    '1. 研究方向1设计：\n'
    '   - 研究对象：[描述]\n'
    '   - 关键变量：[列表]\n'
    '   - 分析方法：[描述]\n'
    '   - 预期结果：[描述]\n'
    '2. 研究方向2设计：\n'
    '   - [类似结构]\n'
    '合作与资源需求：\n'
    '- [所需专业领域]\n'
    '- [关键技术平台]\n'
    '- [数据资源需求]\n'
)

research_inspiration_generator_criteria = textwrap.dedent(
    """\
    1. 整合所有专家的见解提炼创新科研方向
    2. 提出具有原创性和可行性的研究假设
    3. 确保研究方向具有临床转化价值
    4. 考虑当前技术限制和解决方案
    5. 提供清晰的研究设计框架
    6. 识别方法学创新点和技术突破口
    7. 评估研究方向的潜在学术和临床影响
    """
)

research_inspiration_generator = make_lab_analysis_agent(
    "科研灵感生成器",
    research_inspiration_generator_persona,
    research_inspiration_generator_example,
    research_inspiration_generator_criteria,
)

# 创建检验数据生成器
lab_data_generator_persona = (
    '您是检验医学研究的数据科学家，专门负责生成模拟检验数据用于模型训练和测试。'
    '您能够创建多种不同格式的合成检验数据，包括使用不同分隔符、不同命名约定的CSV文件。'
    '您可以生成多样化的数据结构，模拟不同检验项目和临床特征的组合。'
    '您的数据可以包括常规生化、血常规、凝血、免疫、分子诊断等多种检验类型。'
    '您需要使用CodeExecutionToolkit来执行Python代码，生成CSV格式的数据集。'
    '您的代码应该能生成临床上合理的检验数据，反映真实的生理病理关系。'
    '您需要将内容保存为txt文件到本地，并确保生成的CSV文件路径明确。'
)

lab_data_generator_example = (
    '检验数据生成报告\n\n'
    '生成的数据集描述：\n'
    '- 数据集目的：[描述]\n'
    '- 数据格式特点：[描述，例如使用的分隔符、命名约定等]\n'
    '- 病例数量：[数量]\n'
    '- 检验项目数：[数量]\n'
    '- 主要检验类别：[列表]\n\n'
    '模拟疾病特征：\n'
    '- 疾病类型：[模拟的疾病类型]\n'
    '- 关键异常指标：[关键指标及其变化特征]\n'
    '- 疾病进展模式：[描述]\n\n'
    '数据关联性设计：\n'
    '- 指标间相关关系：[描述]\n'
    '- 生理病理学关系：[描述]\n'
    '- 时间序列特性：[描述]\n\n'
    '数据格式变体：\n'
    '- 使用的分隔符：[分隔符]\n'
    '- 特殊列命名规则：[规则描述]\n'
    '- 数据编码方式：[编码]\n\n'
    '数据生成代码执行结果：\n'
    '[代码执行输出内容]\n\n'
    '生成的CSV文件路径：[文件路径]\n\n'
    '数据字典：\n'
    '- 列名1：[描述]\n'
    '- 列名2：[描述]\n'
    '...'
)

lab_data_generator_criteria = textwrap.dedent(
    """\
    1. 使用CodeExecutionToolkit执行Python代码生成检验数据
    2. 生成多种不同格式的CSV文件，包括不同分隔符、不同命名约定
    3. 模拟真实检验数据的分布特征和正常/异常范围
    4. 创建具有生理病理学合理性的指标相关关系
    5. 包含多种类型的检验项目(生化、血常规、凝血等)
    6. 模拟不同疾病状态的检验特征模式
    7. 生成具有时间序列特性的纵向数据
    8. 确保数据包含足够的变异性和模式，以便进行统计分析
    9. 提供清晰的数据字典和元数据
    10. 将生成的CSV保存在容易访问的路径
    11. 确保生成的数据适合后续建模和分析
    """
)

lab_data_generator = make_lab_analysis_agent(
    "检验数据生成器",
    lab_data_generator_persona,
    lab_data_generator_example,
    lab_data_generator_criteria,
)
# 创建疾病模式分析师
disease_pattern_analyzer_persona = (
    '您是专门分析检验数据中疾病特征模式的专家。'
    '您根据检验结果的模式识别可能的疾病类型和进展阶段。'
    '您需要将内容保存为txt文件到本地'
)

disease_pattern_analyzer_example = (
    '疾病模式分析报告\n'
    '检验模式特征：\n'
    '- 模式1：[相关指标组合] - [可能的临床意义]\n'
    '- 模式2：[相关指标组合] - [可能的临床意义]\n'
    '可能的疾病类型：\n'
    '1. [疾病1]：\n'
    '   - 支持证据：[相关检验结果]\n'
    '   - 符合程度：[评估]\n'
    '   - 鉴别要点：[关键区分特征]\n'
    '2. [疾病2]：\n'
    '   - 支持证据：[相关检验结果]\n'
    '   - 符合程度：[评估]\n'
    '   - 鉴别要点：[关键区分特征]\n'
    '疾病进展评估：\n'
    '- 疾病阶段：[评估结果]\n'
    '- 活动性指标：[相关指标及意义]\n'
    '- 严重程度：[评估结果]\n'
    '进一步诊断建议：\n'
    '- [建议1]\n'
    '- [建议2]\n'
)

disease_pattern_analyzer_criteria = textwrap.dedent(
    """\
    1. 识别检验结果中具有特异性的疾病模式
    2. 分析可能的疾病类型及其可能性
    3. 评估疾病的活动性和进展阶段
    4. 提供关键鉴别诊断要点
    5. 建议有针对性的进一步确诊措施
    """
)

disease_pattern_analyzer = make_lab_analysis_agent(
    "疾病模式分析师",
    disease_pattern_analyzer_persona,
    disease_pattern_analyzer_example,
    disease_pattern_analyzer_criteria,
)

# 创建数据质量控制专家
data_quality_specialist_persona = (
    '您是专门分析检验数据质量和可靠性的专家。'
    '您帮助识别数据中的异常值、变异性和潜在错误，并提供质量改进建议。'
    '您需要将内容保存为txt文件到本地'
)

data_quality_specialist_example = (
    '数据质量分析报告\n'
    '数据完整性评估：\n'
    '- 缺失数据情况：[分析]\n'
    '- 数据覆盖率：[评估]\n'
    '- 时间序列完整性：[评估]\n'
    '异常值分析：\n'
    '1. [异常值1]：[详细描述]\n'
    '2. [异常值2]：[详细描述]\n'
    '数据变异性：\n'
    '- 生物学变异评估：[分析结果]\n'
    '- 分析学变异评估：[分析结果]\n'
    '- 前分析变异评估：[分析结果]\n'
    '潜在干扰因素：\n'
    '- [因素1]：[影响分析]\n'
    '- [因素2]：[影响分析]\n'
    '质量改进建议：\n'
    '1. [建议1]：[详细描述]\n'
    '2. [建议2]：[详细描述]\n'
)

data_quality_specialist_criteria = textwrap.dedent(
    """\
    1. 评估数据的完整性和一致性
    2. 识别潜在的异常值和离群点
    3. 分析生物学和分析学变异来源
    4. 评估可能的干扰和混淆因素
    5. 提供改进数据质量的具体建议
    """
)

data_quality_specialist = make_lab_analysis_agent(
    "数据质量控制专家",
    data_quality_specialist_persona,
    data_quality_specialist_example,
    data_quality_specialist_criteria,
)

# 创建运动规划师
exercise_planner_persona = (
    '您是专门为用户设计个性化运动方案的运动规划师。'
    '您根据用户的体能水平、偏好和目标创建有效的运动计划。'
    '您需要将内容保存为txt文件到本地'
)

exercise_planner_example = (
    '运动规划方案\n'
    '用户运动现状：\n'
    '- 当前活动水平：[描述]\n'
    '- 运动偏好：[列举]\n'
    '- 体能评估：[评估]\n'
    '运动计划：\n'
    '1. 有氧运动：\n'
    '   - 类型：[运动类型]\n'
    '   - 频率：[次/周]\n'
    '   - 强度：[描述]\n'
    '   - 时长：[分钟/次]\n'
    '2. 力量训练：\n'
    '   - 类型：[运动类型]\n'
    '   - 频率：[次/周]\n'
    '   - 组数和重复次数：[详情]\n'
    '3. 灵活性训练：\n'
    '   - 类型：[训练类型]\n'
    '   - 频率：[次/周]\n'
    '週计划安排：\n'
    '[详细周计划表]\n'
    '注意事项：\n'
    '- [注意事项1]\n'
    '- [注意事项2]\n'
)

exercise_planner_criteria = textwrap.dedent(
    """\
    1. 根据用户体能水平设计适合的运动方案
    2. 平衡有氧运动、力量训练和灵活性训练
    3. 考虑用户的时间限制和偏好
    4. 设计循序渐进的运动强度增加计划
    5. 包含明确的运动指导和注意事项
    """
)

exercise_planner = make_lab_analysis_agent(
    "运动规划师",
    exercise_planner_persona,
    exercise_planner_example,
    exercise_planner_criteria,
)

# 创建行为心理专家
behavior_specialist_persona = (
    '您是专门分析用户心理状态和行为模式的心理专家。'
    '您帮助识别影响体重管理的心理因素并提供行为改变策略。'
    '您需要将内容保存为txt文件到本地'
)

behavior_specialist_example = (
    '行为心理分析\n'
    '心理因素评估：\n'
    '- 饮食行为模式：[分析]\n'
    '- 压力应对方式：[分析]\n'
    '- 自我效能感：[评估]\n'
    '- 动机水平：[评估]\n'
    '行为障碍：\n'
    '1. [障碍1]：[详细描述]\n'
    '2. [障碍2]：[详细描述]\n'
    '行为改变策略：\n'
    '1. [策略1]：[详细描述]\n'
    '2. [策略2]：[详细描述]\n'
    '习惯培养计划：\n'
    '- [习惯1]：[培养方法]\n'
    '- [习惯2]：[培养方法]\n'
    '自我监测建议：\n'
    '- [监测方法1]\n'
    '- [监测方法2]\n'
)

behavior_specialist_criteria = textwrap.dedent(
    """\
    1. 评估影响体重管理的心理因素
    2. 识别不健康的行为模式
    3. 提供具体的行为改变策略
    4. 设计渐进式的习惯培养计划
    5. 增强用户的自我效能感和长期动机
    """
)

behavior_specialist = make_lab_analysis_agent(
    "行为心理专家",
    behavior_specialist_persona,
    behavior_specialist_example,
    behavior_specialist_criteria,
)

# 创建数据分析师
data_analyst_persona = (
    '您是专门分析用户体重变化数据和趋势的数据分析师。'
    '您使用统计方法识别影响体重的关键因素和模式。'
    '您能够使用Python生成数据分析代码，处理任意格式的CSV数据并可视化结果。'
    '您擅长解析和理解各种CSV格式，不限于特定列名或结构。'
    '您需要首先探索CSV数据的结构，然后基于其内容进行灵活分析。'
    '您需要使用CodeExecutionToolkit执行您的Python分析代码。'
    '您需要将内容保存为txt文件到本地'
)

data_analyst_example = (
    '数据分析报告\n'
    '数据集概述：\n'
    '- 数据源：[描述]\n'
    '- 记录数量：[数量]\n'
    '- 关键特征：[列表]\n'
    '数据预处理：\n'
    '- 数据清洗步骤：[步骤列表]\n'
    '- 特征工程：[技术列表]\n'
    '体重变化趋势：\n'
    '- 短期趋势（过去1个月）：[分析]\n'
    '- 中期趋势（过去3个月）：[分析]\n'
    '- 长期趋势（过去6个月+）：[分析]\n'
    '关键影响因素：\n'
    '1. [因素1]：相关性分析 [结果]\n'
    '2. [因素2]：相关性分析 [结果]\n'
    '预测模型：\n'
    '- 基于当前行为的体重预测：[预测结果]\n'
    '- 改变后的体重预测：[预测结果]\n'
    '数据洞察：\n'
    '- [洞察1]\n'
    '- [洞察2]\n'
    '建议数据收集：\n'
    '- [建议1]\n'
    '- [建议2]\n'
    '数据分析代码执行结果：\n'
    '[代码执行输出内容]\n'
)

data_analyst_criteria = textwrap.dedent(
    """\
    1. 使用CodeExecutionToolkit执行Python分析代码
    2. 首先探索任意格式CSV数据的结构，以识别关键列和特征
    3. 灵活适应不同的数据格式和命名约定
    4. 分析用户体重变化的时间序列数据（如果存在）
    5. 识别影响体重的关键变量和因素
    6. 建立预测模型评估不同行为的潜在影响
    7. 提供基于数据的可行性建议
    8. 设计有效的数据收集和监测策略
    9. 生成可视化结果展示分析发现
    10. 确保代码能处理各种格式的CSV数据文件，包括不同的分隔符和编码
    """
)

data_analyst = make_lab_analysis_agent(
    "数据分析师",
    data_analyst_persona,
    data_analyst_example,
    data_analyst_criteria,
)

# 创建高级统计建模专家
advanced_modeling_persona = (
    '您是体重管理领域的高级统计建模专家，专长于开发复杂的预测模型和多因素分析。'
    '您运用机器学习、深度学习和高级统计方法构建创新的体重管理解决方案。'
    '您能够使用Python实现各种统计模型和机器学习算法，处理任意格式的CSV数据。'
    '您擅长处理非标准数据格式，能够灵活适应各种CSV结构和命名约定。'
    '您需要先探索数据结构并理解其内容，然后再进行特征工程和建模。'
    '您需要使用CodeExecutionToolkit执行您的Python建模代码。'
    '您的模型能够整合多源数据，包括生物标志物、行为模式、环境因素和基因信息。'
    '您需要将内容保存为txt文件到本地'
)

advanced_modeling_example = (
    '高级体重管理建模分析\n\n'
    '多维度因素分析：\n'
    '1. 代谢组学分析：\n'
    '   - 关键代谢物标志物：[标志物列表及影响]\n'
    '   - 代谢通路异常：[通路分析结果]\n'
    '   - 个性化代谢特征：[详细描述]\n\n'
    '2. 时空行为模式分析：\n'
    '   - 活动-时间模式识别：[模式描述]\n'
    '   - 环境触发因素映射：[关键触发因素]\n'
    '   - 行为轨迹预测：[预测结果]\n\n'
    '3. 生理-心理状态建模：\n'
    '   - 压力-饮食关系函数：f(x) = [函数模型]\n'
    '   - 情绪-活动水平耦合：[相关性系数与解释]\n'
    '   - 睡眠-代谢效率算法：[算法描述]\n\n'
    '复杂系统建模：\n'
    '1. 非线性动态模型：\n'
    '   - 体重变化微分方程：[数学表达式]\n'
    '   - 系统稳定性分析：[分析结果]\n'
    '   - 扰动响应预测：[模拟结果]\n\n'
    '2. 多尺度生理模型：\n'
    '   - 细胞水平：[模型描述]\n'
    '   - 器官系统水平：[模型描述]\n'
    '   - 全身整合水平：[模型描述]\n\n'
    '预测性智能算法：\n'
    '1. 个性化体重轨迹预测：\n'
    '   - 短期预测（7天）：[预测值与置信区间]\n'
    '   - 中期预测（30天）：[预测值与置信区间]\n'
    '   - 长期预测（180天）：[预测值与置信区间]\n\n'
    '2. 干预敏感性分析：\n'
    '   - 响应曲面模型：[模型可视化描述]\n'
    '   - 最佳干预点识别：[关键点列表]\n'
    '   - 干预组合优化：[最优组合建议]\n\n'
    '3. 适应性学习算法：\n'
    '   - 反馈灵敏度：[量化指标]\n'
    '   - 模型自我优化机制：[机制描述]\n'
    '   - 预测精度演化：[精度变化趋势]\n\n'
    '建议实施方案：\n'
    '1. 数据采集增强策略：[策略详情]\n'
    '2. 个性化干预程序：[程序框架]\n'
    '3. 动态反馈系统设计：[系统架构]\n'
    '4. 预期效果量化指标：[关键指标列表]\n\n'
    '模型实现代码执行结果：\n'
    '[代码执行输出内容]\n'
)

advanced_modeling_criteria = textwrap.dedent(
    """\
    1. 使用CodeExecutionToolkit执行Python建模代码
    2. 将复杂的生物医学数据转化为可操作的体重管理见解
    3. 创建整合多源数据的全面预测模型
    4. 应用前沿统计和机器学习方法进行个性化分析
    5. 识别个体特异性的体重调节机制
    6. 量化不同干预措施的预期效果
    7. 开发动态适应的个性化方案
    8. 提供基于证据的精准干预建议
    9. 确保代码能正确读取和处理CSV数据
    10. 输出模型评估指标和可视化结果
    """
)

advanced_modeler = make_lab_analysis_agent(
    "高级统计建模专家",
    advanced_modeling_persona,
    advanced_modeling_example,
    advanced_modeling_criteria,
)

# 创建生物信息学分析专家
bioinformatics_persona = (
    '您是体重管理领域的生物信息学分析专家，专长于整合基因组学、蛋白组学和代谢组学数据。'
    '您利用计算生物学方法发现影响体重调节的分子机制和生物标志物。'
    '您能够分析微生物组数据并确定其与能量代谢的关系。'
    '您需要将内容保存为txt文件到本地'
)

bioinformatics_example = (
    '生物信息学分析报告\n\n'
    '基因组分析：\n'
    '1. 体重相关基因变异：\n'
    '   - 风险等位基因：[基因列表及其功能]\n'
    '   - 多基因风险评分：[评分结果及解释]\n'
    '   - 表观遗传修饰：[关键修饰及影响]\n\n'
    '2. 基因表达模式：\n'
    '   - 差异表达基因：[上/下调基因列表]\n'
    '   - 基因表达模块：[关键模块及功能]\n'
    '   - 调控网络分析：[网络特征及中心基因]\n\n'
    '微生物组分析：\n'
    '1. 肠道菌群多样性：\n'
    '   - α多样性指数：[多样性得分及解释]\n'
    '   - β多样性模式：[群落差异分析]\n'
    '   - 关键菌属丰度：[关键菌属列表及作用]\n\n'
    '2. 功能代谢通路：\n'
    '   - 能量收获相关通路：[通路列表及活性]\n'
    '   - 短链脂肪酸产生：[产量估计及影响]\n'
    '   - 宿主-微生物互作：[关键互作机制]\n\n'
    '代谢组分析：\n'
    '1. 代谢物特征：\n'
    '   - 差异代谢物：[代谢物列表及变化方向]\n'
    '   - 代谢标志物：[候选标志物及诊断价值]\n'
    '   - 代谢网络重构：[网络特征描述]\n\n'
    '2. 代谢通量分析：\n'
    '   - 能量产生效率：[通量估计及比较]\n'
    '   - 底物利用偏好：[主要能源底物分析]\n'
    '   - 代谢灵活性评估：[灵活性指标及解释]\n\n'
    '多组学整合分析：\n'
    '1. 基因-蛋白-代谢整合：\n'
    '   - 关键信号通路：[通路名称及状态]\n'
    '   - 调控轴识别：[关键调控轴及影响]\n'
    '   - 分子互作网络：[网络特征及关键节点]\n\n'
    '2. 表型关联分析：\n'
    '   - 分子特征-体重关联：[显著关联及强度]\n'
    '   - 表型预测生物标志物：[标志物组合及准确率]\n'
    '   - 分子表型划分：[亚型特征及临床意义]\n\n'
    '个性化干预建议：\n'
    '1. 基于基因型的营养建议：[针对性饮食调整]\n'
    '2. 微生物组调控策略：[益生菌/益生元方案]\n'
    '3. 代谢靶向干预：[代谢通路调节方法]\n'
    '4. 监测生物标志物：[建议监测的关键指标]\n'
)

bioinformatics_criteria = textwrap.dedent(
    """\
    1. 整合多层次组学数据提供全面分子视角
    2. 识别个体特异性的体重调节机制
    3. 发现可用于个性化干预的生物标志物
    4. 分析微生物组与宿主代谢的互作关系
    5. 构建体重调节的分子通路和网络模型
    6. 提供基于分子机制的精准干预策略
    7. 设计个体化的生物标志物监测方案
    """
)

bioinformatics_analyst = make_lab_analysis_agent(
    "生物信息学分析专家",
    bioinformatics_persona,
    bioinformatics_example,
    bioinformatics_criteria,
)

# 创建数字孪生模拟专家
digital_twin_persona = (
    '您是体重管理领域的数字孪生模拟专家，专长于创建个体的虚拟生理模型。'
    '您构建整合多维数据的动态模拟系统，用于预测干预结果和优化个性化方案。'
    '您的模型能模拟代谢、行为和环境因素的复杂互动。'
    '您需要将内容保存为txt文件到本地'
)

digital_twin_example = (
    '数字孪生体重管理模拟报告\n\n'
    '数字孪生模型构建：\n'
    '1. 个体参数化：\n'
    '   - 基础代谢特征：[参数列表及数值]\n'
    '   - 激素调节模型：[模型特征及参数]\n'
    '   - 能量平衡动力学：[方程及系数]\n\n'
    '2. 行为-生理耦合：\n'
    '   - 活动-能耗映射：[函数关系描述]\n'
    '   - 饮食-代谢响应：[时间序列模型]\n'
    '   - 睡眠-内分泌互作：[互作模型参数]\n\n'
    '3. 环境因素整合：\n'
    '   - 时空活动约束：[约束条件参数]\n'
    '   - 社会影响网络：[网络拓扑及影响强度]\n'
    '   - 压力源-行为触发：[触发模型特征]\n\n'
    '模拟分析结果：\n'
    '1. 基线状态模拟：\n'
    '   - 当前平衡点分析：[平衡点特征及稳定性]\n'
    '   - 代谢稳态评估：[代谢灵活性指标]\n'
    '   - 昼夜节律特征：[周期性参数分析]\n\n'
    '2. 干预方案模拟：\n'
    '   - 饮食调整方案A：[模拟轨迹及结果]\n'
    '   - 运动方案B：[模拟轨迹及结果]\n'
    '   - 行为干预C：[模拟轨迹及结果]\n'
    '   - 综合方案D：[模拟轨迹及结果]\n\n'
    '3. 敏感性与稳健性分析：\n'
    '   - 参数敏感性排名：[敏感参数列表]\n'
    '   - 扰动响应特征：[系统韧性评估]\n'
    '   - 长期稳定性分析：[稳定性条件]\n\n'
    '4. 最优化轨迹规划：\n'
    '   - 目标函数定义：[目标函数表达式]\n'
    '   - 约束条件设定：[约束条件列表]\n'
    '   - 最优干预序列：[时序干预方案]\n'
    '   - 预期轨迹及不确定性：[轨迹及置信区间]\n\n'
    '个性化应用方案：\n'
    '1. 关键反馈点识别：[干预关键时间点]\n'
    '2. 适应性调整规则：[调整算法描述]\n'
    '3. 实时监测建议：[监测参数及频率]\n'
    '4. 闭环控制策略：[干预-反馈循环设计]\n'
)

digital_twin_criteria = textwrap.dedent(
    """\
    1. 构建整合生理、行为和环境因素的全面个体模型
    2. 准确模拟不同干预方案的动态响应
    3. 识别个体特异的敏感参数和干预点
    4. 预测长期体重轨迹及稳定性条件
    5. 设计最优干预序列和调整规则
    6. 评估干预方案的稳健性和不确定性
    7. 提供基于模拟的个性化精准方案
    """
)

digital_twin_specialist = make_lab_analysis_agent(
    "数字孪生模拟专家",
    digital_twin_persona,
    digital_twin_example,
    digital_twin_criteria,
)

# 创建综合方案设计师
plan_designer_persona = (
    '您是整合各专家意见设计综合体重管理方案的设计师。'
    '您创建个性化、平衡且可持续的体重管理计划。'
    '您需要将内容保存为txt文件到本地'
)

plan_designer_example = (
    '综合体重管理方案\n'
    '目标设定：\n'
    '- 短期目标（1个月）：[目标详情]\n'
    '- 中期目标（3个月）：[目标详情]\n'
    '- 长期目标（6个月+）：[目标详情]\n'
    '饮食计划：\n'
    '[基于饮食分析师建议的综合计划]\n'
    '运动计划：\n'
    '[基于运动规划师建议的综合计划]\n'
    '行为改变策略：\n'
    '[基于行为心理专家建议的综合策略]\n'
    '进度监测计划：\n'
    '- 监测指标：[列举]\n'
    '- 监测频率：[详情]\n'
    '- 反馈机制：[详情]\n'
    '调整机制：\n'
    '- [调整策略1]\n'
    '- [调整策略2]\n'
)

plan_designer_criteria = textwrap.dedent(
    """\
    1. 整合所有专家的建议创建统一方案
    2. 设定符合SMART原则的目标
    3. 确保方案的可行性和可持续性
    4. 考虑用户的个人情况和偏好
    5. 包含适应性调整机制和应对策略
    """
)

plan_designer = make_lab_analysis_agent(
    "综合方案设计师",
    plan_designer_persona,
    plan_designer_example,
    plan_designer_criteria,
)

# 创建示例数据生成器
data_generator_persona = (
    '您是体重管理研究的数据科学家，专门负责生成模拟数据用于模型训练和测试。'
    '您能够创建多种不同格式的合成用户数据，包括使用不同分隔符、不同命名约定的CSV文件。'
    '您可以生成多样化的数据结构，有时使用标准的健康监测命名规范，有时使用非标准或简化的命名方式。'
    '您的数据可能包括体重变化、饮食记录、活动水平和生理指标等，但列名和结构可以灵活多变。'
    '您需要使用CodeExecutionToolkit来执行Python代码，生成CSV格式的数据集。'
    '您的代码应该能生成不同风格的CSV文件，用于测试系统处理任意格式CSV的能力。'
    '您需要将内容保存为txt文件到本地，并确保生成的CSV文件路径明确。'
)

data_generator_example = (
    '数据生成报告\n\n'
    '生成的数据集描述：\n'
    '- 数据集目的：[描述]\n'
    '- 数据格式特点：[描述，例如使用的分隔符、命名约定等]\n'
    '- 样本量：[数量]\n'
    '- 时间范围：[范围]\n'
    '- 特征列表：[列表]\n\n'
    '数据分布特征：\n'
    '- 体重变化模式：[描述]\n'
    '- 干预响应模式：[描述]\n'
    '- 噪声与变异性：[描述]\n\n'
    '数据格式变体：\n'
    '- 使用的分隔符：[分隔符]\n'
    '- 特殊列命名规则：[规则描述]\n'
    '- 数据编码方式：[编码]\n\n'
    '数据生成代码执行结果：\n'
    '[代码执行输出内容]\n\n'
    '生成的CSV文件路径：[文件路径]\n\n'
    '数据字典：\n'
    '- 列名1：[描述]\n'
    '- 列名2：[描述]\n'
    '...'
)

data_generator_criteria = textwrap.dedent(
    """\
    1. 使用CodeExecutionToolkit执行Python代码生成数据
    2. 生成多种不同格式的CSV文件，包括不同分隔符、不同命名约定
    3. 有时使用标准的健康监测命名规范，有时使用非标准或简化的命名方式
    4. 生成真实且多样化的用户数据，包括体重、饮食、活动等
    5. 确保数据包含足够的变异性和模式，以便进行统计分析
    6. 模拟不同干预措施的效果和用户响应
    7. 包含各种特征之间的相关性和交互效应
    8. 生成结构化但格式多样的CSV数据
    9. 提供清晰的数据字典和元数据
    10. 将生成的CSV保存在容易访问的路径
    11. 确保生成的数据与用户案例相关，并适合后续建模
    """
)

data_generator = make_lab_analysis_agent(
    "示例数据生成器",
    data_generator_persona,
    data_generator_example,
    data_generator_criteria,
)

# 创建工作团队
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=0.2).as_dict())

workforce = Workforce(
    '检验科数据分析专家团队',
    coordinator_agent_kwargs = {"model": model},
    task_agent_kwargs = {"model": model},
)
workforce.add_single_agent_worker(
    '检验数据生成器：创建用于分析的模拟检验数据',
    worker=lab_data_generator,
).add_single_agent_worker(
    '患者基本信息分析师：分析患者基本信息和临床数据',
    worker=patient_analyzer,
).add_single_agent_worker(
    '检验指标分析师：分析检验结果和临床意义',
    worker=lab_indicator_analyzer,
).add_single_agent_worker(
    '疾病模式分析师：从检验结果识别疾病特征模式',
    worker=disease_pattern_analyzer,
).add_single_agent_worker(
    '数据质量控制专家：评估数据质量和可靠性',
    worker=data_quality_specialist,
).add_single_agent_worker(
    '检验数据分析师：分析数据趋势和模式',
    worker=data_analyst,
).add_single_agent_worker(
    '分子诊断分析专家：分析分子检测数据',
    worker=molecular_diagnostics_analyst,
).add_single_agent_worker(
    '参考值研究专家：研究检验参考区间和临界值',
    worker=reference_value_researcher,
).add_single_agent_worker(
    '科研灵感生成器：提出创新研究方向和假设',
    worker=research_inspiration_generator,
)

# 更新处理函数
def process_lab_data_analysis(user_info: str, input_csv_path: str = None) -> str:
    """
    通过检验科数据分析专家团队处理检验数据
    
    参数：
        user_info (str): 用户提供的信息
        input_csv_path (str, optional): 用户提供的CSV数据文件路径，如果为None则使用系统生成的数据
    
    返回：
        str: 最终体重管理方案
    """
    # 构建处理流程，根据是否有用户提供的CSV决定第一步
    if input_csv_path and os.path.exists(input_csv_path):
        first_step = f"step1:使用用户提供的CSV数据文件（路径：{input_csv_path}）进行分析。尝试读取并分析数据，接受任意格式的CSV文件，无需特定结构。生成数据摘要。"
    else:
        first_step = "step1:使用示例数据生成器创建模拟的体重管理数据集，保存为CSV格式。必须使用CodeExecutionToolkit执行Python代码生成真实的CSV数据文件。"
    
    task = Task(
        content=f"通过体重管理专家团队处理此用户案例。"
        f"{first_step}"
        "step2:分析用户基本信息和健康数据。"
        "step3:分析用户的饮食结构和营养摄入。"
        "step4:根据用户情况设计个性化运动方案。"
        "step5:分析影响用户体重管理的心理因素并提供行为改变策略。"
        "step6:数据分析师必须使用CodeExecutionToolkit执行Python代码，读取" + 
        ("用户提供的" if input_csv_path and os.path.exists(input_csv_path) else "步骤1生成的") + 
        "CSV数据文件进行统计分析和可视化。需要适应任意格式的CSV，不限制列名或结构。" +
        "首先需要探索数据，然后再进行分析。"
        "step7:高级统计建模专家必须使用CodeExecutionToolkit执行Python代码，读取" + 
        ("用户提供的" if input_csv_path and os.path.exists(input_csv_path) else "步骤1生成的") + 
        "CSV数据文件构建机器学习模型，进行预测和评估。需要适应任意格式的CSV，不限制列名或结构。" +
        "首先需要探索数据和特征，再进行建模。"
        "step8:进行生物信息学分析，识别个体代谢和微生物组特征。"
        "step9:创建数字孪生模型，模拟不同干预方案的效果。"
        "step10:整合各专家意见，设计综合体重管理方案。记得所有输出使用中文。",
        additional_info=user_info,
        id="0",
    )
    
    processed_task = workforce.process_task(task)
    return processed_task.result

# 示例用户信息
example_user_info = """
检验数据分析需求：
- 目标：分析不同疾病患者的实验室检验结果，发现潜在的科研灵感和新的研究方向
- 数据类型：多种检验项目的CSV格式数据
- 关注疾病：肝病、肾病、心血管疾病、糖尿病等常见疾病
- 期望发现：检验指标间的新型关联模式、疾病早期预警指标、新型生物标志物组合
- 分析深度：需要包括基础统计分析、高级统计建模和机器学习方法
- 科研导向：希望发现具有临床转化价值的研究方向
- 数据格式要求：系统应能处理不同格式的CSV文件，包括不同的分隔符和字段命名方式
"""

# 创建示例检验CSV文件函数
def create_example_lab_csv():
    """
    创建示例检验数据CSV文件作为输入示例
    """
    code = """
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# 设置随机种子以确保可重现性
np.random.seed(42)

# 生成病人ID
num_patients = 200
patient_ids = [f'P{str(i).zfill(4)}' for i in range(1, num_patients + 1)]

# 生成检验日期 - 过去30天
end_date = datetime.now().date()
start_date = end_date - timedelta(days=30)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# 创建疾病类型
disease_categories = ['肝病', '肾病', '糖尿病', '心血管疾病', '健康对照']
disease_probs = [0.2, 0.2, 0.2, 0.2, 0.2]  # 各疾病概率
patient_diseases = np.random.choice(disease_categories, size=num_patients, p=disease_probs)
patient_disease_dict = {patient_ids[i]: patient_diseases[i] for i in range(num_patients)}

# 创建人口统计学数据
ages = np.random.normal(55, 15, num_patients)  # 平均55岁，标准差15
ages = np.clip(ages, 18, 90).astype(int)  # 限制在18-90岁

genders = np.random.choice(['男', '女'], size=num_patients)
patient_demographics = pd.DataFrame({
    '患者ID': patient_ids,
    '年龄': ages,
    '性别': genders,
    '疾病类型': patient_diseases
})

# 保存人口统计学数据
demo_path = 'patient_demographics.csv'
patient_demographics.to_csv(demo_path, index=False)
print(f"人口统计学数据已保存至: {os.path.abspath(demo_path)}")

# 设置各疾病检验指标特征
def generate_lab_values(disease_type, num_records):
    base_values = {
        # 肝功能
        'ALT': (40, 10),  # 正常值范围: 9-50 U/L
        'AST': (35, 8),   # 正常值范围: 15-40 U/L
        'GGT': (45, 12),  # 正常值范围: 10-60 U/L
        'ALP': (100, 20), # 正常值范围: 45-125 U/L
        'TBIL': (15, 5),  # 正常值范围: 5-21 umol/L
        'DBIL': (4, 1),   # 正常值范围: 0-7 umol/L
        'TP': (70, 5),    # 正常值范围: 65-85 g/L
        'ALB': (45, 3),   # 正常值范围: 40-55 g/L
        
        # 肾功能
        'CREA': (80, 15), # 正常值范围: 57-97 umol/L
        'BUN': (5, 1),    # 正常值范围: 3.1-8.0 mmol/L
        'UA': (320, 50),  # 正常值范围: 208-428 umol/L
        'eGFR': (90, 10), # 正常值范围: >90 mL/min/1.73m²
        
        # 血糖
        'GLU': (5.2, 0.5), # 正常值范围: 3.9-6.1 mmol/L
        'HbA1c': (5.5, 0.3), # 正常值范围: 4.0-6.0%
        
        # 血脂
        'CHOL': (4.5, 0.5), # 正常值范围: 2.9-5.2 mmol/L
        'TG': (1.5, 0.3),   # 正常值范围: 0.45-1.7 mmol/L
        'HDL': (1.3, 0.2),  # 正常值范围: 1.16-1.55 mmol/L
        'LDL': (2.8, 0.4),  # 正常值范围: 2.1-3.1 mmol/L
        
        # 心肌标志物
        'cTnI': (0.01, 0.005), # 正常值范围: <0.04 ng/mL
        'CK_MB': (3, 1),       # 正常值范围: 0-5 ng/mL
        'NT_proBNP': (50, 20), # 正常值范围: <125 pg/mL
        
        # 血常规
        'WBC': (6.5, 1),    # 正常值范围: 4-10 *10^9/L
        'RBC': (4.5, 0.3),  # 正常值范围: 3.5-5.5 *10^12/L
        'HGB': (140, 10),   # 正常值范围: 120-160 g/L
        'PLT': (250, 30),   # 正常值范围: 100-300 *10^9/L
        'NEUT': (60, 5),    # 正常值范围: 50-70%
        'LYMPH': (30, 4),   # 正常值范围: 20-40%
        
        # 炎症标志物
        'CRP': (3, 1),      # 正常值范围: 0-5 mg/L
        'ESR': (10, 3),     # 正常值范围: 0-15 mm/h
        'PCT': (0.05, 0.02) # 正常值范围: <0.1 ng/mL
    }
    
    # 根据疾病类型调整参数
    if disease_type == '肝病':
        # 肝功能异常
        adjustments = {
            'ALT': (120, 40),
            'AST': (100, 35),
            'GGT': (150, 50),
            'ALP': (180, 40),
            'TBIL': (35, 15),
            'DBIL': (12, 5),
            'TP': (60, 8),
            'ALB': (35, 5)
        }
    elif disease_type == '肾病':
        # 肾功能异常
        adjustments = {
            'CREA': (180, 50),
            'BUN': (12, 3),
            'UA': (450, 70),
            'eGFR': (45, 15),
            'TP': (60, 8),
            'ALB': (35, 5)
        }
    elif disease_type == '糖尿病':
        # 血糖异常
        adjustments = {
            'GLU': (9.5, 2),
            'HbA1c': (8.5, 1.5),
            'CHOL': (5.5, 1),
            'TG': (2.5, 1),
            'HDL': (0.9, 0.2),
            'LDL': (3.5, 0.8)
        }
    elif disease_type == '心血管疾病':
        # 心血管指标异常
        adjustments = {
            'CHOL': (6, 1),
            'TG': (2.8, 1.2),
            'HDL': (0.8, 0.2),
            'LDL': (4, 1),
            'cTnI': (0.1, 0.08),
            'CK_MB': (8, 3),
            'NT_proBNP': (500, 200)
        }
    elif disease_type == '健康对照':
        # 基本正常值，微小波动
        adjustments = {}
    
    # 更新基础值
    for key, value in adjustments.items():
        base_values[key] = value
    
    # 为每个检验指标生成数值
    lab_data = {}
    for indicator, (mean, std) in base_values.items():
        # 添加随机波动
        values = np.random.normal(mean, std, num_records)
        # 确保没有负值
        values = np.maximum(values, 0)
        lab_data[indicator] = values
        
    return lab_data

# 生成检验数据
lab_records = []

for patient_id in patient_ids:
    disease = patient_disease_dict[patient_id]
    
    # 每个患者有1-3条记录
    num_records = np.random.randint(1, 4)
    record_dates = np.random.choice(date_range, size=num_records, replace=False)
    record_dates = sorted(record_dates)
    
    lab_values = generate_lab_values(disease, num_records)
    
    for i, date in enumerate(record_dates):
        lab_records.append({
            '患者ID': patient_id,
            '检验日期': date,
            'ALT': lab_values['ALT'][i],
            'AST': lab_values['AST'][i],
            'GGT': lab_values['GGT'][i],
            'ALP': lab_values['ALP'][i],
            'TBIL': lab_values['TBIL'][i],
            'DBIL': lab_values['DBIL'][i],
            'TP': lab_values['TP'][i],
            'ALB': lab_values['ALB'][i],
            'CREA': lab_values['CREA'][i],
            'BUN': lab_values['BUN'][i],
            'UA': lab_values['UA'][i],
            'eGFR': lab_values['eGFR'][i],
            'GLU': lab_values['GLU'][i],
            'HbA1c': lab_values['HbA1c'][i],
            'CHOL': lab_values['CHOL'][i],
            'TG': lab_values['TG'][i],
            'HDL': lab_values['HDL'][i],
            'LDL': lab_values['LDL'][i],
            'CK_MB': lab_values['CK_MB'][i],
            'NT_proBNP': lab_values['NT_proBNP'][i],
            'WBC': lab_values['WBC'][i],
            'RBC': lab_values['RBC'][i],
            'HGB': lab_values['HGB'][i],
            'PLT': lab_values['PLT'][i],
            'NEUT': lab_values['NEUT'][i],
            'LYMPH': lab_values['LYMPH'][i],
            'CRP': lab_values['CRP'][i],
            'ESR': lab_values['ESR'][i],
            'PCT': lab_values['PCT'][i]
        })

# 创建DataFrame
df = pd.DataFrame(lab_records)
df.index.name = 'date'

# 用户基本信息
user_id = "user_001"
age = 32
gender = "male"
height = 178  # cm
initial_weight = 80  # kg
target_weight = 75  # kg

# 生成体重数据 - 添加一些真实波动和缓慢增加趋势
weight_trend = np.linspace(initial_weight, initial_weight + 5, len(df))  # 逐渐增加5kg
daily_fluctuation = np.random.normal(0, 0.3, len(df))  # 日常波动
weekend_effect = np.array([0.2 if d.weekday() >= 5 else 0 for d in df.index])  # 周末增加
weight = weight_trend + daily_fluctuation + weekend_effect
df['weight'] = np.round(weight, 1)

# 生成卡路里摄入量
base_calories = 2200  # 基础卡路里
# 工作日和周末的不同模式
daily_pattern = np.array([
    base_calories - 200 if d.weekday() < 5 else base_calories + 400 
    for d in df.index
])
# 添加随机变化
calorie_variation = np.random.normal(0, 200, len(df))
calories = daily_pattern + calorie_variation
df['calories_intake'] = np.round(calories).astype(int)

# 生成蛋白质、碳水和脂肪摄入
df['protein_g'] = np.round(df['calories_intake'] * 0.15 / 4 + np.random.normal(0, 10, len(df)))
df['carbs_g'] = np.round(df['calories_intake'] * 0.55 / 4 + np.random.normal(0, 20, len(df)))
df['fat_g'] = np.round(df['calories_intake'] * 0.30 / 9 + np.random.normal(0, 8, len(df)))

# 生成活动水平(步数)
base_steps = 3000  # 基础步数
# 工作日和周末的不同模式
daily_steps = np.array([
    base_steps if d.weekday() < 5 else base_steps + np.random.randint(1000, 3000) 
    for d in df.index
])
# 偶尔的高活动日
high_activity_days = np.random.randint(0, len(df), size=15)  # 15天高活动
for day in high_activity_days:
    daily_steps[day] += np.random.randint(4000, 8000)
df['steps'] = daily_steps.astype(int)

# 生成睡眠时间
base_sleep = 6  # 基础睡眠时间(小时)
# 工作日和周末的不同模式
sleep_pattern = np.array([
    base_sleep if d.weekday() < 5 else base_sleep + 1.5
    for d in df.index
])
# 添加随机变化
sleep_variation = np.random.normal(0, 0.7, len(df))
df['sleep_hours'] = np.round(sleep_pattern + sleep_variation, 1)

# 生成压力水平 (1-10)
base_stress = 7  # 基础压力水平
# 工作日和周末的不同模式
stress_pattern = np.array([
    base_stress if d.weekday() < 5 else base_stress - 2
    for d in df.index
])
# 添加随机变化与偶尔的高压力日
stress_variation = np.random.normal(0, 1, len(df))
high_stress_days = np.random.randint(0, len(df), size=20)  # 20天高压力
for day in high_stress_days:
    stress_variation[day] += 2
df['stress_level'] = np.clip(np.round(stress_pattern + stress_variation), 1, 10).astype(int)

# 生成水摄入量
base_water = 1000  # 基础水摄入量(ml)
water_variation = np.random.normal(0, 300, len(df))
df['water_ml'] = np.round(base_water + water_variation, -1).astype(int)

# 生成锻炼分钟数
# 大多数日子没有锻炼
exercise_minutes = np.zeros(len(df))
# 随机选择一些日子进行锻炼
exercise_days = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
exercise_minutes[exercise_days == 1] = np.random.randint(20, 90, size=sum(exercise_days))
df['exercise_minutes'] = exercise_minutes.astype(int)

# 生成锻炼类型
exercise_types = ['无', '跑步', '健身房', '游泳', '骑行', '瑜伽', '篮球']
# 为每天分配锻炼类型，没有锻炼的日子为"无"
df['exercise_type'] = ['无'] * len(df)
for i in range(len(df)):
    if df.loc[df.index[i], 'exercise_minutes'] > 0:
        df.loc[df.index[i], 'exercise_type'] = np.random.choice(exercise_types[1:])

# 生成早餐、午餐、晚餐的规律性 (0-不规律, 1-规律)
meal_regularity = np.random.choice([0, 1], size=(len(df), 3), p=[0.3, 0.7])
df['breakfast_regular'] = meal_regularity[:, 0]
df['lunch_regular'] = meal_regularity[:, 1]
df['dinner_regular'] = meal_regularity[:, 2]

# 生成晚餐时间
base_dinner_time = 20  # 基础晚餐时间 (24小时制)
dinner_variation = np.random.normal(0, 1, len(df))
df['dinner_time'] = np.clip(np.round(base_dinner_time + dinner_variation, 1), 17, 23)

# 生成加工食品比例
base_processed_food = 0.6  # 基础加工食品比例
processed_variation = np.random.normal(0, 0.15, len(df))
df['processed_food_ratio'] = np.clip(base_processed_food + processed_variation, 0.1, 1.0)
df['processed_food_ratio'] = np.round(df['processed_food_ratio'], 2)

# 生成饮酒量(标准杯)
alcohol = np.zeros(len(df))
# 偶尔饮酒的日子
alcohol_days = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
alcohol[alcohol_days == 1] = np.random.randint(1, 5, size=sum(alcohol_days))
df['alcohol_units'] = alcohol.astype(int)

# 生成零食卡路里
base_snack = 300  # 基础零食卡路里
# 压力高的日子零食增加
snack_calories = base_snack + df['stress_level'] * 30 + np.random.normal(0, 100, len(df))
df['snack_calories'] = np.round(snack_calories).astype(int)

# 添加用户ID和基本信息作为常量列
df['user_id'] = user_id
df['age'] = age
df['gender'] = gender
df['height_cm'] = height
df['target_weight_kg'] = target_weight

# 计算BMI
df['bmi'] = np.round(df['weight'] / ((height/100) ** 2), 1)

# 计算目标差异
df['weight_to_target'] = np.round(df['weight'] - target_weight, 1)

# 保存CSV
output_path = 'weight_management_data.csv'
df.to_csv(output_path)
print(f"数据已保存到: {os.path.abspath(output_path)}")
print(f"数据形状: {df.shape}")
print("\\n数据预览:")
print(df.head())
print("\\n数据描述统计:")
print(df.describe())

# 返回文件路径
print(f"\\n数据文件路径: {os.path.abspath(output_path)}")
"""
    
    return code

# 创建验证CSV文件函数
def validate_csv_file(file_path: str) -> str:
    """
    生成用于验证检验CSV文件的Python代码，支持任意格式的CSV文件
    
    参数:
        file_path (str): CSV文件路径
        
    返回:
        str: 生成的Python代码
    """
    code = """
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# 尝试以多种方式读取CSV文件
def try_read_csv(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None
    
    # 尝试多种读取方式
    try:
        return pd.read_csv(file_path)
    except:
        pass
        
    try:
        return pd.read_csv(file_path, sep=None, engine='python')
    except:
        pass
        
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        pass
        
    try:
        return pd.read_csv(file_path, encoding='latin1')
    except:
        pass
    
    # 尝试自动检测分隔符
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            
        # 尝试多种编码
        text = None
        for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
            try:
                text = content.decode(encoding)
                break
            except:
                continue
                
        if text is None:
            return None
            
        # 检测分隔符
        sample_lines = text.split('\\n')[:10]
        seps = [',', ';', '\\t', '|', ' ']
        sep_counts = {}
        
        for sep in seps:
            sep_counts[sep] = sum(line.count(sep) for line in sample_lines if line)
            
        best_sep = max(sep_counts, key=sep_counts.get)
        if sep_counts[best_sep] > 0:
            return pd.read_csv(StringIO(text), sep=best_sep)
    except:
        pass
        
    return None

# 主要分析过程
try:
    print(f"分析检验CSV文件: """ + file_path + """")
    
    # 读取文件
    df = try_read_csv(""" + f'"{file_path}"' + """)
    
    if df is None:
        print("无法读取CSV文件，但将继续分析过程")
        info = {
            "file_path": """ + f'"{file_path}"' + """,
            "error": "无法读取文件",
            "message": "文件格式可能有问题，但将尝试继续分析"
        }
        
        with open('csv_analysis_info.txt', 'w') as f:
            f.write(str(info))
            
        print("信息已保存")
    else:
        # 显示基本信息
        print(f"文件验证成功!")
        print(f"记录数: {len(df)}")
        print(f"特征数: {len(df.columns)}")
        
        # 显示列信息
        print("\\n数据列:")
        for col in df.columns:
            print(f"- {col}: {df[col].dtype}")
            
        # 查找日期列
        date_cols = []
        date_patterns = ['date', 'time', 'day', 'month', 'year', '日期', '时间', '日', '月', '年', '检验日期', 'test_date']
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns) or df[col].dtype == 'datetime64[ns]':
                date_cols.append(col)
                
        # 查找未识别的日期列
        for col in df.columns:
            if col not in date_cols and df[col].dtype == object:
                try:
                    sample = df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist() if len(df[col].dropna()) > 0 else []
                    if sample and all(isinstance(x, str) for x in sample):
                        pd.to_datetime(sample)
                        date_cols.append(col)
                except:
                    pass
        
        # 显示日期列
        if date_cols:
            print("\\n可能的日期/时间列:")
            for col in date_cols:
                try:
                    temp = pd.to_datetime(df[col], errors='coerce')
                    if not temp.isna().all():
                        print(f"- {col}: {temp.min()} 至 {temp.max()}")
                    else:
                        print(f"- {col}: 无法解析为日期")
                except:
                    print(f"- {col}: 无法解析为日期")
        
        # 查找ID列
        id_cols = []
        id_patterns = ['id', 'patient', 'subject', 'person', '患者', '编号', 'patient_id', '患者ID']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in id_patterns):
                id_cols.append(col)
        
        # 显示ID列
        if id_cols:
            print("\\n找到可能的ID列:")
            for col in id_cols:
                unique_count = df[col].nunique()
                print(f"- {col}: {unique_count} 个唯一值")
        
        # 查找数值型检验指标列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 排除ID和日期列等非检验指标
        exclude_patterns = ['id', 'date', 'time', 'age', 'year', '年龄', '日期', '时间']
        lab_cols = [col for col in numeric_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        # 显示检验指标列的基本统计
        if lab_cols:
            print(f"\\n找到 {len(lab_cols)} 个可能的检验指标列:")
            # 只显示前10个指标避免输出过长
            for col in lab_cols[:10]:
                print(f"- {col}:")
                print(f"  平均值: {df[col].mean():.2f}")
                print(f"  中位数: {df[col].median():.2f}")
                print(f"  最小值: {df[col].min():.2f}")
                print(f"  最大值: {df[col].max():.2f}")
            
            if len(lab_cols) > 10:
                print(f"  ... 还有 {len(lab_cols) - 10} 个指标未显示")
        
        # 缺失值检查
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\\n缺失值统计:")
            for col in missing[missing > 0].index:
                print(f"- {col}: {missing[col]} 缺失 ({missing[col]/len(df)*100:.2f}%)")
        else:
            print("\\n数据完整，无缺失值")
        
        # 初步异常值检查（超出正常医学参考范围的数据）
        # 这里只是示例性地检查几个常见指标
        reference_ranges = {
            'ALT': (0, 50),      # ALT正常范围0-50 U/L
            'AST': (0, 40),      # AST正常范围0-40 U/L
            'CREA': (44, 133),   # 肌酐正常范围44-133 umol/L
            'GLU': (3.9, 6.1),   # 空腹血糖正常范围3.9-6.1 mmol/L
            'WBC': (4, 10),      # 白细胞正常范围4-10 *10^9/L
            'HGB': (120, 160)    # 血红蛋白正常范围120-160 g/L
        }
        
        # 检查所有列名中可能包含这些指标的列
        print("\\n潜在异常值检查:")
        for ref_name, (lower, upper) in reference_ranges.items():
            matching_cols = [col for col in lab_cols if ref_name.lower() in col.lower()]
            for col in matching_cols:
                if df[col].notnull().any():
                    below = (df[col] < lower).sum()
                    above = (df[col] > upper).sum()
                    if below > 0 or above > 0:
                        print(f"- {col}: {below} 个值低于 {lower}, {above} 个值高于 {upper}")
        
        # 保存分析结果
        info = {
            "file_path": """ + f'"{file_path}"' + """,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "possible_date_columns": date_cols,
            "possible_id_columns": id_cols,
            "possible_lab_columns": lab_cols
        }
        
        print("\\nCSV文件有效，可以进行后续分析")
        print("\\n文件分析信息:")
        print(info)
        
        with open('csv_analysis_info.txt', 'w') as f:
            f.write(str(info))
        
        print("\\n文件分析信息已保存到: csv_analysis_info.txt")
        
except Exception as e:
    print(f"发生错误: {str(e)}")
    print("尽管出现错误，仍将尝试继续分析过程")
"""
    
    return code

# 创建命令行入口点
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='检验科数据分析专家团队系统')
    parser.add_argument('--user_info', type=str, help='用户需求信息文件路径', default=None)
    parser.add_argument('--csv_file', type=str, help='用户提供的检验CSV数据文件路径', default=None)
    parser.add_argument('--generate_sample_csv', action='store_true', help='是否只生成示例检验CSV数据而不执行完整分析')
    args = parser.parse_args()
    
    # 只生成示例CSV
    if args.generate_sample_csv:
        from camel.toolkits import CodeExecutionToolkit
        
        print("正在生成示例检验CSV数据文件...")
        code_toolkit = CodeExecutionToolkit(verbose=True)
        result = code_toolkit.execute_code(create_example_lab_csv())
        print("示例检验CSV数据生成完成。")
        exit(0)
    
    # 验证用户提供的CSV文件
    if args.csv_file:
        if not os.path.exists(args.csv_file):
            print(f"错误：提供的CSV文件 '{args.csv_file}' 不存在")
            exit(1)
        
        from camel.toolkits import CodeExecutionToolkit
        
        print(f"正在验证CSV文件: {args.csv_file}...")
        code_toolkit = CodeExecutionToolkit(verbose=True)
        result = code_toolkit.execute_code(validate_csv_file(args.csv_file))
        
        print("CSV文件验证完成，准备开始分析...")
    
    # 获取用户信息
    user_text = ""
    if args.user_info:
        if not os.path.exists(args.user_info):
            print(f"错误：提供的用户需求信息文件 '{args.user_info}' 不存在")
            exit(1)
        
        with open(args.user_info, 'r', encoding='utf-8') as f:
            user_text = f.read()
    else:
        user_text = example_user_info
    
    # 执行分析
    print("正在进行检验数据分析...")
    result = process_lab_data_analysis(user_text, args.csv_file)
    print("\n===== 分析结果 =====\n")
    print(result)
