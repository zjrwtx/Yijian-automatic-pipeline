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
     FunctionTool, SearchToolkit,PubMedToolkit,GoogleScholarToolkit,ArxivToolkit,SemanticScholarToolkit
     ,FileWriteToolkit,
     BrowserToolkit,
     RetrievalToolkit,
     CodeExecutionToolkit,
     
     
)
import asyncio
from camel.types import ModelPlatformType, ModelType  
from camel.societies.workforce import Workforce
from camel.configs import DeepSeekConfig,ChatGPTConfig,GeminiConfig

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
    *CodeExecutionToolkit(verbose=True).get_tools()
]
def make_weight_management_agent(
    role: str,
    persona: str,
    example_output: str,
    criteria: str,
) -> ChatAgent:
    msg_content = textwrap.dedent(
        f"""\
        您是体重管理领域的专业人员。
        您的角色：{role}
        您的职责和特点：{persona}
        输出格式示例：
        {example_output}
        您的分析标准：
        {criteria}
        """
    )

    sys_msg = BaseMessage.make_assistant_message(
        role_name="体重管理专业人员",
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

# 创建用户信息分析师
profile_analyzer_persona = (
    '您是专门分析用户基本情况和健康数据的体重管理分析师。'
    '您专注于捕捉关键健康指标、生活习惯和体重历史记录。'
    '您需要将内容保存为txt文件到本地'
)

profile_analyzer_example = (
    '用户信息分析\n'
    '基本信息：\n'
    '- 年龄：[用户年龄]\n'
    '- 性别：[用户性别]\n'
    '- 身高：[用户身高]\n'
    '- 当前体重：[当前体重]\n'
    '- BMI指数：[计算结果]\n'
    '体重历史：\n'
    '- [过去体重变化记录]\n'
    '健康状况：\n'
    '- [相关健康状况概述]\n'
    '生活习惯：\n'
    '- 饮食习惯：[描述]\n'
    '- 运动习惯：[描述]\n'
    '- 睡眠模式：[描述]\n'
)

profile_analyzer_criteria = textwrap.dedent(
    """\
    1. 收集完整的用户基本信息
    2. 计算BMI和其他相关健康指标
    3. 记录用户的体重历史变化
    4. 分析当前生活习惯模式
    5. 识别可能影响体重管理的健康因素
    """
)

profile_analyzer = make_weight_management_agent(
    "用户信息分析师",
    profile_analyzer_persona,
    profile_analyzer_example,
    profile_analyzer_criteria,
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

diet_analyzer = make_weight_management_agent(
    "饮食分析师",
    diet_analyzer_persona,
    diet_analyzer_example,
    diet_analyzer_criteria,
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

exercise_planner = make_weight_management_agent(
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

behavior_specialist = make_weight_management_agent(
    "行为心理专家",
    behavior_specialist_persona,
    behavior_specialist_example,
    behavior_specialist_criteria,
)

# 创建数据分析师
data_analyst_persona = (
    '您是专门分析用户体重变化数据和趋势的数据分析师。'
    '您使用统计方法识别影响体重的关键因素和模式。'
    '您能够使用Python生成数据分析代码，处理CSV数据并可视化结果。'
    '您需要使用CodeExecutionToolkit执行您的Python分析代码。'
    '您需要将内容保存为txt文件到本地'
)

data_analyst_example = (
    '数据分析报告\n'
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
    2. 分析用户体重变化的时间序列数据
    3. 识别影响体重的关键变量和因素
    4. 建立预测模型评估不同行为的潜在影响
    5. 提供基于数据的可行性建议
    6. 设计有效的数据收集和监测策略
    7. 生成可视化结果展示分析发现
    8. 确保代码能正确读取CSV数据文件
    """
)

data_analyst = make_weight_management_agent(
    "数据分析师",
    data_analyst_persona,
    data_analyst_example,
    data_analyst_criteria,
)

# 创建高级统计建模专家
advanced_modeling_persona = (
    '您是体重管理领域的高级统计建模专家，专长于开发复杂的预测模型和多因素分析。'
    '您运用机器学习、深度学习和高级统计方法构建创新的体重管理解决方案。'
    '您能够使用Python实现各种统计模型和机器学习算法，处理CSV格式的数据。'
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

advanced_modeler = make_weight_management_agent(
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

bioinformatics_analyst = make_weight_management_agent(
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

digital_twin_specialist = make_weight_management_agent(
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

plan_designer = make_weight_management_agent(
    "综合方案设计师",
    plan_designer_persona,
    plan_designer_example,
    plan_designer_criteria,
)

# 创建示例数据生成器
data_generator_persona = (
    '您是体重管理研究的数据科学家，专门负责生成模拟数据用于模型训练和测试。'
    '您能够创建合成但真实的用户数据，包括体重变化、饮食记录、活动水平和生理指标等。'
    '您需要使用CodeExecutionToolkit来执行Python代码，生成CSV格式的数据集。'
    '您的代码应该生成完整的、真实的、适合后续分析的数据。'
    '您需要将内容保存为txt文件到本地，并确保生成的CSV文件路径明确。'
)

data_generator_example = (
    '数据生成报告\n\n'
    '生成的数据集描述：\n'
    '- 数据集目的：[描述]\n'
    '- 样本量：[数量]\n'
    '- 时间范围：[范围]\n'
    '- 特征列表：[列表]\n\n'
    '数据分布特征：\n'
    '- 体重变化模式：[描述]\n'
    '- 干预响应模式：[描述]\n'
    '- 噪声与变异性：[描述]\n\n'
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
    2. 生成真实且多样化的用户数据，包括体重、饮食、活动等
    3. 确保数据包含足够的变异性和模式，以便进行统计分析
    4. 模拟不同干预措施的效果和用户响应
    5. 包含各种特征之间的相关性和交互效应
    6. 生成干净且结构化的CSV数据
    7. 提供清晰的数据字典和元数据
    8. 将生成的CSV保存在容易访问的路径
    9. 确保生成的数据与用户案例相关，并适合后续建模
    """
)

data_generator = make_weight_management_agent(
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
    '体重管理专家团队',
    coordinator_agent_kwargs = {"model": model},
    task_agent_kwargs = {"model": model},
)
workforce.add_single_agent_worker(
    '示例数据生成器：创建用于分析的体重管理数据',
    worker=data_generator,
).add_single_agent_worker(
    '用户信息分析师：分析用户基本信息和健康数据',
    worker=profile_analyzer,
).add_single_agent_worker(
    '饮食分析师：分析用户饮食结构和营养摄入',
    worker=diet_analyzer,
).add_single_agent_worker(
    '运动规划师：设计个性化运动方案',
    worker=exercise_planner,
).add_single_agent_worker(
    '行为心理专家：分析心理因素并提供行为改变策略',
    worker=behavior_specialist,
).add_single_agent_worker(
    '数据分析师：分析用户体重变化数据和趋势',
    worker=data_analyst,
).add_single_agent_worker(
    '高级统计建模专家：构建复杂预测模型和多因素分析',
    worker=advanced_modeler,
).add_single_agent_worker(
    '生物信息学分析专家：分析基因组和微生物组数据',
    worker=bioinformatics_analyst,
).add_single_agent_worker(
    '数字孪生模拟专家：创建个体虚拟生理模型',
    worker=digital_twin_specialist,
).add_single_agent_worker(
    '综合方案设计师：整合各专家意见设计综合方案',
    worker=plan_designer,
)

# 更新处理函数
def process_weight_management_case(user_info: str, input_csv_path: str = None) -> str:
    """
    通过体重管理专家团队处理用户信息
    
    参数：
        user_info (str): 用户提供的信息
        input_csv_path (str, optional): 用户提供的CSV数据文件路径，如果为None则使用系统生成的数据
    
    返回：
        str: 最终体重管理方案
    """
    # 构建处理流程，根据是否有用户提供的CSV决定第一步
    if input_csv_path and os.path.exists(input_csv_path):
        first_step = f"step1:使用用户提供的CSV数据文件（路径：{input_csv_path}）进行分析。验证并检查数据格式，生成数据摘要。"
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
        "CSV数据文件进行统计分析和可视化。"
        "step7:高级统计建模专家必须使用CodeExecutionToolkit执行Python代码，读取" + 
        ("用户提供的" if input_csv_path and os.path.exists(input_csv_path) else "步骤1生成的") + 
        "CSV数据文件构建机器学习模型，进行预测和评估。"
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
个人基本信息：
- 姓名：张先生
- 年龄：32岁
- 性别：男
- 身高：178cm
- 当前体重：85kg
- 职业：IT程序员，工作日久坐8-10小时

体重历史：
- 大学毕业时（22岁）：70kg
- 28岁时：75kg
- 30岁时：80kg
- 最近两年增重5kg

健康状况：
- 无慢性疾病
- 轻度高血压（130/85mmHg）
- 最近体检显示轻度脂肪肝
- 经常感到疲劳，尤其是下午

生活习惯：
- 饮食：工作日多外卖，周末偶尔下厨；喜欢高碳水、高脂肪食物；每周点外卖4-5次
- 早餐常吃油条、豆浆或者不吃
- 午餐通常是快餐（盖浇饭、炒饭等）
- 晚餐较晚（晚上8点后），常吃得较多
- 零食：喜欢碳酸饮料、薯片、巧克力等，工作压力大时会吃更多零食
- 饮水量：每天约1000ml
- 饮酒：社交场合偶尔饮酒，每周1-2次

- 运动：几乎没有固定运动习惯；偶尔周末打打篮球（每月1-2次）
- 步数：工作日平均3000步/天
- 尝试过健身但坚持不下来，最长坚持过2个月

- 睡眠：工作日平均6小时/晚，周末7-8小时
- 经常熬夜到凌晨1点以后
- 睡眠质量一般，偶尔失眠

心理状态：
- 工作压力大，经常感到焦虑
- 有通过进食缓解压力的习惯
- 对体重增加感到忧虑但缺乏足够动力改变
- 曾尝试多次减重但都半途而废

目标：
- 希望在6个月内减至75kg
- 改善体质，增加精力
- 建立健康可持续的生活方式
- 能够长期保持健康体重
"""

# 创建示例CSV文件路径函数
def create_example_weight_csv():
    """
    创建示例体重管理CSV文件作为输入示例
    """
    code = """
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 设置随机种子以确保可重现性
np.random.seed(42)

# 创建日期范围 - 过去180天
end_date = datetime.now().date()
start_date = end_date - timedelta(days=180)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# 初始化数据框
data = pd.DataFrame(index=dates)
data.index.name = 'date'

# 用户基本信息
user_id = "user_001"
age = 32
gender = "male"
height = 178  # cm
initial_weight = 80  # kg
target_weight = 75  # kg

# 生成体重数据 - 添加一些真实波动和缓慢增加趋势
weight_trend = np.linspace(initial_weight, initial_weight + 5, len(dates))  # 逐渐增加5kg
daily_fluctuation = np.random.normal(0, 0.3, len(dates))  # 日常波动
weekend_effect = np.array([0.2 if d.weekday() >= 5 else 0 for d in dates])  # 周末增加
weight = weight_trend + daily_fluctuation + weekend_effect
data['weight'] = np.round(weight, 1)

# 生成卡路里摄入量
base_calories = 2200  # 基础卡路里
# 工作日和周末的不同模式
daily_pattern = np.array([
    base_calories - 200 if d.weekday() < 5 else base_calories + 400 
    for d in dates
])
# 添加随机变化
calorie_variation = np.random.normal(0, 200, len(dates))
calories = daily_pattern + calorie_variation
data['calories_intake'] = np.round(calories).astype(int)

# 生成蛋白质、碳水和脂肪摄入
data['protein_g'] = np.round(data['calories_intake'] * 0.15 / 4 + np.random.normal(0, 10, len(dates)))
data['carbs_g'] = np.round(data['calories_intake'] * 0.55 / 4 + np.random.normal(0, 20, len(dates)))
data['fat_g'] = np.round(data['calories_intake'] * 0.30 / 9 + np.random.normal(0, 8, len(dates)))

# 生成活动水平(步数)
base_steps = 3000  # 基础步数
# 工作日和周末的不同模式
daily_steps = np.array([
    base_steps if d.weekday() < 5 else base_steps + np.random.randint(1000, 3000) 
    for d in dates
])
# 偶尔的高活动日
high_activity_days = np.random.randint(0, len(dates), size=15)  # 15天高活动
for day in high_activity_days:
    daily_steps[day] += np.random.randint(4000, 8000)
data['steps'] = daily_steps.astype(int)

# 生成睡眠时间
base_sleep = 6  # 基础睡眠时间(小时)
# 工作日和周末的不同模式
sleep_pattern = np.array([
    base_sleep if d.weekday() < 5 else base_sleep + 1.5
    for d in dates
])
# 添加随机变化
sleep_variation = np.random.normal(0, 0.7, len(dates))
data['sleep_hours'] = np.round(sleep_pattern + sleep_variation, 1)

# 生成压力水平 (1-10)
base_stress = 7  # 基础压力水平
# 工作日和周末的不同模式
stress_pattern = np.array([
    base_stress if d.weekday() < 5 else base_stress - 2
    for d in dates
])
# 添加随机变化与偶尔的高压力日
stress_variation = np.random.normal(0, 1, len(dates))
high_stress_days = np.random.randint(0, len(dates), size=20)  # 20天高压力
for day in high_stress_days:
    stress_variation[day] += 2
data['stress_level'] = np.clip(np.round(stress_pattern + stress_variation), 1, 10).astype(int)

# 生成水摄入量
base_water = 1000  # 基础水摄入量(ml)
water_variation = np.random.normal(0, 300, len(dates))
data['water_ml'] = np.round(base_water + water_variation, -1).astype(int)

# 生成锻炼分钟数
# 大多数日子没有锻炼
exercise_minutes = np.zeros(len(dates))
# 随机选择一些日子进行锻炼
exercise_days = np.random.choice([0, 1], size=len(dates), p=[0.85, 0.15])
exercise_minutes[exercise_days == 1] = np.random.randint(20, 90, size=sum(exercise_days))
data['exercise_minutes'] = exercise_minutes.astype(int)

# 生成锻炼类型
exercise_types = ['无', '跑步', '健身房', '游泳', '骑行', '瑜伽', '篮球']
# 为每天分配锻炼类型，没有锻炼的日子为"无"
data['exercise_type'] = ['无'] * len(dates)
for i in range(len(dates)):
    if data.loc[data.index[i], 'exercise_minutes'] > 0:
        data.loc[data.index[i], 'exercise_type'] = np.random.choice(exercise_types[1:])

# 生成早餐、午餐、晚餐的规律性 (0-不规律, 1-规律)
meal_regularity = np.random.choice([0, 1], size=(len(dates), 3), p=[0.3, 0.7])
data['breakfast_regular'] = meal_regularity[:, 0]
data['lunch_regular'] = meal_regularity[:, 1]
data['dinner_regular'] = meal_regularity[:, 2]

# 生成晚餐时间
base_dinner_time = 20  # 基础晚餐时间 (24小时制)
dinner_variation = np.random.normal(0, 1, len(dates))
data['dinner_time'] = np.clip(np.round(base_dinner_time + dinner_variation, 1), 17, 23)

# 生成加工食品比例
base_processed_food = 0.6  # 基础加工食品比例
processed_variation = np.random.normal(0, 0.15, len(dates))
data['processed_food_ratio'] = np.clip(base_processed_food + processed_variation, 0.1, 1.0)
data['processed_food_ratio'] = np.round(data['processed_food_ratio'], 2)

# 生成饮酒量(标准杯)
alcohol = np.zeros(len(dates))
# 偶尔饮酒的日子
alcohol_days = np.random.choice([0, 1], size=len(dates), p=[0.8, 0.2])
alcohol[alcohol_days == 1] = np.random.randint(1, 5, size=sum(alcohol_days))
data['alcohol_units'] = alcohol.astype(int)

# 生成零食卡路里
base_snack = 300  # 基础零食卡路里
# 压力高的日子零食增加
snack_calories = base_snack + data['stress_level'] * 30 + np.random.normal(0, 100, len(dates))
data['snack_calories'] = np.round(snack_calories).astype(int)

# 添加用户ID和基本信息作为常量列
data['user_id'] = user_id
data['age'] = age
data['gender'] = gender
data['height_cm'] = height
data['target_weight_kg'] = target_weight

# 计算BMI
data['bmi'] = np.round(data['weight'] / ((height/100) ** 2), 1)

# 计算目标差异
data['weight_to_target'] = np.round(data['weight'] - target_weight, 1)

# 保存CSV
output_path = 'weight_management_data.csv'
data.to_csv(output_path)
print(f"数据已保存到: {os.path.abspath(output_path)}")
print(f"数据形状: {data.shape}")
print("\\n数据预览:")
print(data.head())
print("\\n数据描述统计:")
print(data.describe())

# 返回文件路径
print(f"\\n数据文件路径: {os.path.abspath(output_path)}")
"""
    
    return code

# 创建验证CSV文件函数
def validate_csv_file(file_path: str) -> dict:
    """
    验证用户提供的CSV文件是否符合要求并返回基本信息
    
    参数:
        file_path (str): CSV文件路径
        
    返回:
        dict: 包含验证结果和基本统计信息
    """
    code = f"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    # 检查文件是否存在
    if not os.path.exists('{file_path}'):
        print("错误: 文件不存在")
        exit(1)
        
    # 尝试加载CSV文件
    df = pd.read_csv('{file_path}')
    
    # 基本信息
    print("文件验证成功!")
    print(f"记录数: {{len(df)}}")
    print(f"特征数: {{len(df.columns)}}")
    print("\\n数据列:")
    for col in df.columns:
        print(f"- {{col}}: {{df[col].dtype}}")
    
    # 检查是否有日期列
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or df[col].dtype == 'datetime64[ns]']
    if date_cols:
        print("\\n时间范围:")
        for col in date_cols:
            try:
                if df[col].dtype != 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
                print(f"- {{col}}: {{df[col].min()}} 至 {{df[col].max()}}")
            except:
                print(f"- {{col}}: 无法解析为日期")
    
    # 检查是否有体重列
    weight_cols = [col for col in df.columns if 'weight' in col.lower()]
    if weight_cols:
        print("\\n体重数据统计:")
        for col in weight_cols:
            print(f"- {{col}}:")
            print(f"  平均值: {{df[col].mean():.2f}}")
            print(f"  最小值: {{df[col].min()}}")
            print(f"  最大值: {{df[col].max()}}")
    
    # 基本数据统计
    print("\\n数据统计摘要:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().to_string())
    
    # 检查缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\\n缺失值统计:")
        for col in missing[missing > 0].index:
            print(f"- {{col}}: {{missing[col]}} 缺失 ({{missing[col]/len(df)*100:.2f}}%)")
    else:
        print("\\n数据完整，无缺失值")
    
    # 返回结果
    print("\\nCSV文件有效，可以进行后续分析")
    
except Exception as e:
    print(f"错误: {{str(e)}}")
    exit(1)
"""
    
    return code

# 创建命令行入口点
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='体重管理专家团队分析系统')
    parser.add_argument('--user_info', type=str, help='用户信息文件路径', default=None)
    parser.add_argument('--csv_file', type=str, help='用户提供的CSV数据文件路径', default=None)
    parser.add_argument('--generate_sample_csv', action='store_true', help='是否只生成示例CSV数据而不执行完整分析')
    args = parser.parse_args()
    
    # 只生成示例CSV
    if args.generate_sample_csv:
        from camel.toolkits import CodeExecutionToolkit
        
        print("正在生成示例CSV数据文件...")
        code_toolkit = CodeExecutionToolkit(verbose=True)
        result = code_toolkit.execute_code(create_example_weight_csv())
        print("示例CSV数据生成完成。")
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
            print(f"错误：提供的用户信息文件 '{args.user_info}' 不存在")
            exit(1)
        
        with open(args.user_info, 'r', encoding='utf-8') as f:
            user_text = f.read()
    else:
        user_text = example_user_info
    
    # 执行分析
    print("正在进行体重管理分析...")
    result = process_weight_management_case(user_text, args.csv_file)
    print("\n===== 分析结果 =====\n")
    print(result)