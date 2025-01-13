from getpass import getpass
import os
import textwrap
from typing import List, Dict

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.tasks import Task
from camel.toolkits import FunctionTool, SearchToolkit
from camel.types import ModelPlatformType, ModelType
from camel.societies.workforce import Workforce

# Configure API keys
openai_api_key = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = openai_api_key

def make_medical_agent(
    role: str,
    persona: str,
    example_output: str,
    criteria: str,
) -> ChatAgent:
    msg_content = textwrap.dedent(
        f"""\
        You are a medical professional in the laboratory department.
        Your role: {role}
        Your persona and responsibilities: {persona}
        Example of your output format:
        {example_output}
        Your analysis criteria:
        {criteria}
        """
    )

    sys_msg = BaseMessage.make_assistant_message(
        role_name="Medical Laboratory Professional",
        content=msg_content,
    )

    model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen2.5:0.5b",
    # Ensure using ollama verison >= 0.5.1 to use structured output feature
    model_config_dict={"temperature": 0.4,"max_tokens":4090},
)

    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
    )

    return agent

# Create Clinical Record Summarizer
summarizer_persona = (
    'You are a medical scribe specialized in converting doctor-patient '
    'conversations into structured clinical notes. You focus on capturing '
    'key medical information, patient history, and clinical observations.'
)

summarizer_example = (
    'CLINICAL SUMMARY\n'
    'Patient Information:\n'
    '- Name: [Patient Name]\n'
    '- Date: [Visit Date]\n'
    'Chief Complaint: [Primary symptoms]\n'
    'History of Present Illness: [Detailed description]\n'
    'Current Symptoms: [List of current symptoms]\n'
    'Relevant Medical History: [Past medical conditions]'
)

summarizer_criteria = textwrap.dedent(
    """\
    1. Capture all relevant medical information
    2. Maintain medical terminology accuracy
    3. Follow standard clinical documentation format
    4. Include temporal information about symptoms
    5. Note any allergies or medications mentioned
    """
)

clinical_summarizer = make_medical_agent(
    "Clinical Record Summarizer",
    summarizer_persona,
    summarizer_example,
    summarizer_criteria,
)

# Create Clinical Analyzer
analyzer_persona = (
    'You are a clinical analyst who specializes in analyzing patient '
    'symptoms and medical histories to identify potential conditions and '
    'diseases. You use pattern recognition and medical knowledge to '
    'suggest possible diagnoses.'
)

analyzer_example = (
    'CLINICAL ANALYSIS\n'
    'Primary Symptoms:\n'
    '- [Symptom 1] - Duration and severity\n'
    '- [Symptom 2] - Duration and severity\n'
    'Potential Conditions:\n'
    '1. [Condition 1] - Primary consideration because...\n'
    '2. [Condition 2] - Secondary consideration because...\n'
    'Risk Factors Present:\n'
    '- [Risk factor 1]\n'
    '- [Risk factor 2]'
)

analyzer_criteria = textwrap.dedent(
    """\
    1. Identify primary and secondary symptoms
    2. List potential conditions in order of likelihood
    3. Consider patient demographics and risk factors
    4. Note any red flags or urgent concerns
    5. Suggest differential diagnoses
    """
)

clinical_analyzer = make_medical_agent(
    "Clinical Analyzer",
    analyzer_persona,
    analyzer_example,
    analyzer_criteria,
)



recommender_persona = (
    'You are a laboratory test specialist who recommends appropriate '
    'diagnostic tests based on clinical presentations. You stay updated '
    'with current testing guidelines and protocols.'
)

recommender_example = (
    'LABORATORY TEST RECOMMENDATIONS\n'
    'Recommended Tests:\n'
    '1. [Test Name]\n'
    '   - Purpose: [Why this test is needed]\n'
    '   - Expected Results: [What we\'re looking for]\n'
    '2. [Test Name]\n'
    '   - Purpose: [Why this test is needed]\n'
    '   - Expected Results: [What we\'re looking for]'
)

recommender_criteria = textwrap.dedent(
    """\
    1. Recommend appropriate diagnostic tests
    2. Explain the purpose of each test
    3. Consider cost-effectiveness
    4. Follow current testing guidelines
    5. Note any pre-test requirements
    """
)

test_recommender = make_medical_agent(
    "Lab Test Recommender",
    recommender_persona,
    recommender_example,
    recommender_criteria,
)

# Create Lab Report Generator
report_persona = (
    'You are a laboratory report specialist who analyzes test results '
    'and generates comprehensive laboratory reports. You ensure accurate '
    'interpretation and clear communication of findings.'
)

report_example = (
    'LABORATORY REPORT\n'
    'Test Results:\n'
    '1. [Test Name]\n'
    '   - Result: [Value]\n'
    '   - Reference Range: [Range]\n'
    '   - Interpretation: [Clinical significance]\n'
    'Summary of Findings:\n'
    '[Overall interpretation and recommendations]'
)

report_criteria = textwrap.dedent(
    """\
    1. Include all test results with reference ranges
    2. Provide clear interpretation of results
    3. Flag any critical or abnormal values
    4. Suggest follow-up tests if needed
    5. Include quality control status
    """
)

report_generator = make_medical_agent(
    "Lab Report Generator",
    report_persona,
    report_example,
    report_criteria,
)

# Create Workforce
workforce = Workforce('Medical Laboratory Workflow')

workforce.add_single_agent_worker(
    'Clinical Record Summarizer: Converts doctor-patient conversations '
    'into structured clinical notes',
    worker=clinical_summarizer,
).add_single_agent_worker(
    'Clinical Analyzer: Analyzes symptoms and suggests potential '
    'conditions',
    worker=clinical_analyzer,
).add_single_agent_worker(
    'Lab Test Recommender: Recommends appropriate diagnostic tests',
    worker=test_recommender,
).add_single_agent_worker(
    'Lab Report Generator: Analyzes results and generates reports',
    worker=report_generator,
)

# Example usage
def process_clinical_case(conversation_text: str) -> str:
    """
    Process a clinical case through the medical laboratory workflow.
    
    Args:
        conversation_text (str): The doctor-patient conversation text
    
    Returns:
        str: The final laboratory report
    """
    task = Task(
        content="Process this clinical case through the laboratory workflow. "
        "First, summarize the clinical conversation. Then, analyze the "
        "symptoms and suggest potential conditions. Next, recommend "
        "appropriate laboratory tests. Finally, generate a comprehensive "
        "laboratory report based on the test results.",
        additional_info=conversation_text,
        id="0",
    )
    
    processed_task = workforce.process_task(task)
    return processed_task.result

# Example clinical conversation

example_conversation = """
Doctor: 您好，请详细说说您的症状。

Patient: 医生您好，我这情况很复杂，大概半年前开始，最初是觉得特别疲劳，以为是工作压力大。但后来出现了关节疼痛，特别是手指、手腕和膝盖，早上特别明显，活动一会儿后会好一些。

Doctor: 关节疼痛的具体表现是什么样的？是双侧还是单侧？

Patient: 是双侧的，而且是对称的。最近三个月，我还发现脸上和胸前经常出现一些红斑，像蝴蝶一样的形状，太阳光照射后会加重。有时候还会发低烧，37.5-38度之间。

Doctor: 明白了。您有没有出现口腔溃疡、脱发或者其他症状？

Patient: 确实有！经常会长溃疡，差不多两周一次。最近半年掉头发特别厉害，还总觉得眼睛干涩。最让我担心的是，有时候会觉得胸闷气短，爬楼梯都困难，之前从来没有这种情况。

Doctor: 我注意到您的手指关节有些肿胀。最近有没有出现手指发白或发紫的情况，特别是在寒冷环境下？

Patient: 对，冬天的时候特别明显，手指会先发白，然后变成紫色，最后变红，还会感觉刺痛。我父亲说我最近消瘦很多，实际上我没有刻意减肥，但是半年内瘦了将近10公斤。

Doctor: 您家族史中有类似的疾病史吗？或者其他自身免疫性疾病？

Patient: 我姑姑好像也有类似的情况，具体什么病我不太清楚。我注意到最近经常感觉心跳很快，有时候会超过100下/分钟，还经常出现夜汗。

Doctor: 您平时有服用什么药物吗？包括中药或保健品？

Patient: 之前吃过止痛药和一些维生素，但效果不明显。最近还出现了肌肉疼痛，特别是大腿和上臂，感觉浑身没劲。有时候早上起床，手指会僵硬半小时左右才能活动自如。对了，最近还经常出现头痛，有时候会头晕，视物模糊。

Doctor: 您的工作和生活习惯有什么变化吗？比如作息、压力源等。

Patient: 工作压力一直都挺大的，但最近半年确实更甚。经常失眠，睡眠质量特别差。有时候会莫名其妙地焦虑。最近还发现，一些以前经常吃的食物现在会出现过敏反应，起荨麻疹。

Doctor: 您刚才提到的胸闷气短，有没有出现过胸痛？运动后会加重吗？

Patient: 有时会隐隐作痛，但不是很剧烈。深呼吸的时候会感觉胸部不适，最近还出现了干咳的情况。有几次半夜被胸闷惊醒，同时伴有盗汗。
"""


# Process the case
result = process_clinical_case(example_conversation)
print(result)