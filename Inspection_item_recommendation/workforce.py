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
from camel.configs import DeepSeekConfig,ChatGPTConfig
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
    model_platform=ModelPlatformType.OPENAI,
    model_type="gpt-4o",
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
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type="gpt-4o",
    # Ensure using ollama verison >= 0.5.1 to use structured output feature
    model_config_dict={"temperature": 0.4,"max_tokens":4090},
)
# Create Workforce

workforce = Workforce(
    'Medical Laboratory Workflow',
    coordinator_agent_kwargs = {"model": model},
    task_agent_kwargs = {"model": model},
)
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
)
# .add_single_agent_worker(
#     'Lab Report Generator: Analyzes results and generates reports',
#     worker=report_generator,
# )

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
        "appropriate laboratory tests.  ",
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


# example output
"""
Worker node 1601014880656 (Clinical Record Summarizer: Converts doctor-patient conversations into structured clinical notes) get task 0.0: Summarize the clinical conversation from the doctor-patient dialogue. [Assigned to: Clinical Record Summarizer]
======
Reply from Worker node 1601014880656 (Clinical Record Summarizer: Converts doctor-patient conversations into structured clinical notes):

CLINICAL SUMMARY

Patient Information:
- Name: [Patient Name]
- Date: [Visit Date]

Chief Complaint:
- Fatigue, joint pain, facial rash, fever, and other systemic symptoms.

History of Present Illness:
- The patient reports a complex set of symptoms beginning approximately six months ago. Initially, the patient experienced significant fatigue, attributed to work stress. Subsequently, the patient developed joint pain, particularly in the fingers, wrists, and knees, which is most pronounced in the morning but improves with activity.
- Over the past three months, the patient has noticed a butterfly-shaped rash on the face and chest, exacerbated by sun exposure, along with low-grade fevers ranging from 37.5 to 38 degrees Celsius.

Current Symptoms:
- Symmetrical joint pain and swelling.
- Facial and chest rash.
- Recurrent oral ulcers, approximately bi-weekly.
- Significant hair loss over the past six months.
- Dry eyes.
- Episodes of chest tightness and shortness of breath, particularly when climbing stairs.
- Raynaud's phenomenon: fingers turning white, then purple, then red, with tingling sensation in cold environments.
- Unintentional weight loss of approximately 10 kilograms over six months.
- Palpitations with heart rate exceeding 100 bpm, night sweats.
- Muscle pain in the thighs and upper arms, morning stiffness in fingers lasting about 30 minutes.
- Headaches, dizziness, and blurred vision.
- Insomnia and anxiety.
- New onset food allergies causing urticaria.
- Occasional chest pain, not severe, with discomfort on deep breathing and dry cough.

Relevant Medical History:
- Family history: Aunt with similar symptoms, details unknown.
- No specific autoimmune diseases noted.

Medications:
- Previously used painkillers and vitamins with minimal effect.

Allergies:
- Recent development of food allergies causing urticaria.

Social History:
- Increased work-related stress and poor sleep quality over the past six months.

The clinical presentation suggests a possible autoimmune condition, potentially systemic lupus erythematosus (SLE), given the constellation of symptoms including joint pain, rash, oral ulcers, and systemic involvement.
======Worker node 1601013938832 (Clinical Analyzer: Analyzes symptoms and suggests potential conditions) get task 0.1: Analyze the summarized symptoms and suggest potential conditions. [Assigned to: Clinical Analyzer]
======
Reply from Worker node 1601013938832 (Clinical Analyzer: Analyzes symptoms and suggests potential conditions):

CLINICAL ANALYSIS

Primary Symptoms:
- Fatigue - Persistent over six months
- Joint pain and swelling - Symmetrical, particularly in fingers, wrists, and knees, worse in the morning
- Facial and chest rash - Butterfly-shaped, worsens with sun exposure
- Recurrent oral ulcers - Bi-weekly
- Significant hair loss - Over six months
- Dry eyes
- Chest tightness and shortness of breath - Especially with exertion
- Raynaud's phenomenon - Fingers change color in cold
- Unintentional weight loss - 10 kg over six months
- Palpitations and night sweats
- Muscle pain - Thighs and upper arms
- Morning stiffness - Lasting about 30 minutes
- Headaches, dizziness, blurred vision
- Insomnia and anxiety
- New onset food allergies - Urticaria
- Occasional chest pain and dry cough

Potential Conditions:
1. Systemic Lupus Erythematosus (SLE) - Primary consideration due to the presence of joint pain, butterfly rash, oral ulcers, systemic symptoms like fatigue, fever, and Raynaud's phenomenon, which are characteristic of SLE.
2. Sjögren's Syndrome - Secondary consideration due to symptoms of dry eyes and mouth, which could overlap with SLE.
3. Mixed Connective Tissue Disease (MCTD) - Considered due to the combination of symptoms like Raynaud's phenomenon, joint pain, and muscle pain.
4. Rheumatoid Arthritis - Considered due to symmetrical joint pain and morning stiffness, although the presence of a rash and systemic symptoms makes SLE more likely.

Risk Factors Present:
- Family history of similar symptoms (possible autoimmune condition)
- Increased work-related stress

Red Flags or Urgent Concerns:
- Significant unintentional weight loss
- Palpitations and chest tightness
- Episodes of shortness of breath

Differential Diagnoses:
- Consideration of other autoimmune diseases such as dermatomyositis or polymyositis due to muscle pain and systemic symptoms.
- Possible overlap syndrome given the range of symptoms affecting multiple systems.

The clinical presentation strongly suggests an autoimmune etiology, with systemic lupus erythematosus being the most likely diagnosis. Further diagnostic workup, including serological tests (e.g., ANA, anti-dsDNA antibodies), is recommended to confirm the diagnosis and assess disease activity.
======Worker node 1601014882000 (Lab Test Recommender: Recommends appropriate diagnostic tests) get task 0.2: Recommend appropriate laboratory tests based on the analyzed symptoms and potential conditions. [Assigned to: Lab Test Recommender]       
======
Reply from Worker node 1601014882000 (Lab Test Recommender: Recommends appropriate diagnostic tests):

LABORATORY TEST RECOMMENDATIONS

Recommended Tests:
1. **Antinuclear Antibody (ANA) Test**
   - **Purpose:** To screen for autoimmune disorders, particularly systemic lupus erythematosus (SLE), given the presence of a butterfly rash, joint pain, and other systemic symptoms.   
   - **Expected Results:** A positive ANA test would support the suspicion of an autoimmune condition like SLE.

2. **Anti-double-stranded DNA (anti-dsDNA) Antibody Test**    
   - **Purpose:** To specifically diagnose SLE, as anti-dsDNAantibodies are highly specific for this condition.
   - **Expected Results:** Elevated levels would strongly indicate SLE.

3. **Anti-Smith (anti-Sm) Antibody Test**
   - **Purpose:** To further confirm SLE diagnosis, as anti-Sm antibodies are also specific to SLE.
   - **Expected Results:** Presence of anti-Sm antibodies would confirm SLE diagnosis.

4. **Complete Blood Count (CBC)**
   - **Purpose:** To assess for anemia, leukopenia, or thrombocytopenia, which are common in SLE and other autoimmune diseases.
   - **Expected Results:** Abnormal counts could indicate systemic involvement typical of SLE.

5. **Erythrocyte Sedimentation Rate (ESR) and C-Reactive Protein (CRP)**
   - **Purpose:** To evaluate the presence of inflammation and disease activity.
   - **Expected Results:** Elevated levels would indicate active inflammation, supporting an autoimmune process.

6. **Rheumatoid Factor (RF) and Anti-Cyclic Citrullinated Peptide (anti-CCP) Antibody Test**
   - **Purpose:** To rule out rheumatoid arthritis, given the joint symptoms.
   - **Expected Results:** Positive results would suggest rheumatoid arthritis, although SLE is more likely given the full symptom profile.

7. **Anti-Ro/SSA and Anti-La/SSB Antibody Tests**
   - **Purpose:** To assess for Sjögren's syndrome, which can overlap with SLE and cause dry eyes and mouth.
   - **Expected Results:** Positive results could indicate Sjögren's syndrome or overlap syndrome.

8. **Urinalysis**
   - **Purpose:** To check for kidney involvement, which is common in SLE.
   - **Expected Results:** Presence of proteinuria or hematuria would suggest renal involvement.

9. **Complement Levels (C3, C4)**
   - **Purpose:** To evaluate complement consumption, which is common in active SLE.
   - **Expected Results:** Low levels would indicate active disease.

**Pre-test Requirements:**
- Fasting is not required for these tests, but patients should be informed about the possibility of multiple blood draws.
- It is advisable to avoid NSAIDs or other medications that might interfere with test results prior to testing, if clinically feasible.

These tests are recommended based on the clinical suspicion of systemic lupus erythematosus and other potential autoimmune conditions. They are cost-effective and aligned with current diagnostic guidelines for suspected autoimmune diseases.
======The final answer to the root task, which involves processing the clinical case through the laboratory workflow, is as follows:

1. **Clinical Conversation Summary:**
   - The patient has been experiencing a complex set of symptoms over the past six months, including fatigue, symmetrical joint pain and swelling, a butterfly-shaped rash on the face and chest, recurrent oral ulcers, significant hair loss, dry eyes, chest tightness, Raynaud's phenomenon, unintentional weight loss, palpitations, night sweats, muscle pain, morning stiffness, headaches, dizziness, blurred vision, insomnia, anxiety, new onset food allergies, occasional chest pain, and dry cough. The patient's family history includes an aunt with similar symptoms, suggesting a possible autoimmune condition.

2. **Symptom Analysis and Potential Conditions:**
   - The primary symptoms and their analysis suggest the likelihood of an autoimmune condition, with systemic lupus erythematosus (SLE) being the primary consideration due to the presence of joint pain, butterfly rash, oral ulcers, and systemic symptoms. Other potential conditions include Sjögren's Syndrome, Mixed Connective Tissue Disease (MCTD), and Rheumatoid Arthritis. The risk factors include a family history of similar symptoms and increased work-related stress.

3. **Laboratory Test Recommendations:**
   - To confirm the suspected diagnosis of SLE and rule out other conditions, the following laboratory tests are recommended:
     - Antinuclear Antibody (ANA) Test
     - Anti-double-stranded DNA (anti-dsDNA) Antibody Test    
     - Anti-Smith (anti-Sm) Antibody Test
     - Complete Blood Count (CBC)
     - Erythrocyte Sedimentation Rate (ESR) and C-Reactive Protein (CRP)
     - Rheumatoid Factor (RF) and Anti-Cyclic Citrullinated Peptide (anti-CCP) Antibody Test
     - Antinuclear Antibody (ANA) Test
     - Anti-double-stranded DNA (anti-dsDNA) Antibody Test    
     - Anti-Smith (anti-Sm) Antibody Test
     - Complete Blood Count (CBC)
     - Erythrocyte Sedimentation Rate (ESR) and C-Reactive Protein (CRP)
     - Rheumatoid Factor (RF) and Anti-Cyclic Citrullinated Peptide (anti-CCP) Antibody Test
     - Anti-double-stranded DNA (anti-dsDNA) Antibody Test    
     - Anti-Smith (anti-Sm) Antibody Test
     - Complete Blood Count (CBC)
     - Erythrocyte Sedimentation Rate (ESR) and C-Reactive Protein (CRP)
     - Rheumatoid Factor (RF) and Anti-Cyclic Citrullinated Peptide (anti-CCP) Antibody Test
     - Anti-Smith (anti-Sm) Antibody Test
     - Complete Blood Count (CBC)
     - Erythrocyte Sedimentation Rate (ESR) and C-Reactive Protein (CRP)
     - Rheumatoid Factor (RF) and Anti-Cyclic Citrullinated Peptide (anti-CCP) Antibody Test
     - Complete Blood Count (CBC)
     - Erythrocyte Sedimentation Rate (ESR) and C-Reactive Protein (CRP)
     - Rheumatoid Factor (RF) and Anti-Cyclic Citrullinated Peptide (anti-CCP) Antibody Test
tein (CRP)
     - Rheumatoid Factor (RF) and Anti-Cyclic Citrullinated Peptide (anti-CCP) Antibody Test
     - Anti-Ro/SSA and Anti-La/SSB Antibody Tests
     - Urinalysis
     - Rheumatoid Factor (RF) and Anti-Cyclic Citrullinated Peptide (anti-CCP) Antibody Test
     - Anti-Ro/SSA and Anti-La/SSB Antibody Tests
     - Urinalysis
     - Anti-Ro/SSA and Anti-La/SSB Antibody Tests
     - Urinalysis
     - Complement Levels (C3, C4)

These steps provide a comprehensive approach to diagnosing and managing the patient's condition, focusing on confirming the presence of an autoimmune disorder, particularly systemic lupus erythematosus.

"""