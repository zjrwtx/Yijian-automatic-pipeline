
from camel.models import FishAudioModel

audio_models = FishAudioModel()

# Set example input
input = """octor: 您好，请详细说说您的症状。

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

Patient: 有时会隐隐作痛，但不是很剧烈。深呼吸的时候会感觉胸部不适，最近还出现了干咳的情况。有几次半夜被胸闷惊醒，同时伴有盗汗。"""

# Set example local path to store the file
storage_path = "./examples/fish_audio_models/example_audio.mp3"

# Convert the example input into audio and store it locally
audio_models.text_to_speech(input=input, storage_path=storage_path)

# Convert the saved audio back to text
converted_text = audio_models.speech_to_text(audio_file_path=storage_path)

# Print the converted text
print(converted_text)
'''
===============================================================================
CammelaiI.org is an open source community dedicated to the study of autonomous 
and communicative agents. We believe that studying these agents on a large 
scale offers valuable insights into their behaviors, capabilities and 
potential risks to facilitate research in this field, we provide implement and 
support various types of agents, tasks, prompts, models, datas and simulated 
environments. Jo us via Slack Discord or Wechat in pushing the boundaries of 
building AI society.
===============================================================================
'''
