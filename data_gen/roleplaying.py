from colorama import Fore

from camel.societies import RolePlaying
from camel.utils import print_text_animated

def patient_note():
    patient_note = input("请输入患者的病历：")
    return patient_note


patient_note=patient_note()
def main(model=None, chat_turn_limit=50) -> None:
    task_prompt = "根据给出病例反向模拟一个医生和一个患者的对话，医生问问题，患者回答问题。模拟的对话要尽可能的详细，包括医生的问诊过程和患者的回答过程。以下是病例内容:"+patient_note
    role_play_session = RolePlaying(
        assistant_role_name="医生",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="患者",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
    )

    print(
        Fore.GREEN
        + f"AI Assistant sys message:\n{role_play_session.assistant_sys_msg}\n"
    )
    print(
        Fore.BLUE + f"AI User sys message:\n{role_play_session.user_sys_msg}\n"
    )

    print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")
    print(
        Fore.CYAN
        + "Specified task prompt:"
        + f"\n{role_play_session.specified_task_prompt}\n"
    )
    print(Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n")

    n = 0
    input_msg = role_play_session.init_chat()
    assistant_history = ""
    
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        if assistant_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI Assistant terminated. Reason: "
                    f"{assistant_response.info['termination_reasons']}."
                )
            )
            break
        if user_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI User terminated. "
                    f"Reason: {user_response.info['termination_reasons']}."
                )
            )
            break

        assistant_history += assistant_response.msg.content + "\n"

        print_text_animated(
            Fore.BLUE + f"AI User:\n\n{user_response.msg.content}\n"
        )
        print_text_animated(
            Fore.GREEN + "AI Assistant:\n\n"
            f"{assistant_response.msg.content}\n"
        )

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break

        input_msg = assistant_response.msg

    print("\n完整的Assistant对话历史：")
    print(assistant_history)


if __name__ == "__main__":
    main()
