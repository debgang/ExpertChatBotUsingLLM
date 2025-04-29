from functions_restaurant import *
from IPython.display import display, HTML
def dialogue_mgmt_system(conversations):
    conversation = initialize_conversation()

    introduction = get_chat_completions(conversation)

    display(introduction + '\n')

    recommended_restaurants = None

    user_input = ''

    while(user_input != "exit"):

        user_input = conversations["input_queue"].get(30)

        moderation = moderation_check(user_input)
        if moderation == 'Flagged':
            conversations["output_queue"].put("This message has been flagged. Please restart your conversation.")
            break

        if recommended_restaurants is None or recommended_restaurants == 'Not Found':

            conversation.append({"role": "user", "content": user_input})
            response_assistant = get_chat_completions(conversation)
            moderation = moderation_check(response_assistant)
            if moderation == 'Flagged':
                conversations["output_queue"].put("This message has been flagged. Please restart your conversation.")
                break


            confirmation = data_sufficieny_layer(response_assistant)
            if "No" in confirmation.get('result'):
                conversation.append({"role": "assistant", "content": str(response_assistant)})
                conversations["output_queue"].put(str(response_assistant) + "\n")

            else:
                response = dictionary_present(response_assistant)
                print("disctinary present:" + str(response))
                retry_count = 0;
                while recommended_restaurants is None or recommended_restaurants == 'Not Found' and retry_count < 3:
                    recommended_restaurants = compare_restaurants_with_user_request(response)
                    retry_count += 1
                validated_reco = recommendation_validation(response, recommended_restaurants)

                conversation_reco = initialize_conv_reco(validated_reco)

                conversation_reco.append({"role": "user", "content": "This is my user profile" + str(response)})

                recommendation = get_chat_completions(conversation_reco)

                moderation = moderation_check(recommendation)
                if moderation == 'Flagged':
                    conversations["output_queue"].put("Sorry, this message has been flagged. Please restart your conversation.")
                    break

                conversation_reco.append({"role": "assistant", "content": str(recommendation)})

                conversations["output_queue"].put("Thank you for providing all the information. Here are the recommendations :" + str(recommendation) + '\n')
        else:
            conversation_reco.append({"role": "user", "content": user_input})

            response_asst_reco = get_chat_completions(conversation_reco)

            moderation = moderation_check(response_asst_reco)
            if moderation == 'Flagged':
                conversations["output_queue"].put("Sorry, this message has been flagged. Please restart your conversation.")
                break

            conversations["output_queue"].put('\n' + response_asst_reco + '\n')
            conversation.append({"role": "assistant", "content": response_asst_reco})
                

              
     