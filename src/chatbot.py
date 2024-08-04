import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(user_input, chat_history_ids):
    # Encode the user input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Create the attention mask
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8,
        attention_mask=attention_mask  # Add the attention mask here
    )

    # Decode and return the model's response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

def main():
    print("Chatbot: Hello! I'm a simple chatbot. Type 'quit' to exit.")
    chat_history_ids = None

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break

        response, chat_history_ids = generate_response(user_input, chat_history_ids)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()