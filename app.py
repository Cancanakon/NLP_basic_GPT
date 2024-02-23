from transformers import GPT2LMHeadModel, GPT2Tokenizer
from translate import Translator

translator_en = Translator(to_lang="en", from_lang="tr")
translator_tr = Translator(to_lang="tr", from_lang="en")

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

while True:
    user_sentence_tr = str(input("You: "))
    if user_sentence_tr.lower() == 'exit':
        break

    user_sentence_en = translator_en.translate(user_sentence_tr)

    input_ids = tokenizer.encode(user_sentence_en, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=1,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        do_sample=True,
        pad_token_id=model.config.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    bot_sentence_tr = translator_tr.translate(generated_text)

    print(f"BOT ANSWER => {bot_sentence_tr}")
