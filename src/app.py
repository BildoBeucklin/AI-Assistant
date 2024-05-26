from flask import Flask, render_template, request, jsonify
from FAQBot import FAQ_Bot
from ConvBot import conv_bot
from LlamaGenerator import llamaGenerator
from bs4 import BeautifulSoup
import torch
import spacy

convBot = conv_bot(api_base = "http://localhost:4891/v1")

print("Init\n")
print("Load LLama Generator")
llama_generator = llamaGenerator()
print(f"Topics: {convBot.get_topics()}")
faq_bot = FAQ_Bot(topic_list=convBot.get_topics())
nlp = spacy.load('de_core_news_md')

pre_processor = FAQ_Bot.pre_processor(nlp=nlp)
# Clear GPU memory
torch.cuda.empty_cache()

# Initialize context
threshold = 0.1
split_sentences = []

print("Ready\n")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    answer = "Gib Fehlermeldung, dass keine Eingabe angekommen ist."
    isHTML = 0
    info = ""
    user_input = request.form['user_input']
    prompt = llama_generator.set_up_propmt(info, user_input=user_input)
    split_sentences = []
    split_sentences, is_questions = pre_processor.get_questions(user_input)
    if is_questions:
        for sentence in split_sentences:
            pred_topics = faq_bot.get_intent(sentence, faq_bot.topics)
            topic = "".join(pred_topics['labels'][0])
            print(pred_topics)
            if pred_topics['scores'][0] > 0.2:
                info, isHTML = convBot.get_topic_content(topic=topic)
                prompt = llama_generator.set_up_propmt(info, user_input=user_input)
            elif pred_topics['scores'][0] > 0.1:
                info, isHTML = convBot.get_topic_content(topic=topic)
                prompt = llama_generator.set_up_prompt_with_context(info, user_input=user_input)   
    answer = llama_generator.generate_output(prompt)
    soup = BeautifulSoup(answer, "html.parser")
    html_formatted_text = str(soup).replace("\n", "<br>").replace("\t", "<span>&#9;</span>")
    response = html_formatted_text
    response_indicator = 'TXT'
        
    return jsonify({'response': response}, {'responseIndicator': response_indicator})

if __name__ == '__main__':
    app.run(debug=True, port=3000)

