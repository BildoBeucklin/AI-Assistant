from FAQBot import FAQ_Bot
import openai
import spacy
import torch
import sqlite3



openai.api_key = "not needed for a local LLM"


'''Note: Unfortunately, a typo has sneaked into the system prompt during training 
("ENDINCSTRUCTION"), which will be corrected in the next model version. For optimal
performance, the typo in the system prompt should be adopted. "ENDINSTRUCTION" is 
written correctly in the user prompt.
'''

class conv_bot:
    def __init__(self, api_base):
        openai.api_base = api_base
        openai.api_key = "not needed for a local LLM"
    
    def set_up_prompt_with_context(self, info, user_input):
        system_prompt = f"""Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworte diese wortwörtlich mit einem Zitat aus den Quellen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist! USER: BEGININPUT
        BEGINCONTEXT
        Url: https://www.plup-die-elefantin.one
        {info}
        ENDINPUT
        BEGININSTRUCTION {user_input} ENDINSTRUCTION ASSISTANT:"""
        return system_prompt

    def generate_with_qpt4all_qpi(self, prompt):
        model = "em_german_mistral_v01.Q4_0"
        print(prompt)
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=200,
            temperature=0.28,
            top_p=0.95,
            n=1,
            echo=True,
            stream=False
        )
        
        return response


    def get_topics(self):
        # Connect to SQLite database
        conn = sqlite3.connect('q_n_a.db')
        cursor = conn.cursor()

        # Execute a SELECT query to retrieve text data
        cursor.execute(f'''SELECT topic FROM topics''')

        # Fetch all the results
        topics = cursor.fetchall()

        # Close the connection
        conn.close()

        return topics


    def get_topic_content(self, topic):
        # Connect to SQLite database
        conn = sqlite3.connect('q_n_a.db')
        cursor = conn.cursor()

        # Execute a SELECT query to retrieve text data
        cursor.execute(f'''SELECT content, isHTML FROM topics where topic = "{topic}"''')

        # Fetch all the results
        content = cursor.fetchone()

        # Close the connection
        conn.close()
        if content:
            # Ergebnis verarbeiten
            text, isHTML = content
            print(f'isHTML: {isHTML}')
        else:
            print('Kein Ergebnis gefunden.')
        #text = ''.join(text)
        return text.replace('"', ""), isHTML

    def generate_answer_with_context(self, topic, user_input):
        text = self.get_topic_content(topic)
        prompt = self.set_up_prompt_with_context(text, user_input=user_input)
        answer = self.generate_with_qpt4all_qpi(prompt=prompt)
        answer = answer["choices"][0]["text"]
        answer = answer.strip()
        return answer

    def genrate_answer(self, user_input):
        answer = self.generate_with_qpt4all_qpi(user_input)
        answer = answer["choices"][0]["text"]
        answer = answer.strip() 
        return answer
