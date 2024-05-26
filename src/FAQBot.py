from transformers import pipeline
import re


class FAQ_Bot:

    def __init__(self, topic_list):
        # Set NLP Model and build pipeline
        nlp_model_name = "deepset/gelectra-large-germanquad"
        self.q_n_a = pipeline('question-answering', model=nlp_model_name, tokenizer=nlp_model_name,)

        # Load the zero-shot classification pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.topics = topic_list

    # Define a function to recognize the intent of a given text
    def get_intent(self, text, intents):
        # Get the model's predictions
        intent = self.classifier(text, intents, multi_label=False)
        # Return the recognized intent
        return intent

    def make_html_list(self, text):
        # Split the text into lines
        lines = text.split('\n')
        # Process each line and create the HTML list
        html_list = "<ul class='list'>\n"
        for line in lines:
            # Add the line as an HTML list item
            html_list += f"  <li>\n    {line}\n  </li>\n"

        html_list += "</ul>"
        return html_list

    def find_sentence_boundaries(self, text, word_start, word_end):
        # Define sentence-ending punctuation marks.
        sentence_endings = r'[<>]'
        
        # Find the beginning of the sentence.
        sentence_start = word_start
        while sentence_start > 0 and not re.search(sentence_endings, text[sentence_start - 1]):
            sentence_start -= 1
        
        # Find the end of the sentence.
        sentence_end = word_end
        while sentence_end < len(text) and not re.search(sentence_endings, text[sentence_end]):
            sentence_end += 1
        
        # Extract the sentence.
        sentence = text[sentence_start:sentence_end].strip()
        
        return sentence

    def get_response_with_context(self, QA_input, threshold):
        question = QA_input['question']
        context = QA_input['context']

        res = self.q_n_a(question=question, context=context)
        if res['score'] > threshold:
            answer = res['answer']
        else:
            answer = "None"
        print(res)
        if res['score']>0.03:
            whole_sentence = self.find_sentence_boundaries(context, res['start'], res['end'])
            print(whole_sentence)
            answer = self.make_html_list(whole_sentence)
            print(answer) 
        else:
            answer = whole_sentence = "Darauf konnte ich leider keine Antwort finden, versuche es mit anderen Worten nochmal."
        
        return answer, whole_sentence    
    

    class pre_processor:
        def __init__(self, nlp):
            self.w_words = ["wer", "welcher", "welche", "welches", "welchen", "was", "wessen", "wann", "worum", "wo", "wohin", "wie", "wieso", "weshalb", "warum"]
            self.nlp = nlp
        
        def parse_sentences(self, prompt):
            # Process the prompt using spaCy
            doc = self.nlp(prompt)

            # Extract individual sentences
            sentences = [sent.text for sent in doc.sents]

            return sentences
        
        def get_questions(self, prompt):
            is_min_one_question = False
            questions = []
            sentences = self.parse_sentences(prompt=prompt)
            for s in sentences:
                print(s + ": ")
                if self.is_questions(sentence=s):
                    print ("yes")
                    is_min_one_question = True
                    questions.append(s)
            
            return questions, is_min_one_question


        def is_questions(self, sentence):
            # Process the sentence using SpaCy
            doc = self.nlp(sentence)
            if sentence.strip().endswith("?"):
                return True
            if any(token.pos_ in ["PRON", "ADV"] and token.text.lower() in self.w_words for token in doc):
                return True
            for token in doc:
                #print(token. text, token.dep_, token.tag_, token.pos_, token.lemma_, token.head, [child.text for child in token.children])
                # Check if the sentence exhibits subject-verb inversion
                if token.dep_ == "ROOT" and token.tag_ == "VAFIN" and token.head.dep_ == "nsubj":
                    #print("yes")
                    return True
                # Check for syntactic structures indicative of questions
                if token.dep_ != "cm" and token.tag_ == "KOKOM":
                    #print("yes")
                    return True
                if token in token.head.children and token.text.lower() in self.w_words and token.head.pos_ == "VERB":
                    #print("yes")
                    return True      
            return False


