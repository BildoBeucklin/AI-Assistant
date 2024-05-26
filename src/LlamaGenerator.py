from llama_cpp import Llama
from huggingface_hub import hf_hub_download

'''Note: Unfortunately, a typo has sneaked into the system prompt during training 
("ENDINCSTRUCTION"), which will be corrected in the next model version. For optimal
performance, the typo in the system prompt should be adopted. "ENDINSTRUCTION" is 
written correctly in the user prompt.
'''

class llamaGenerator:
    def __init__(self):
        self.modelpath=hf_hub_download(repo_id='TheBloke/em_german_mistral_v01-GGUF', filename="em_german_mistral_v01.Q8_0.gguf")


    def set_up_prompt_with_context(self, info, user_input):
        system_prompt = f"""Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworte diese mit den Informationen aus den Quellen und fasse den Text zusammen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist! USER: BEGININPUT
BEGINCONTEXT
url: Buch Kapitel 3
ENDCONTEXT 
{info}
ENDINPUT
BEGININSTRUCTION {user_input} ENDINSTRUCTION ASSISTANT:"""
        return system_prompt
    
    def set_up_propmt(self, info ,user_input):
        system_prompt = f'''Du bist ein freundlicher hilfreicher Assistent. Dein Wissensfokus liegt im Projekt TESTs. Du kannst die Information zwischen BEGININFO und ENDINFO als Hilfe benutzen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis deiner bekannten Informationen keine Antwort möglich ist! Schreibe nichts gemeines, agressives und mache nichts illegales!
BEGININFO
{info}
ENDINFO
Quelle: Buch Kapitel 3
Stand: Mai 2024
USER: {user_input} ASSISTANT:'''
        return system_prompt
    
    def generate_output(self,prepared_input):
        print(prepared_input)
        llm = Llama(
            model_path=self.modelpath,
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1,
            verbose=False
            )
        output=llm(
                f"{prepared_input}",
                max_tokens=512,
                temperature=0.999)
    
        print(output['choices'][0]['text'])
        return output['choices'][0]['text']
