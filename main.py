from flask import Flask,render_template, request, redirect
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import warnings
warnings.filterwarnings('ignore')
# import punctuator
# from punctuator import Punctuator
from rpunct import RestorePuncts
import language_tool_python
import spacy 
# import spacy_transformers
nlp = spacy.load('en_core_web_sm') 
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import numpy as np
from transformers.modeling_bert import BertModel
from summarizer import Summarizer
from summarizer import TransformerSummarizer


app = Flask(__name__)


@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        print("Audio Received")

    if "file" not in request.files:
        return redirect(request.url)

    aud_file = request.files["file"]

    if aud_file.filename =="":
        return redirect(request.url)

    if aud_file:
        # call speech to text
        process_audio(aud_file)
        # call spelling
        audio_text =  file_read('AudioText.txt')
        spelling_correct(audio_text)
        # call punctuation
        spell_text = file_read('SpellCorrect.txt')
        punct(spell_text)
        # call summarization
        punct_text = file_read('punctcorrect.txt')
        summary = summarizer(text=punct_text, tokenizer=nlp, max_sent_in_summary=5)
    return render_template('index.html', output = summary)

def process_audio(aud_file):
    txtf = open('AudioText.txt','w+')
    myaudio = AudioSegment.from_wav(aud_file)
    chunks_length_ms = 10000
    chunks =make_chunks(myaudio,chunks_length_ms)
    for i,chunk in enumerate(chunks):
        chunkName = aud_file+'_{0}_wav'.format(i)
        print('I am exporting',chunkName)
        chunk.export(chunkName, format = 'wav')
        file = chunkName
        r = sr.Recognizer()
        with sr.AudioFile(file) as source:
            audio_listened = r.listen(source)
            try:
                rec = r.recognize_google(audio_listened)
                print(rec)
                txtf.writelines(rec)
            except sr.RequestError as e:
                print('Check your Intrenet')     
try:
  os.makedirs('Chunked')
except:
  pass

def spelling_correct(my_text): 
  # using the tool  
    my_tool = language_tool_python.LanguageTool('en-In')   
    my_matches = my_tool.check(my_text)  
    
    # defining some variables  
    myMistakes = []  
    myCorrections = []  
    startPositions = []  
    endPositions = []  
    
    # using the for-loop  
    for rules in my_matches:  
        if len(rules.replacements) > 0:  
            startPositions.append(rules.offset)  
            endPositions.append(rules.errorLength + rules.offset)  
            myMistakes.append(my_text[rules.offset : rules.errorLength + rules.offset])  
            myCorrections.append(rules.replacements[0])  
    
    # creating new object  
    my_NewText = list(my_text)   
        
    # rewriting the correct passage  
    for n in range(len(startPositions)):  
        for i in range(len(my_text)):  
            my_NewText[startPositions[n]] = myCorrections[n]  
            if (i > startPositions[n] and i < endPositions[n]):  
                my_NewText[i] = ""  
        
    my_NewText = "".join(my_NewText)  
    spf = open('SpellCorrect.txt','w+')  
    spf.write(my_NewText)

def file_read(file_name):  
    text_file = open('file_name.txt')
    my_text = text_file.read()
    return my_text

def punct(punct_text):
  rpunct = RestorePuncts()
  rp = rpunct.punctuate(punct_text)
  punct = open('punctcorrect.txt','w+')  
  punct.write(rp)
  
def summarizer(text, tokenizer, max_sent_in_summary=3):
    # Create spacy document for further sentence level tokenization
    doc = nlp(text.replace("\n", ""))
    sentences = [sent.text.strip() for sent in doc.sents]
    # Let's create an organizer which will store the sentence ordering to later reorganize the 
    # scored sentences in their correct order
    sentence_organizer = {k:v for v,k in enumerate(sentences)}
    # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
    tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                        strip_accents='unicode', 
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        ngram_range=(1, 3), 
                                        use_idf=1,smooth_idf=1,
                                        sublinear_tf=1,
                                        stop_words = 'english')
    # Passing our sentences treating each as one document to TF-IDF vectorizer
    tf_idf_vectorizer.fit(sentences)
    # Transforming our sentences to TF-IDF vectors
    sentence_vectors = tf_idf_vectorizer.transform(sentences)
    # Getting sentence scores for each sentences
    sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    # Getting top-n sentences
    N = max_sent_in_summary
    top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
    # Let's now do the sentence ordering using our prebaked sentence_organizer
    # Let's map the scored sentences with their indexes
    mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
    # Ordering our top-n sentences in their original ordering
    mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
    ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
    # Our final summary
    summary = " ".join(ordered_scored_sentences)
    return summary

def bert_summary(text):
    bert_model = Summarizer()
    bert_summary = ''.join(bert_model(text, min_length=60))
    return bert_summary

def gp2(text):
    GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    gp2_text = ''.join(GPT2_model(text, min_length=60))
    print(gp2_text)
    
# Returning the app
if __name__ == '__main__':
    app.run(debug = True)
