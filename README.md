# Audio Text Summarization
## Introduction:
People in today's hectic environment don't have time to listen to long audio files or lectures. They are staying away from any and all lengthy audio sources. On the other hand, many individuals would be interested if the same audio was presented in a summarized version. In that situation, the Audio to Text Summarizer project will be useful.
## Task:
To build a model that gives the accurate summary for the audio file given.
## Approach:
1.	Get the audio file as input. Audio can be of any size. 
2.	Divide the audio files into chunks using pydub library. Each chunk will be 10000 ms in length.
3.	Using the SpeechRecognizer library, convert each audio segment to text.
4.	Performed spelling check using the language_tool_python library. 
	- Finds the error word. 
	- Replaces the incorrect word with the right one.
5.	Utilizing the rpunct library, punctuate the entire document.
6.	Summarized the punctuated text with tf_idf vectorizer, BERT, GPT2
7.	The BERT summarizer outperformed and it’s chosen for deployment.
8.	Create a HTML and CSS file to get the audio file input and display the output after the audio file has been summarized.
9.	Deploy the project in flask framework.
