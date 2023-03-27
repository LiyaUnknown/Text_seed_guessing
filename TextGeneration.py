import nltk
import re as ree
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

article = "Python Overview Python is a high level, interpreted, interactive and object oriented scripting language. Python is designed to be highly readable. Python is Interpreted − Python is processed at runtime by the interpreter. Python is Interactive − You can actually sit at a Python prompt and interact with the interpreter directly to write your programs. Python is Object Oriented − Python supports Object Oriented style or technique of programming that encapsulates code within objects. Python is a Beginner Language − Python is a great language for the beginner level programmers and supports the development of a wide range of applications from simple text processing to WWW browsers to games. History of Python Python is derived from many other languages, including ABC, Modula 3, C, C++, Algol 68, SmallTalk, and Unix shell and other scripting languages. Python is copyrighted. Python Features Python features include − Easy to learn − Python has few keywords, simple structure, and a clearly defined syntax. Easy to read − Python code is more clearly defined and visible to the eyes. Easy to maintain − Python source code is fairly easy to maintain. A broad standard library − Python bulk of the library is very portable and cross platform compatible on UNIX, Windows, and Macintosh. Interactive Mode − Python has support for an interactive mode which allows interactive testing and debugging of snippets of code. Extendable − You can add low level modules to the Python interpreter. GUI Programming − Python supports GUI applications that can be created and ported to many system calls, libraries and windows systems, such as Windows MFC, Macintosh, and the X Window system of Unix. Scalable − Python provides a better structure and support for large programs than shell scripting. Apart from the above mentioned features, Python has a big list of good features, few are listed below − "

def clean(text):
        text = ree.sub('[0-9]+.\t','',str(text))
        text = ree.sub('\n ','',str(text))
        text = ree.sub('\n',' ',str(text))
        text = ree.sub("'s",'',str(text))
        text = ree.sub("-",' ',str(text))
        text = ree.sub("— ",'',str(text))
        text = ree.sub('\"','',str(text))
        text = ree.sub("Mr\.",'Mr',str(text))
        text = ree.sub("Mrs\.",'Mrs',str(text))
        text = ree.sub("[\(\[].*?[\)\]]", "", str(text))
        text = text.replace("," , " ")
        text = text.replace("  " , " ")
        return text

def sort_by_meaning(text , list_) :
    model = SentenceTransformer("all-MiniLM-L12-v1")
    words = [text]
    for i in list_ : 
        words.append(i)
    embeddings = model.encode(words)
    encod_list = []
    for g in enumerate(words[1::]) : 
        encod_list.append(((cosine_similarity(embeddings[0].reshape(1,-1) , embeddings[g[0]+1].reshape(1,-1))[0][0] ),g[1]))
    sorted(encod_list)
    reversed(encod_list)
    return encod_list

text = clean(article)
ni = nltk.ngrams(nltk.word_tokenize(text) , 2)

text_ = ""
sen = []
for h in ni : 
    for j in h : 
        text_ += j + " "
    sen.append(text_)
    text_ = ""
g = ""

training_sentences = []
training_labels = []

for i in sen : 
    pos_tag = nltk.pos_tag(nltk.word_tokenize(i))
    for k in pos_tag[0:-1] : 
        if k[0] != k[1] :
            g += k[1] + " "
    training_labels.append(pos_tag[-1][0])
    training_sentences.append(g.strip())
    g = ""

def make_sen(mt) : 
    pat = ""
    new_sentence = mt
    for s in nltk.pos_tag(nltk.word_tokenize(mt)) : 
        pat += s[1] + " "
    pat = (sort_by_meaning(pat , training_sentences))
    jkl = ""
    print(training_labels[training_sentences.index(max(pat)[1])])
    new_sentence += " "+ training_labels[training_sentences.index(max(pat)[1])]
    for z in new_sentence.split(" ")[1::] :
        jkl += z + " "
    make_sen(jkl.strip())

make_sen("python is ")
