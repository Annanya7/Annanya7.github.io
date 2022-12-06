---
layout: post
title: Deep Short Text Classification using Hybrid Approach
featured: true 
---
## Problem Statement 
Online social platforms (OSPs) contain whole lot of information which is broadcasted all over the world. Unaware of repercussions these platforms have least control over the type of content they project which gives a wrong message to the society. Political parties use these platforms to create a biased audience for themselves which is not a fair play. A general picture of the discourses that each political party disseminates through Twitter during the electoral campaign and their underlying tensions and sentiment. While polls are useful to quickly measure the perception of the citizenship about these and other topics, there is an obvious limitation: it is based on the subjective perception of citizens and, therefore, is not useful to objectively measure the real tension underlying the political context. 

## Code 
``` python
# Importing the libraries 
!pip install tweet-preprocessor
!pip install transformers
!pip install opendatasets
!pip install tqdm
!pip3 install googletrans==3.1.0a0
# Reading the csv file 
import pandas as pd
df = pd. read_csv('/content/tweets100.csv')
df.head()
```
Output
```python

cuenta	partido	timestamp	tweet
0	a34133350b0605cb24081843f63176ca	psoe	1363973492	@vesteve3 @manubenas @ccoo_rm @desobediencia_ ...
1	a34133350b0605cb24081843f63176ca	psoe	1364061035	“@kirovast: @Hugo_Moran muy fan de la "radical...
2	a34133350b0605cb24081843f63176ca	psoe	1364116804	@ALTAS_PRESIONES Nuevos dueños para las renova...
3	a34133350b0605cb24081843f63176ca	psoe	1364120967	@jumanjisolar @solartradex @josea_dolera El di...
4	a34133350b0605cb24081843f63176ca	psoe	1364152692	“@cesarnayu: https://t.co/J4OTXj1x7w … Por fav...
```
Translating Spanish tweets into English
```python
from googletrans import Translator
import pandas as pd
translator = Translator()
tweets = df['tweet'].tolist()
df['English'] = df['tweet'].apply(translator.translate,src='es',dest='en').apply(getattr,args=('text',))
df
```
Output
```python
cuenta	partido	timestamp	tweet	English
0	a34133350b0605cb24081843f63176ca	psoe	1363973492	@vesteve3 @manubenas @ccoo_rm @desobediencia_ ...	@vesteve3 @manubenas @ccoo_rm @desobediencia_ ...
1	a34133350b0605cb24081843f63176ca	psoe	1364061035	“@kirovast: @Hugo_Moran muy fan de la "radical...	“@kirovast: @Hugo_Moran a big fan of "social r...
2	a34133350b0605cb24081843f63176ca	psoe	1364116804	@ALTAS_PRESIONES Nuevos dueños para las renova...	@ALTAS_PRESIONES New owners for renewables. At...
3	a34133350b0605cb24081843f63176ca	psoe	1364120967	@jumanjisolar @solartradex @josea_dolera El di...	@jumanjisolar @solartradex @josea_dolera The e...
4	a34133350b0605cb24081843f63176ca	psoe	1364152692	“@cesarnayu: https://t.co/J4OTXj1x7w … Por fav...	“@cesarnayu: https://t.co/J4OTXj1x7w … Please,...
...	...	...	...	...	...
94	8c926e81af138f29b2b39b83d7551e0b	ciudadanos	1504816373	#NuevaFotoDePerfil https://t.co/onZQVOxzvq	#NewProfilePhoto https://t.co/onZQVOxzvq
95	b2c8bc25ba1bc91cee7acfcc0f690444	psoe	1506155538	Todos los estudios lo corroboran.Andalucia ent...	All the studies corroborate it. Andalusia amon...
96	b2c8bc25ba1bc91cee7acfcc0f690444	psoe	1507097500	Ganar a los independentistas; por Alfredo Pére...	Win the independentistas; by Alfredo Pérez Rub...
97	b2c8bc25ba1bc91cee7acfcc0f690444	psoe	1507279702	Hernando (PP) en vez de venir a Andalucía con ...	Hernando (PP) instead of coming to Andalusia w...
98	b2c8bc25ba1bc91cee7acfcc0f690444	psoe	1507279869	La culpa de que el PP nacional tenga una image...	The blame for the fact that the national PP ha...
```
### Data Preprocessing
Data preprocessing can refer to manipulation or dropping of data before it is used in order to ensure or enhance performance.We clean the tweets by removing the urls, @ etc.We remove the stop words and also convert the words into their ropt form.

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
import re
def clean_reviews(text):
  text = re.sub(r'@[A-Za-z0-9]+','',text)
  text = re.sub(r'#','',text)
  text = re.sub(r'RT[\s]+','',text)
  text = re.sub(r'https?:\/\/\S+','',text)
  text = re.sub(r'[^\w\s]','',text)
  return text
df['English'] = df['English'].apply(clean_reviews)
df.head()
```
Output
```python

cuenta	partido	timestamp	tweet	English
0	a34133350b0605cb24081843f63176ca	psoe	1363973492	@vesteve3 @manubenas @ccoo_rm @desobediencia_ ...	_rm _ good cheer for this spring that we a...
1	a34133350b0605cb24081843f63176ca	psoe	1364061035	“@kirovast: @Hugo_Moran muy fan de la "radical...	_moran a big fan of social radicalism in the ...
2	a34133350b0605cb24081843f63176ca	psoe	1364116804	@ALTAS_PRESIONES Nuevos dueños para las renova...	_presiones new owners for renewables at that t...
3	a34133350b0605cb24081843f63176ca	psoe	1364120967	@jumanjisolar @solartradex @josea_dolera El di...	_dolera the energy price differential with g...
4	a34133350b0605cb24081843f63176ca	psoe	1364152692	“@cesarnayu: https://t.co/J4OTXj1x7w … Por fav...	please it is important to spread this messa...
```
Stop word removal , Stemming & Lemmatization
```python
df['English'] = df['English'].apply(lambda x: x.split())
import spacy
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words
df['English'] = df['English'].apply(lambda sentence: [word for word in sentence if word not in stopwords])
df['English'].head()
stemmer = PorterStemmer()
df['English'] = df['English'].apply(lambda sentence: [stemmer.stem(word) for word in sentence])
df['English'].head()
from nltk.stem import WordNetLemmatizer
def lematize(review):
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review]
    review = ' '.join(review)
    return review
df['English'] = df['English'].apply(lematize)
df
```
Output
```python
cuenta	partido	timestamp	tweet	English
0	a34133350b0605cb24081843f63176ca	psoe	1363973492	@vesteve3 @manubenas @ccoo_rm @desobediencia_ ...	_rm _ good cheer spring start
1	a34133350b0605cb24081843f63176ca	psoe	1364061035	“@kirovast: @Hugo_Moran muy fan de la "radical...	_moran big fan social radic face democrat devalu
2	a34133350b0605cb24081843f63176ca	psoe	1364116804	@ALTAS_PRESIONES Nuevos dueños para las renova...	_presion new owner renew time longer problem m...
3	a34133350b0605cb24081843f63176ca	psoe	1364120967	@jumanjisolar @solartradex @josea_dolera El di...	_dolera energi price differenti germani franc ...
4	a34133350b0605cb24081843f63176ca	psoe	1364152692	“@cesarnayu: https://t.co/J4OTXj1x7w … Por fav...	import spread messag rais awar retweet
...	...	...	...	...	...
94	8c926e81af138f29b2b39b83d7551e0b	ciudadanos	1504816373	#NuevaFotoDePerfil https://t.co/onZQVOxzvq	newprofilephoto
95	b2c8bc25ba1bc91cee7acfcc0f690444	psoe	1506155538	Todos los estudios lo corroboran.Andalucia ent...	studi corrobor andalusia worst financ look way
96	b2c8bc25ba1bc91cee7acfcc0f690444	psoe	1507097500	Ganar a los independentistas; por Alfredo Pére...	win independentista alfredo pérez rubalcaba _r...
97	b2c8bc25ba1bc91cee7acfcc0f690444	psoe	1507279702	Hernando (PP) en vez de venir a Andalucía con ...	hernando pp instead come andalusia employ plan...
98	b2c8bc25ba1bc91cee7acfcc0f690444	psoe	1507279869	La culpa de que el PP nacional tenga una image...	blame fact nation pp imag dirt barrack andalus...
```
Remove the extra columns
```python
df= df.drop('cuenta',axis = 1)
df = df.drop('timestamp',axis = 1)
df = df.drop('tweet',axis=1)
df
```
Output

```python
partido	English
0	psoe	_rm _ good cheer spring start
1	psoe	_moran big fan social radic face democrat devalu
2	psoe	_presion new owner renew time longer problem m...
3	psoe	_dolera energi price differenti germani franc ...
4	psoe	import spread messag rais awar retweet
...	...	...
94	ciudadanos	newprofilephoto
95	psoe	studi corrobor andalusia worst financ look way
96	psoe	win independentista alfredo pérez rubalcaba _r...
97	psoe	hernando pp instead come andalusia employ plan...
98	psoe	blame fact nation pp imag dirt barrack andalus...
```
```python
tweets_corpus = []
for i in range(len(df['English'])):
  r = df.iloc[i,-1]
  r = "".join(r)
  tweets_corpus.append(r)
```
Output
```python
['_rm _ good cheer spring start',
 '_moran big fan social radic face democrat devalu',
 '_presion new owner renew time longer problem magnific bet',
 '_dolera energi price differenti germani franc remain',
 'import spread messag rais awar retweet',
 'govern continu hydrolog privat plan',
 'cosped crise longer affect spain guindo crisi cypru resolv infect spain',
 'spain import energi portug cheaper 1 2 month franc',
 'andalusian govern file appeal unconstitution law 152012 rrdd 292012 22013',
 'maria dolor de cosped sue greenpeac violat honor simul defer lawsuit',
 'step compli wfd',
 'yesterday tc publish sentenc forest law 2003 cañet want rescu past time',
 '_moran countri interest brick come',
 'obviou erron analysi scope measur consequ',
 'ibex yield 1',
 'rajoy deceiv eu econom deficit environment climat deficit',
 'rajoy tri cover bárcena intern agenda _e taparfle',
 '_informacion 19yearold student invent system clean ocean plastic',
 'legisl vehicl emiss climat chang pp mayan propheci',
 'read collaps energi consumpt bank spain 2013 wors govern 2013 better',
 'usa',
 '2030 intermedi station eu 2050 energi roadmap',
 'case sulfur nitrogen oxid emiss fuel consumpt',
 'like placid saturday afternoon reflect',
 'person tribut lui martínez noval good socialist endear man',
 'propos govern stabl secur roadmap renew',
 '_rubalcaba_ tri answer ask',
 'frack',
 'close door frack demagogueri open let accept precautionari principl',
 'like psoe propos moratorium expert report decis parliament',
 '',
 '_rubalcaba_ week psoe sit renew sector',
 '_rubalcaba_ initi psoe parliament come dialogu renew',
 '_rubalcaba_ _asoc anpier interlocutor mutual collabor',
 'current action recours tc andalusia',
 'ep recommend precautionari principl moratorium group expert decid franc 2011',
 'frack requir countri regul case',
 'sierra de guadarrama nation park citizen particip frustrat',
 'real problem infraprotectionist speech condemn',
 'precautionari principl sustain equival',
 'popul promis wealth exploit natur resourc conserv',
 'thank greet melilla open line melillamadridasturia regard',
 'govern elimin expir date yogurt prohibit expir',
 'thing nonprecaut usual disast',
 'environment legisl need strengthen advis act insuffici caution',
 'sustain natur resourc auster econom one',
 'precautionari principl preced legisl action',
 'strang fli half spain time cloud let bit ground',
 'sevenyearold girl rape soldier kill deprav',
 'earthquak result human activ',
 'duel cosped santamaría iberdrola goe electr bill',
 'disturb report enresa instabl terrain chosen atc wouldnt appropri reconsid',
 'time celebr weekend lena festiv virgin flower chang snow',
 'flood refer geolog subsoil cast best option atc',
 'borehol indic plaster subsoil',
 'geolog studi atc juli 2012 enresa websit',
 'im afraid atc lot cloth cut evid difficult hide',
 'think atc process restart technic sound save secur time',
 '_villar atc',
 'today reform approv chang hous polici boost sector im afraid isnt',
 'berlusconi parti win new elect itali itali lose',
 '_villar total agre',
 '_villar plant wast',
 'govern amnesti half invas speci theyr set amnesti',
 'torrent ballest write chronicl presid embodi today scienc anticip outrag',
 'soria assur spain miss frack race campoamor said dont run wors',
 'coastal law lawless coast',
 'coastal law lawless coast',
 '_1982_ cheer',
 '_abasc propos vote amend jointli program agre compli',
 'mayor villa attack vox press confer plenari session salari lower villaviciosadeodon',
 'littl slander hurt mayor',
 'villaviciosadeodon',
 '_aznar mayor posit trust vox villaviciosadeodon',
 '_villa renounc diem plenari session consid unnecessari expens neighbor villaviciosadeodon',
 '_villa mayor posit trust vox villaviciosadeodon',
 '_abasc henchman pp villaviciosadeodón attack _villa clear conscienc clear object',
 '_abasc henchman pp attack _villa clear conscienc clear object villaviciosadeodon',
 '_abasc _villa neighbor dont attack',
 'new plenari session mayor explain eat spend protocol forward villaviciosadeodon',
 'mayor fail compli plenari agreement tell urg forc follow villaviciosadeodon',
 'coward brave 18 year old commit better spain cheer inma villaviciosadeodon',
 'ú2 barcelona 2015 spectacular',
 'victoria love prudenc',
 'reveng justic',
 'know _iglesia didnt want debat _rivera',
 'cheer',
 'presid illus',
 'yovoyvistalegre13d dream come true albertrivera',
 'vistalegrenaranja illus albertrivera presid 20d',
 'venceralailusión',
 'complex yovotoaalbert',
 '_vox _villa _pozuelo _torrelodon _majadahonda _lasroza _hoyo',
 '_rivera great carlo',
 'newprofilephoto',
 'studi corrobor andalusia worst financ look way',
 'win independentista alfredo pérez rubalcaba _rubalcaba_',
 'hernando pp instead come andalusia employ plan invest commit financ reform dedic insult',
 'blame fact nation pp imag dirt barrack andalusian pita lie ppa']
```
Find out the unique words and their counts in the given tweets using np.unique() function.
```python
vocabulary, counts = np.unique(tweets_corpus, return_counts=True)
vocabulary.shape, counts.shape
def get_one_hot_vector(word):

    vec = np.zeros((vocabulary.shape[0], ))

    index = (vocabulary == word).argmax()

    vec[index] = 1

    return vec

dataset = []

for word in tweets_corpus:

    dataset.append(get_one_hot_vector(word))

dataset = np.asarray(dataset)

dataset.shape
# Applying Bi-Gram Model
X = np.zeros((dataset.shape[0]-1, dataset.shape[1]*2)) # Bigram

for i in range(X.shape[0]-1):
    X[i] = np.hstack((dataset[i], dataset[i+1]))
print(X[0], X[0].shape, X.shape)

# Applying TFidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer()
XV = tf_transformer.fit_transform(X).toarray()
```
Output
```python
rray([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.70710678,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.70710678, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        ])
```
### Label Encoding
Encode the political parties using one hot encoder, so that we can find out the target variable using K-means clustering approach that depends on defining a cluster to a word based on its neighbours.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = le.fit_transform(df['partido'])
df.drop("partido", axis=1, inplace=True)
df["partido"] = label
df
```
Output
```python
English	partido
0	_rm _ good cheer spring start	1
1	_moran big fan social radic face democrat devalu	1
2	_presion new owner renew time longer problem m...	1
3	_dolera energi price differenti germani franc ...	1
4	import spread messag rais awar retweet	1
...	...	...
94	newprofilephoto	0
95	studi corrobor andalusia worst financ look way	1
96	win independentista alfredo pérez rubalcaba _r...	1
97	hernando pp instead come andalusia employ plan...	1
98	blame fact nation pp imag dirt barrack andalus...	1
```
### Word2Vec
The Word2Vec model is used to extract the notion of relatedness across words or products such as semantic relatedness, synonym detection, concept categorization, selectional preferences, and analogy. A Word2Vec model learns meaningful relations and encodes the relatedness into vector similarity.

```python
# Creating a word2vec model
w2v_model = Word2Vec(min_count=3,
                     window=4,
                     sample=1e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=multiprocessing.cpu_count()-1)

w2v_model.build_vocab(sentences, progress_per=50000)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
w2v_model.init_sims(replace=True)
w2v_model.save("word2vec.model")
file_export = file_model.copy()
file_export['old_title'] = file_export.English
file_export.old_title = file_export.old_title.str.join(' ')
file_export.English = file_export.English.apply(lambda x: ' '.join(bigram[x]))
file_export.partido = file_export.partido.astype('int8')
```
### K-Means Clustering
The main idea behind this approach is that negative and positive words usually are surrounded by similar words. This means that if we would have movie reviews dataset, word ‘boring’ would be surrounded by the same words as word ‘tedious’, and usually such words would have somewhere close to the words such as ‘didn’t’ (like), which would also make word didn’t be similar to them. On the other hand, it would be unlikely to have happened, that word ‘tedious’ had more similar surrounding to word ‘exciting’, than to word ‘boring’. With such assumption, words could form clusters (based on similarity of their surrounding) of negative words that have similar surroundings, positive words that have similar surroundings, and some neutral words that end up between them (such as ‘movie’).

```python
word_vectors = Word2Vec.load('/content/word2vec.model').wv
model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors.astype('double'))
word_vectors.similar_by_vector(model.cluster_centers_[1], topn=10, restrict_vocab=None)
```
Output
```python
[('q', 1.0000001192092896),
 ('y', 0.29571235179901123),
 ('3', 0.29370227456092834),
 ('v', 0.2920180559158325),
 ('t', 0.2912953495979309),
 ('k', 0.2904706597328186),
 ('b', 0.2899188995361328),
 ('9', 0.2889474630355835),
 ('w', 0.28844600915908813),
 ('s', 0.2878337502479553)]
```

```python
words['cluster_value'] = [1 if i==positive_cluster_index else -1 for i in words.cluster]
words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)
words['sentiment_coeff'] = words.closeness_score * words.cluster_value
words.head(10)
```
Output

```python
words	vectors	cluster	cluster_value	closeness_score	sentiment_coeff
0		[-0.0066649946, 0.038520537, 0.053885072, 0.12...	0	-1	35.369154	-35.369154
1	e	[-0.0103809135, 0.040530924, 0.05399493, 0.118...	0	-1	39.282472	-39.282472
2	a	[-0.006433389, 0.03976579, 0.047027912, 0.1147...	0	-1	32.173391	-32.173391
3	r	[-0.010321431, 0.04335944, 0.051656134, 0.1157...	0	-1	36.368626	-36.368626
4	i	[-0.01026403, 0.039620288, 0.04810302, 0.11551...	0	-1	32.736377	-32.736377
5	o	[-0.01147047, 0.04001286, 0.051128674, 0.11199...	0	-1	30.434056	-30.434056
6	n	[-0.0026341928, 0.03622556, 0.050882377, 0.116...	0	-1	38.033099	-38.033099
7	t	[-0.0015534153, 0.03577799, 0.05681902, 0.1210...	0	-1	28.959745	-28.959745
8	l	[-0.01180796, 0.043870233, 0.053773, 0.1216033...	0	-1	30.796244	-30.796244
9	s	[-0.009624589, 0.035570484, 0.047106408, 0.121...	0	-1	30.414874	-30.414874
```

```python
words[['words', 'sentiment_coeff']].to_csv('sentiment_dictionary.csv', index=False)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
tfidf.fit(file_weighting.English)
features = pd.Series(tfidf.get_feature_names())
transformed = tfidf.transform(file_weighting.English)
def create_tfidf_dictionary(x, transformed_file, features):
    '''
    create dictionary for each input sentence x, where each word has assigned its tfidf score
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer

    '''
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo

def replace_tfidf_words(x, transformed_file, features):
    '''
    replacing each word with it's calculated tfidf dictionary with scores of each word
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    '''
    dictionary = create_tfidf_dictionary(x, transformed_file, features)   
    return list(map(lambda y:dictionary[f'{y}'], x.English.split()))
replacement_df = pd.DataFrame(data=[replaced_closeness_scores, replaced_tfidf_scores, file_weighting.English, file_weighting.partido]).T
replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence', 'sentiment']
replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
replacement_df['prediction'] = (replacement_df.sentiment_rate>0).astype('int8')
replacement_df['sentiment'] = [1 if i==1 else 0 for i in replacement_df.sentiment]
```
Output
```python
English	partido
0	_ r m _ g o o d c h e e r s p r i n g ...	1
1	_ m o r a n b i g f a n s o c i a l r ...	1
2	_ p r e s i o n n e w o w n e r r e n e ...	1
3	_ d o l e r a e n e r g i p r i c e d i ...	1
4	i m p o r t s p r e a d m e s s a g r a ...	1
...	...	...
93	n e w p r o f i l e p h o t o	0
94	s t u d i c o r r o b o r a n d a l u s i ...	1
95	w i n i n d e p e n d e n t i s t a a l f ...	1
96	h e r n a n d o p p i n s t e a d c o m ...	1
97	b l a m e f a c t n a t i o n p p i m ...	1
```
### CNN 
Convolution Neural Networks(CNNs) are multi-layered artificial neural networks with the ability to detect complex features in data, for instance, extracting features in image and text data. CNNs have majorly been used in computer vision tasks such as image classification, object detection, and image segmentation. However, recently CNNs have been applied to text problems.

```python
from tensorflow.keras.layers import Dense, Embedding,GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding

model_1D = Sequential([
    Embedding(vocab_size, 8, input_length=max_length),
   Conv1D(32, 4, activation='relu',use_bias = True),
    GlobalMaxPooling1D(),
  Dense(10, activation='relu'),
  Dense(1, activation='sigmoid')
])
history = model_1D.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```
Output
```python
Epoch 1/10
3/3 [==============================] - 1s 84ms/step - loss: 0.6886 - acc: 0.6410 - val_loss: 0.6740 - val_acc: 0.9500
Epoch 2/10
3/3 [==============================] - 0s 17ms/step - loss: 0.6825 - acc: 0.6667 - val_loss: 0.6615 - val_acc: 0.9500
Epoch 3/10
3/3 [==============================] - 0s 19ms/step - loss: 0.6764 - acc: 0.6667 - val_loss: 0.6496 - val_acc: 0.9500
Epoch 4/10
3/3 [==============================] - 0s 16ms/step - loss: 0.6703 - acc: 0.6667 - val_loss: 0.6379 - val_acc: 0.9500
Epoch 5/10
3/3 [==============================] - 0s 16ms/step - loss: 0.6647 - acc: 0.6667 - val_loss: 0.6259 - val_acc: 0.9500
Epoch 6/10
3/3 [==============================] - 0s 17ms/step - loss: 0.6598 - acc: 0.6667 - val_loss: 0.6141 - val_acc: 0.9500
Epoch 7/10
3/3 [==============================] - 0s 16ms/step - loss: 0.6535 - acc: 0.6667 - val_loss: 0.6027 - val_acc: 0.9500
Epoch 8/10
3/3 [==============================] - 0s 20ms/step - loss: 0.6483 - acc: 0.6667 - val_loss: 0.5902 - val_acc: 0.9500
Epoch 9/10
3/3 [==============================] - 0s 18ms/step - loss: 0.6421 - acc: 0.6667 - val_loss: 0.5775 - val_acc: 0.9500
Epoch 10/10
3/3 [==============================] - 0s 17ms/step - loss: 0.6370 - acc: 0.6667 - val_loss: 0.5641 - val_acc: 0.9500
```
### Test Accuracy
Output
```python
1/1 [==============================] - 0s 20ms/step - loss: 0.5641 - acc: 0.9500
Testing Accuracy is 94.9999988079071 
```

### LSTM
LSTM stands for Long-Short Term Memory. LSTM is a type of recurrent neural network but is better than traditional recurrent neural networks in terms of memory. Having a good hold over memorizing certain patterns LSTMs perform fairly better. As with every other NN, LSTM can have multiple hidden layers and as it passes through every layer, the relevant information is kept and all the irrelevant information gets discarded in every single cell. LSTM is effective in memorizing important information.If we look and other non-neural network classification techniques they are trained on multiple word as separate inputs that are just word having no actual meaning as a sentence, and while predicting the class it will give the output according to statistics and not according to meaning. That means, every single word is classified into one of the categories.This is not the same in LSTM. In LSTM we can use a multiple word string to find out the class to which it belongs. This is very helpful while working with Natural language processing. If we use appropriate layers of embedding and encoding in LSTM, the model will be able to find out the actual meaning in input string and will give the most accurate output class. The following code will elaborate the idea on how text classification is done using LSTM.

```python
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
def RNN():
    inputs = Input(name='inputs',shape=[max_length])
    layer = Embedding(vocab_size,50,input_length=max_length)(inputs)
    layer = LSTM(100)(layer)
    layer = Dense(100,name='FC1')(layer)
    layer = Activation('softmax')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
model_RNN = RNN()
model_RNN.summary()
model_RNN.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
RNN_fit = model_RNN.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss_RNN, accuracy_RNN = model_RNN.evaluate(X_test,y_test)
print('Testing Accuracy is {} '.format(accuracy_RNN*100))
```
Output
```python
Epoch 1/10
3/3 [==============================] - 3s 406ms/step - loss: 0.6923 - accuracy: 0.6667 - val_loss: 0.6810 - val_accuracy: 0.9500
Epoch 2/10
3/3 [==============================] - 1s 194ms/step - loss: 0.6862 - accuracy: 0.6667 - val_loss: 0.6421 - val_accuracy: 0.9500
Epoch 3/10
3/3 [==============================] - 0s 158ms/step - loss: 0.6740 - accuracy: 0.6667 - val_loss: 0.6230 - val_accuracy: 0.9500
Epoch 4/10
3/3 [==============================] - 0s 148ms/step - loss: 0.6710 - accuracy: 0.6667 - val_loss: 0.6165 - val_accuracy: 0.9500
Epoch 5/10
3/3 [==============================] - 0s 159ms/step - loss: 0.6652 - accuracy: 0.6667 - val_loss: 0.6090 - val_accuracy: 0.9500
Epoch 6/10
3/3 [==============================] - 0s 143ms/step - loss: 0.6682 - accuracy: 0.6667 - val_loss: 0.6042 - val_accuracy: 0.9500
Epoch 7/10
3/3 [==============================] - 0s 153ms/step - loss: 0.6632 - accuracy: 0.6667 - val_loss: 0.5993 - val_accuracy: 0.9500
Epoch 8/10
3/3 [==============================] - 0s 150ms/step - loss: 0.6600 - accuracy: 0.6667 - val_loss: 0.5933 - val_accuracy: 0.9500
Epoch 9/10
3/3 [==============================] - 0s 159ms/step - loss: 0.6583 - accuracy: 0.6667 - val_loss: 0.5888 - val_accuracy: 0.9500
Epoch 10/10
3/3 [==============================] - 1s 222ms/step - loss: 0.6462 - accuracy: 0.6667 - val_loss: 0.5825 - val_accuracy: 0.9500

1/1 [==============================] - 0s 63ms/step - loss: 0.5825 - accuracy: 0.9500
Testing Accuracy is 94.999998807907
```
### Building Model
```python
def build_model(vocab_size,length):
    
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size,100)(inputs1)
    conv1 = Conv1D(filters=32,kernel_size=4,activation='relu')(embedding1)
    drop1 = Dropout(0.4)(conv1)
    maxpool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(maxpool1)
    
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size,100)(inputs2)
    conv2 = Conv1D(filters=32,kernel_size=6,activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    maxpool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(maxpool2)
    
    #channel 3 
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size,100)(inputs3)
    conv3 = Conv1D(filters=32,kernel_size=8,activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    maxpool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(maxpool3)
    
    #merge 
    merged = concatenate([flat1,flat2,flat3])
    
    #Dense layers
    Dense1 = Dense(10,activation='relu')(merged)
    outputs = Dense(1,activation='sigmoid')(Dense1)
    
    model = Model(inputs = [inputs1,inputs2,inputs3],outputs=outputs)
    
    #compile
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics = ['accuracy'])
    
    model.summary()
    
    return model
_, acc = model.evaluate([X_train,X_train,X_train], y_train, verbose=0) 
print('Train Accuracy: %.2f' % (acc*100)) 
_, acc = model.evaluate([X_test,X_test,X_test], y_test, verbose=0) 
print('Test Accuracy: %.2f' % (acc*100))
```
Output
```python
_, acc = model.evaluate([X_train,X_train,X_train], y_train, verbose=0) 
print('Train Accuracy: %.2f' % (acc*100)) 
_, acc = model.evaluate([X_test,X_test,X_test], y_test, verbose=0) 
print('Test Accuracy: %.2f' % (acc*100))
```