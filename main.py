import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import  LatentDirichletAllocation 
import matplotlib.pyplot as plt
import spacy

data_folder = Path("text_files/1981/legislatives")
files = data_folder.glob("*.txt")
data=[]
for f in files:
    text=f.read_text(encoding='utf-8')
    data.append({'text': text})
df = pd.DataFrame(data)
print(len(df))
print(df.head())


def plot_top_words(model, vectorizer, n_top_words, title, nb_lines=2):
    feature_names = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(nb_lines, 5, figsize=(30, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

n_features = 1000
n_topics = 10
STOPWORDS = [x.strip() for x in open('data/stop_word_fr.txt').readlines()]
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
df['lemmatized_text'] = [" ".join([token.lemma_ for token in doc]) for doc in nlp.pipe(df['text'])]



tf_vectorizer = CountVectorizer(max_features=n_features)
tf = tf_vectorizer.fit_transform(df['text'])

lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(tf)

plot_top_words(lda, tf_vectorizer, 10, title=None)





tf_vectorizer_sw = CountVectorizer(max_features=n_features, stop_words=STOPWORDS)
tf_sw = tf_vectorizer_sw.fit_transform(df['text'])

lda_sw = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_sw.fit(tf_sw)

plot_top_words(lda_sw, tf_vectorizer_sw, 10, title=None)




df['lemmatized_text'] = [" ".join([token.lemma_ for token in doc]) for doc in nlp.pipe(df['text'])]

tf_vectorizer_lemma = CountVectorizer(max_features=n_features, stop_words=STOPWORDS)
tf_lemma = tf_vectorizer_lemma.fit_transform(df['lemmatized_text'])

lda_lemma = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_lemma.fit(tf_lemma)

plot_top_words(lda_lemma, tf_vectorizer_lemma, 10,title=None)
