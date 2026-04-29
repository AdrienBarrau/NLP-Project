import pandas as pd
import spacy
from pathlib import Path
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

FOLDER_TEXTS = Path("text_files/1981/legislatives")
FILE_METADATA = "data/archelect_search.csv" 
FILE_STOPWORDS = "data/stop_word_fr.txt"



data_texts = []
for f in FOLDER_TEXTS.glob("*.txt"):
 
    file_id = f.name.replace('.txt', '')
    text = f.read_text(encoding='utf-8')
    data_texts.append({'id': file_id, 'text': text})

df_texts = pd.DataFrame(data_texts)

df_meta = pd.read_csv(FILE_METADATA, sep=",", dtype=str)
df_meta_1981 = df_meta[
    (df_meta['contexte-election'] == 'législatives') &
    (df_meta['date'].str.startswith('1981', na=False))
][['id', 'titulaire-soutien']]


df = pd.merge(df_texts, df_meta_1981, on='id', how='left')

df['parti'] = df['titulaire-soutien'].str.split(';').str[0]
df['parti'] = df['parti'].fillna("Inconnu")
df['parti'] = df['parti'].replace(['non disponible', 'Non disponible', 'ND', 'nd'], 'Inconnu')

print(f"-> Nombre total de documents prêts pour l'analyse : {len(df)}")
print(f"-> Top 10 des partis : \n{df['parti'].value_counts().head(10)}\n")


print("3. Initialisation des modèles NLP (Embeddings et Vectorizer)...")

docs = df["text"].tolist()
classes = df["parti"].tolist() 
sentence_model = SentenceTransformer("dangvantuan/sentence-camembert-base")
print("-> Calcul des embeddings (cela peut prendre quelques minutes)...")
embeddings = sentence_model.encode(docs, show_progress_bar=True)


stopwords_fr = [x.strip() for x in open(FILE_STOPWORDS, encoding='utf-8').readlines()]
vectorizer_model = CountVectorizer(stop_words=stopwords_fr)


print("4. Entraînement de BERTopic...")

topic_model = BERTopic(
    embedding_model=sentence_model,
    vectorizer_model=vectorizer_model,
    language="multilingual",
    calculate_probabilities=False 
)

topics, probabilities = topic_model.fit_transform(docs, embeddings)

print("\nTop 5 des Thèmes identifiés :")
print(topic_model.get_topic_info().head())


import matplotlib.pyplot as plt



fig_barchart = topic_model.visualize_barchart(top_n_topics=10, n_words=10, title="BERTopic : Top Mots par Thème")
fig_barchart.write_image("bertopic_results_parties.png", width=1600, height=1000, scale=2)
print("-> Graphique des thèmes globaux sauvegardé (bertopic_results_parties.png)")



topics_per_class = topic_model.topics_per_class(docs, classes=classes)

top_10_partis = df['parti'].value_counts().nlargest(10).index.tolist()
df_filtered = topics_per_class[topics_per_class['Class'].isin(top_10_partis)].copy()

df_filtered = df_filtered[df_filtered['Topic'].isin(range(10))].copy()

topic_info = topic_model.get_topic_info()
dict_topic_names = {}

for _, row in topic_info.iterrows():
    topic_id = row['Topic']
    if topic_id in range(10):
        words = ", ".join(row['Representation'][:3])
        dict_topic_names[topic_id] = f"Thème {topic_id} : {words}"

df_filtered['Legend_Name'] = df_filtered['Topic'].map(dict_topic_names)


pivot_df = df_filtered.pivot_table(
    index='Class', 
    columns='Legend_Name', 
    values='Frequency', 
    aggfunc='sum', 
    fill_value=0
)


pivot_df = pivot_df.reindex(sorted(pivot_df.columns, key=lambda x: int(x.split(' ')[1])), axis=1)

plt.figure(figsize=(16, 9))
ax = pivot_df.plot(kind='bar', stacked=True, colormap='tab10', figsize=(16, 9), width=0.8)

plt.title("Répartition des 10 thèmes principaux par parti (Législatives 1981)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Parti Politique", fontsize=14)
plt.ylabel("Nombre de documents", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)


plt.legend(title="Thèmes globaux (3 premiers mots)", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)
plt.tight_layout()

plt.savefig("thematic_propensity_by_party.png", dpi=300, bbox_inches='tight')
plt.close()

print("-> Graphique de propension par parti sauvegardé (thematic_propensity_by_party.png)")
print("\nScript terminé avec succès !")
