import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import naive_bayes, metrics, model_selection

from sklearn.feature_extraction.text import CountVectorizer

from src import config

if __name__ == "__main__":
    df = pd.read_csv(config.IMDB_DATA)

    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    y = df.sentiment.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f_, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f_

    for fold in range(5):
        train_df = df[df['kfold'] != fold].reset_index(drop=True)
        test_df = df[df['kfold'] == fold].reset_index(drop=True)

        count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

        count_vec.fit(train_df.review)

        train_df_transformed = count_vec.transform(train_df.review)
        test_df_transformed = count_vec.transform(test_df.review)

        model = naive_bayes.MultinomialNB()

        model.fit(train_df_transformed, y=train_df.sentiment)

        preds = model.predict(test_df_transformed)

        accuracy = metrics.accuracy_score(test_df.sentiment, preds)

        print(f"Fold; {fold}, Accuracy:{accuracy}", "\n")
