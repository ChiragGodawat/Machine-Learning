import pandas as pd
import numpy as np

from sklearn import preprocessing, metrics
from tensorflow.keras import layers, utils, backend as K
from tensorflow.keras.models import Model
import config


def create_model(data, cat_cols):
    inputs = []
    outputs = []

    for c in cat_cols:
        num_unique_values = data[c].nunique()

        # Min Size = half of unique values and max generally=50 (Unless having millions of features)
        emb_dim = int(min(np.ceil(num_unique_values/2), 50))
        inp = layers.Input(shape=(1,))

        # Embedding Size Always number of unique values + 1
        out = layers.Embedding(
            num_unique_values + 1, emb_dim, name=c
        )(inp)

        out = layers.SpatialDropout1D(0.3)(out)

        out = layers.Reshape(target_shape=(emb_dim, ))(out)

        inputs.append(inp)
        outputs.append(out)

    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=y)

    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


def run(fold):
    df = pd.read_csv(config.CAT_DAT_TRAIN_FOLD)
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

        lbl_enc = preprocessing.LabelEncoder()

        df[col] = lbl_enc.fit_transform(df[col].values)

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    model = create_model(df, features)

    xtrain = [df_train[features].values[:, k] for k in range(len(features))]
    xvalid = [df_valid[features].values[:, k] for k in range(len(features))]

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    # Categorization
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    model.fit(
        xtrain,
        ytrain_cat,
        validation_data=(xvalid, yvalid_cat),
        verbose=1,
        batch_size=1024,
        epochs=3
    )

    valid_preds = model.predict(xvalid)[:, 1]

    print(f"Fold: {fold}  ROC_AUC:{metrics.roc_auc_score(yvalid, valid_preds)}")

    K.clear_session()


if __name__ == "__main__":
    for i in range(0, 5):
        run(i)
