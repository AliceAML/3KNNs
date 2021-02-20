from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# chargement documents dans matrices de train / de test,

# format doc : classes, tab, doc
# FIXME problème d'encodage dans corpus

def multi_to_mono_label(file):
    """ramène onedocperline au cas mono-label. Renvoie un tuple (matrice,labels) """
    content = []
    labels = []
    with open(file) as f:
        doc = f.readlines()
    for line in doc:
        current_line = line.split("\t") # classes séparées du doc par tab
        classes = current_line[0].split(",") # classes séparées entre elles par virgule

        for label in classes:
            content.append(current_line[1])
            labels.append(label)

    return (content,labels)

train_corp, y_train = multi_to_mono_label("TD3_sklearn/medium.train.onedocperline")
test_corp, y_test = multi_to_mono_label("TD3_sklearn/medium.test.onedocperline")

def make_normalised_matrix(corpus, vectorizer, train=True, idf=True):
    """Renvoie matrice normalisée"""
    # matrice comptes
    if train:
        x = vectorizer.fit_transform(corpus)
    else:
        x = vectorizer.transform(corpus)

    # normalisation
    if idf:
        transformer = TfidfTransformer(smooth_idf=False)
        x = transformer.fit_transform(x)

    return x

vectorizer = CountVectorizer()
X_train = make_normalised_matrix(train_corp, vectorizer)#, idf=False)
X_test = make_normalised_matrix(test_corp, vectorizer, train=False)#, idf=False)


# apprentissage sur train


# model = Perceptron()
model = SVC()
model.fit(X_train, y_train)

# test sur test
print(f"Accuracy X_test : {model.score(X_test, y_test):.2%}")
print(f"Accuracy X_train : {model.score(X_train, y_train):.2%}")


# TODO afficher précisions sur train et test avec sklearn.metrics.accuracy_score


# TODO ajouter parser pour adresses docs + choix du modèle

# TODO recherche en grille pour hyperparamètres

# TODO validation croisée