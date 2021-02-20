#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-


import sys
import re
import argparse
from math import *
from collections import Counter
import numpy as np

# du fait d'erreurs de calcul, on se retrouve parfois avec
#  des distances negatives
# on prend ici une valeur minimale de distance, positive (pour pouvoir
#  prendre la racine) et non nulle (pour pouvoir prendre l'inverse)
MINDIST =  1e-18


class Example:
    """
    Un exemple : 
    vector = representation vectorielle (Ovector) d'un objet
    gold_class = la classe gold pour cet objet
    """
    def __init__(self, example_number, gold_class, indices):
        self.gold_class = gold_class
        self.example_number = example_number
        self.vector = Ovector(indices)

    def add_feat(self, featname,  is_train, val):
        self.vector.add_feat(featname, is_train, val)


class Ovector:
    """
    Un vecteur representant un objet

    membres
    - f= simple dictionnaire nom_de_trait => valeur
         Les traits non stockes correspondent e une valeur nulle
    - calc_norm : la norme au carre
    """
    def __init__(self, indices):
        self.indices = indices
        self.f = np.zeros((len(self.indices),)) 
        # initialisation avec np.array de taille v
        self.norm = 0 
        # utile pour le calcul du cosinus
        self.norm_square = 0 
        # e calculer apres, car e l'initialisation le vecteur est vide

    def __len__(self):
        return len(self.f)

    def add_feat(self, featname, is_train, val=0.0): 
        # fonction qui ajoute feat au vecteur (f)
        # si apprentissage et mot inconnu
        if featname not in self.indices:
            if is_train : 
                self.indices.add_feat(featname) 
                # ajout de la feat aux indices
                self.f = np.concatenate((self.f, [val])) 
                # augmentation taille du vecteur par concat�nation
        else: # sinon, si le mot est dans le voc
            self.f[self.indices.get_i(featname)] += val

    def resize(self):
        """Renvoie un vecteur de la taille des indices 
        en ajoutant des z�ros. """
        diff = len(self.indices)-len(self.f)
        # diff�rence de taille entre taille du voc et taille du vecteur
        self.f = np.concatenate((self.f, np.zeros((diff,))))
        # concat�nation de z�ros pour avoir un vec de la taille du voc

    def prettyprint(self):
        # tri des traits par valeur numerique decroissante (-self.f[x])
        #  et par ordre alpha en cas d'egalite
        for feat in sorted(self.f, key=lambda x: (-self.f[x], x)):
            print(feat+"\t"+str(self.f[feat]))

class Indices:
    """Stocke vocabulaire et lien entre mots et indices 
    (dans les 2 sens)"""
    def __init__(self):
         self.w2i = {} # clefs = mots du voc, 
         # valeurs = indices dans i2w
         self.i2w = [] # liste des mots du voc

    def add_feat(self, word):
        """Ajoute un mot au vocabulaire"""
        if word not in self.i2w:
            self.i2w.append(word)
            self.w2i[word] = len(self.i2w)-1

    def get_i(self, word):
        return self.w2i.get(word)
    
    def get_w(self, i):
        return self.i2w[i]

    def __contains__(self, word):
        return word in self.i2w

    def __len__(self):
        return len(self.i2w)

class Matrix:
    """Classe qui contient une matrice, les normes et normes carrées, 
    et des méthodes de calcul matriciel"""

    def __init__(self, examples):
        # met tous les vecteurs a� la bonne taille 
        # avant de les assembler en matrice
        for ex in examples:
            if len(ex.vector) < len(indices):
                ex.vector.resize()
            else: # arret des qu'on a vecteur de la taille du voc
                break
        # matrice = assemblage des vecteurs des exemples
        self.array = np.vstack([ex.vector.f for ex in examples])
        self.normes_carre = np.sum(self.array**2, axis=1)[:, np.newaxis] 
        # vecteur de normes des docs
        self.normes = np.sqrt(self.normes_carre)
        self.shape = self.array.shape
    
    def calc_cos(self, other):
        normed_vec_self = self.array / self.normes
        normed_vec_other = other.array / other.normes
        return np.dot(normed_vec_self, normed_vec_other.T)

    def calc_dist(self,other):
        a = self.array
        b = other.array
        sum_norm_sq = self.normes_carre + other.normes_carre.reshape((
            other.normes_carre.shape[0],))

        return np.sqrt(sum_norm_sq - 2*(np.dot(a,b.T)))

    def __getitem__(self, position):
        return self.array[position]

    def __iter__(self):
        return self.array

    def __len__(self):
        return len(self.array)

class KNN:
    """
    K-NN pour la classification de documents (multiclasse)

    membres = 

    k = l'hyperparametre K : le nombre de voisins a considerer

    examples = liste d'instances de Example

    classes = liste des classes (telles que recensees dans les exemples)

    """
    def __init__(self, examples, indices, 
                        K=1, 
                        weight_neighbors=None, 
                        use_cosinus=False, 
                        trace=False):
        """ 
        simple positionnement des membres 
        et recensement des classes connues
        """
        

        # la matrice et les gold classes associ�es 
        # = (matrice, gold_classes)
        self.examples, self.gold_classes = examples
        # self.examples = examples # FIXME pas d'attribut examples
        # indices du voc
        self.indices = indices
        # le nb de voisins
        self.K = K
        # booleen : on pondere les voisins (par inverse de la distance) 
        # ou pas
        self.weight_neighbors = weight_neighbors

        # booleen : pour utiliser plutot la similarite cosinus
        self.use_cosinus = use_cosinus

        self.trace = trace

        if self.trace:
            print(f"Shape de la matrice d'apprentissage: {self.examples.shape}")


    def classify(self, test): # FIXME r��crire l'algo vec la matrice
        """
        Realise la prediction du classifieur K-NN pour le ovector
        pour les valeurs de k allant de 1 e self.K

        A partir d'un vecteur de traits representant un objet
        retourne un vecteur des classes assignees de longueur K : 
        la classe e la i-eme  position est la classe assignee par 
        l'algo K-NN, avec K=i
        """
        if self.use_cosinus:
            func_mesure = Matrix.calc_cos
        else:
            func_mesure = Matrix.calc_dist

        mesures = func_mesure(test,self.examples) 
        # renvoie matrice de mesures pour chaque doc test avec chaque doc example

        tri_indices = np.argsort(mesures, axis=1) 
        # matrice avec ordre des voisins par ligne

        res = np.empty((test.array.shape[0],self.K), dtype=object) 
        # matrice qui stocke la prediction pour chaque valeur de k

        k_voisins = np.empty((test.array.shape[0],self.K), dtype=object) 
        # array des gold_class des k + proches voisins, ordonn�s
        # shape = autant de colonnes que docs tests, autant de lignes que K

        if self.use_cosinus:   # pour cosinus : k derni�res colonnes
            k_indices = np.flip(tri_indices[:, -self.K:], axis=1) 
            # flip pour avoir le plus proche en premier
        else:  # pour distance : k premi�re colonnes
            k_indices = tri_indices[:, :self.K]

        # RECUPERATION DES GOLD CLASS DES K-VOISINS avec COS :
        #  K derni�res colonnes de la matrice TRI_INDICES

        for i,row in enumerate(k_indices): # parcours des k derni�res 
            # colonnes de la matrice de r�sultats
            for j,rank in enumerate(row):
                k_voisins[i][j] = self.gold_classes[rank]

        # VOTE
        for i in range(0, self.K):
            for j,voisins_doc in enumerate(k_voisins):
                vote = Counter(np.sort(voisins_doc[:i+1])) # co
                res[j][i] = max(vote, key=vote.get)

        return res
        # retourne une matrice avec classe predite 
        # selon longueur K pour chaque doc
        

    def evaluate_on_test_set(self, test_examples):
        """ Application du classifieur sur une liste d'exemples 
        de test, et evaluation (accuracy) 
        pour les valeurs de k allant de 1 e self.K
        Retourne une liste d'accuracy (pour les valeurs de k e self.K) 
        = nb de resultats corrects
        """

        # test_examples est un tuple (matrice, gold_classes)
        matrice_test,gold_classes_test = test_examples

        accuracies = [0]*self.K # initialisation de liste 
        # d'accuracy e zero pour tous les k

        predictions = self.classify(matrice_test)

        for i,doc in enumerate(predictions): # pour chaque doc
            for j,pred in enumerate(doc): # pour chaque pr�diction pour 
                # ce doc (avec diff valeurs de k)
                if pred == gold_classes_test[i]:
                    accuracies[j] += 1 # on augmente accuracy pour k = j de 1

        return accuracies
        
        

def read_examples(infile, is_train, indices):
    """ Lit un fichier d'exemples 
    et retourne une liste d'instances de Example
    """
    stream = open(infile)
    examples = []
    example = None
    gold_classes = []
    gold_class = ""
    while 1:
        line = stream.readline()
        if not line:
            break
        line = line[0:-1]
        if line.startswith("EXAMPLE_NB"):
            if example != None:
                examples.append(example) # ajout de l'ex precedent e la liste
                gold_classes.append(gold_class)
            cols = line.split('\t')
            gold_class = cols[3]
            example_number = cols[1]
            example = Example(example_number, gold_class, indices)
        elif line and example != None:
            (featname, val) = line.split('\t')
            example.add_feat(featname, is_train, float(val)) # creation 
            # du vecteur à la volee
            # il sont de taille differentes, mais seront resized a� 
            # la creation de la matrice
    
    if example != None:
        # example.vector.calc_norm() # calcul norm du dernier exemple
        examples.append(example) # ajout du dernier exemple
        gold_classes.append(gold_class)

    matrix = Matrix(examples)

    return (matrix, gold_classes)




usage = """ CLASSIFIEUR de DOCUMENTS, de type K-NN

  """+sys.argv[0]+""" [options] EXAMPLES_FILE TEST_FILE

  EXAMPLES_FILE et TEST_FILE sont au format *.examples
  

"""

parser = argparse.ArgumentParser(usage = usage)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('examples_file', default=None, 
        help='''Exemples utilises comme voisins pour la prediction KNN 
        (au format .examples)''')
parser.add_argument('test_file', default=None, 
        help='Exemples de test (au format .examples)')
parser.add_argument('-t', "--test", action="store_true", 
        help="""A utiliser pour tester les differentes combinaisons 
        d'hyperparametres. Affiche la meilleure combinaison. Default=False""")
parser.add_argument('-k', "--k", default=1, type=int, 
        help='''Hyperparametre K : le nombre max de voisins a considerer 
        pour la classification (toutes les valeurs de 1 a k seront testees). 
        Default=1''')
parser.add_argument('-v', "--trace", action="store_true", 
        help="A utiliser pour declencher un mode verbeux. Default=False")
parser.add_argument('-w', "--weight_neighbors", action="store_true", 
        help="""Ponderation des voisins : si cosinus: ponderation par le cosinus, 
        si distance, ponderation par l'inverse de la distance. Defaut=None""")
parser.add_argument('-c', "--use_cosinus", action="store_true", 
        help="""A utiliser pour passer a une mesure de similarite cosinus, 
        au lieu d'une distance euclidienne. Default=False""")

args = parser.parse_args()

indices = Indices()
#------------------------------------------------------------
# Chargement des exemples d'apprentissage du classifieur KNN
training_examples = read_examples(args.examples_file, indices=indices, 
        is_train=True)
# Chargement des exemples de test
test_examples = read_examples(args.test_file, indices=indices, is_train=False)
len_test = len(test_examples[0].array)

if args.trace:
    print()
    print("APPELS DE READ_EXAMPLES() SUR CORPUS TRAIN ET TEST\n\n")
    print(f"Debut i2w : {indices.i2w[:10]}")
    print(f"Extrait w2i : {list(indices.w2i.items())[:10]}")
    print(f"taille du voc : {len(indices)}")
    print(f"Shape de la matrice d'apprentissage : {training_examples[0].shape}")
    print(f"Shape de la matrice de test : {test_examples[0].shape}")
    print(f"1er ligne de matrice d'apprentissage : {training_examples[0][0]}")
    print()



if args.test == False: # pas de test des hyperparametres

    myclassifier = KNN(examples = training_examples,
                    indices = indices,
                    K = args.k,
                    weight_neighbors = args.weight_neighbors,
                    use_cosinus = args.use_cosinus,
                    trace=args.trace)

    # classification et evaluation sur les exemples de test
    accuracies = myclassifier.evaluate_on_test_set(test_examples)

    for idx, acc in enumerate(accuracies):
        print(f"ACCURACY FOR K = {idx+1} = {acc/len_test:.2%} ({acc}/{len_test})")

else:
    # TESTS DES HYPERPARAMETRES
    if args.trace:
        print("""\n\n***TEST DES HYPERPARAMETRES k et distance/cosinus\n
        (je n'ai pas impl�ment� la pond�ration)\n\n""")
    best_acc = 0
    best_hyper = 0

    if args.k == 1:
        args.k = 5

    for use_cos in [True,False]:
        
        myclassifier = KNN(examples = training_examples,
                    indices = indices,
                    K = args.k,
                    use_cosinus = use_cos,
                    trace=args.trace)
        if args.trace:
            print(f"use_cosinus : {use_cos}")
        accuracies = myclassifier.evaluate_on_test_set(test_examples)

        for idx, acc in enumerate(accuracies):
            if acc > best_acc:
                best_acc = acc
                best_hyper = {"use_cosinus" : use_cos, "K" : idx+1}
            if args.trace:
                print(f"""ACCURACY FOR K = {idx+1} = {acc/len_test:.2%}
                 ({acc}/{len_test})""")
        if args.trace:
            print()

    print(f"""Meilleure combinaison d'hyperparametres : {best_hyper}\n 
    Precision : {best_acc/len_test:.2%} ({best_acc}/{len_test})""")