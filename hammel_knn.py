#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

"""  J'ai implémenté la similarité cosinus et la pondération des voisins. 

Si vous lancez le programme avec l'argument -t cela lancera le test des 4 combinaisons d'hyperparamètres weight_neibours et use_cosinus avec le k de votre choix.

Selon mes tests avec k=10, la meilleure combinaison d'hyperparamètres est {'weight_neighbors': True, 'use_cosinus': True, 'K': 5} avec une précision de 84.00% (168/200).

Le calcul des norm_squares des vecteurs d'exemple s'effectue dans la fonction read_examples
avant l'ajout de chaque example à la liste des examples via une méthode définie dans Ovector.

ALICE HAMMEL"""


import sys
import re
import argparse
from math import *
from collections import Counter


# du fait d'erreurs de calcul, on se retrouve parfois avec des distances négatives
# on prend ici une valeur minimale de distance, positive (pour pouvoir prendre la racine) et non nulle (pour pouvoir prendre l'inverse)
MINDIST =  1e-18


class Example:
    """
    Un exemple : 
    vector = représentation vectorielle (Ovector) d'un objet
    gold_class = la classe gold pour cet objet
    """
    def __init__(self, example_number, gold_class):
        self.gold_class = gold_class
        self.example_number = example_number
        self.vector = Ovector()

    def add_feat(self, featname, val):
        self.vector.add_feat(featname, val)


class Ovector:
    """
    Un vecteur représentant un objet

    membres
    - f= simple dictionnaire nom_de_trait => valeur
         Les traits non stockés correspondent é une valeur nulle
    - calc_norm : la norme au carré
    """
    def __init__(self):
        self.f = {} # initialisation avec dictionnaire vide
        self.norm = 0 # utile pour le calcul du cosinus
        self.norm_square = 0 # ï¿½ calculer aprï¿½s, car ï¿½ l'initialisation le vecteur est vide

    def add_feat(self, featname, val=0.0): # fonction qui ajoute feat au vecteur (f)
        self.f[featname] = val

    def prettyprint(self):
        # tri des traits par valeur numérique décroissante (-self.f[x])
        #  et par ordre alpha en cas d'egalité
        for feat in sorted(self.f, key=lambda x: (-self.f[x], x)):
            print(feat+"\t"+str(self.f[feat]))

    def distance_to_vector(self, other_vector):
        """ distance euclidienne entre self et other_vector, en ayant precalculé les normes au carré de chacun """
        # NB: passer par la formulation  sigma [ (ai - bi)^2 ] = sigma (ai^2) + sigma (bi^2) -2 sigma (ai*bi) 
        #                                                    = norm_square(A) + norm_square(B) - 2 A.B
        return sqrt(self.norm_square + other_vector.norm_square - 2*(self.dot_product(other_vector)))

    def dot_product(self, other_vector):
        """ rend le produit scalaire de self et other_vector """
        
        # création d'une liste de features présentes dans les 2 vecteurs (le produit des autres vaut zéro)
        feats_1 = set(list(self.f.keys()))
        feats_2 = set(list(other_vector.f.keys()))

        common_feats = list(feats_1.intersection(feats_2))

        dot = 0

        for feat in common_feats:
            dot += self.f[feat]*other_vector.f[feat]

        return dot

    def cosinus(self, other_vector):
        """ rend le cosinus de self et other_vector """
        return self.dot_product(other_vector)/(self.norm*other_vector.norm)

    def calc_norm(self):
        """ Effectue le calcul de la norme carré et de la norme et affecte les valeurs aux attributs self.norm et self.norm_square """
        self.norm_square = sum([feat_value**2 for feat_value in self.f.values()])
        self.norm = sqrt(self.norm_square)


class KNN:
    """
    K-NN pour la classification de documents (multiclasse)

    membres = 

    k = l'hyperparametre K : le nombre de voisins a considerer

    examples = liste d'instances de Example

    classes = liste des classes (telles que recensées dans les exemples)

    """
    def __init__(self, examples, 
                        K=1, 
                        weight_neighbors=None, 
                        use_cosinus=False, 
                        trace=False):
        """ 
        simple positionnement des membres et recensement des classes connues
        """
        # les exemples : liste d'instances de Example
        self.examples = examples
        # le nb de voisins
        self.K = K
        # booleen : on pondere les voisins (par inverse de la distance) ou pas
        self.weight_neighbors = weight_neighbors

        # booleen : pour utiliser plutot la similarité cosinus
        self.use_cosinus = use_cosinus

        self.trace = trace
        

    def classify(self, ovector):
        """
        Réalise la prédiction du classifieur K-NN pour le ovector
        pour les valeurs de k allant de 1 é self.K

        A partir d'un vecteur de traits représentant un objet
        retourne un vecteur des classes assignées de longueur K : la classe ï¿½ la i-eme  position est la classe assignée par l'algo K-NN, avec K=i
        """

        liste_dist = []

        # ajout de toutes les distances (entre ovector et les vecteurs examples) à la liste des distances

        for ex_train in self.examples:
            if self.use_cosinus :
                liste_dist.append((ex_train.vector.cosinus(ovector), ex_train.gold_class))
            else:
                liste_dist.append((ex_train.vector.distance_to_vector(ovector), ex_train.gold_class))

        if self.use_cosinus:
            liste_dist.sort(reverse=True) # si on utilise similarité cos on tri du + grd au + petit
        else:
            liste_dist.sort() # sinon tri de la liste par distance (+ petit à  + grande)

        res = [] # liste qui stocke la prédiction pour chaque valeur de k

        k_voisins = [] # liste des k + proches voisins

        for i in range(0,self.K):
            k_voisins.append(liste_dist[i]) # liste ordonnée des k + proches voisins (mesure, gold_class) (il suffit d'ajouter le ième)

            # calcul du vote

            if self.weight_neighbors: # si pondération
                vote = {}
                for mesure,gold_class in k_voisins:
                    if self.use_cosinus:
                        vote[gold_class] = vote.get(gold_class,0) + mesure #pondération par cosinus
                    else:
                        vote[gold_class] = vote.get(gold_class,0) + 1/mesure # pondération par inverse de distance
                # je n'ai pas implémenté le tri alphabétique car une égalité est très peu probable

            else: # sans pondération
                k_voisins.sort(key= lambda x : x[1]) # tri par ordre alphabétique pour renvoyer 1 résultat dans l'ordre alpha dans le cas d'une égalité
                vote = Counter([gold for (_,gold) in k_voisins]) # calcul des résultats sans pondération
            
            res.append(max(vote, key=vote.get)) # ajout de la prédiction à la liste de prédiction

        return res
        # retourne une liste avec classe prédite selon longueur K
        

    def evaluate_on_test_set(self, test_examples):
        """ Application du classifieur sur une liste d'exemples de test, et evaluation (accuracy) 
        pour les valeurs de k allant de 1 à self.K
        Retourne une liste d'accuracy (pour les valeurs de k à self.K) = nb de résultats corrects
        """

        accuracies = [0]*self.K # initialisation de liste d'accuracy à  zéro pour tous les k

        for test_ex in test_examples:
            # print(test_ex.gold_class)
            predictions = self.classify(test_ex.vector)
            for i,pred in enumerate(predictions):
                if pred == test_ex.gold_class:
                    accuracies[i] += 1

        return accuracies
        
        

def read_examples(infile):
    """ Lit un fichier d'exemples 
    et retourne une liste d'instances de Example
    """
    stream = open(infile)
    examples = []
    example = None
    while 1:
        line = stream.readline()
        if not line:
            break
        line = line[0:-1]
        if line.startswith("EXAMPLE_NB"):
            if example != None:
                example.vector.calc_norm() # calcul norm de l'ex précédent
                examples.append(example) # ajout de l'ex précédent à  la liste
            cols = line.split('\t')
            gold_class = cols[3]
            example_number = cols[1]
            example = Example(example_number, gold_class)
        elif line and example != None:
            (featname, val) = line.split('\t')
            example.add_feat(featname, float(val))
    
    if example != None:
        example.vector.calc_norm() # calcul norm du dernier exemple
        examples.append(example) # ajout du dernier exemple
    return examples



usage = """ CLASSIFIEUR de DOCUMENTS, de type K-NN

  """+sys.argv[0]+""" [options] EXAMPLES_FILE TEST_FILE

  EXAMPLES_FILE et TEST_FILE sont au format *.examples
  

"""

parser = argparse.ArgumentParser(usage = usage)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('examples_file', default=None, help='Exemples utilisés comme voisins pour la prédiction KNN (au format .examples)')
parser.add_argument('test_file', default=None, help='Exemples de test (au format .examples)')
parser.add_argument('-t', "--test", action="store_true", help="A utiliser pour tester les différentes combinaisons d'hyperparamètres. Affiche la meilleure combinaison. Default=False")
parser.add_argument('-k', "--k", default=1, type=int, help='Hyperparametre K : le nombre max de voisins a considerer pour la classification (toutes les valeurs de 1 a k seront testees). Default=1')
parser.add_argument('-v', "--trace", action="store_true", help="A utiliser pour declencher un mode verbeux. Default=False")
parser.add_argument('-w', "--weight_neighbors", action="store_true", help="Ponderation des voisins : si cosinus: ponderation par le cosinus, si distance, ponderation par l'inverse de la distance. Defaut=None")
parser.add_argument('-c', "--use_cosinus", action="store_true", help="A utiliser pour passer a une mesure de similarite cosinus, au lieu d'une distance euclidienne. Default=False")

args = parser.parse_args()


#------------------------------------------------------------
# Chargement des exemples d'apprentissage du classifieur KNN
training_examples = read_examples(args.examples_file)
# Chargement des exemples de test
test_examples = read_examples(args.test_file)

print()

if args.test == False: # pas de test des hyperparamètres

    myclassifier = KNN(examples = training_examples,
                    K = args.k,
                    weight_neighbors = args.weight_neighbors,
                    use_cosinus = args.use_cosinus,
                    trace=args.trace)

    # classification et evaluation sur les exemples de test
    accuracies = myclassifier.evaluate_on_test_set(test_examples)

    for idx, acc in enumerate(accuracies):
        print(f"ACCURACY FOR K = {idx+1} = {acc/len(test_examples):.2%} ({acc}/{len(test_examples)})")
else:
    # TESTS DES HYPERPARAMETRES
    print("TEST DES HYPERPARAMETRES weight_neighbors et use_cosinus\n")
    best_acc = 0
    for weight, use_cos in [(True,True),(True,False),(False,True),(False,False)]:
        print(f"weight_neighbors : {weight}, use_cosinus : {use_cos}")
        myclassifier = KNN(examples = training_examples,
                    K = args.k,
                    weight_neighbors = weight,
                    use_cosinus = use_cos)
        accuracies = myclassifier.evaluate_on_test_set(test_examples)

        for idx, acc in enumerate(accuracies):
            if acc > best_acc:
                best_acc = acc
                best_hyper = {"weight_neighbors" : weight, "use_cosinus" : use_cos, "K" : idx+1}
            
            print(f"ACCURACY FOR K = {idx+1} = {acc/len(test_examples):.2%} ({acc}/{len(test_examples)})")
        print()

    print(f"Meilleure combinaison d'hyperparamètres : {best_hyper}\n Précision : {best_acc/len(test_examples):.2%} ({best_acc}/{len(test_examples)})")



# la meilleure combinaison d'hyperparamètres est {'weight_neighbors': True, 'use_cosinus': True, 'K': 5} avec une précision de 84.00% (168/200).

# OUTPUT
# TEST DES HYPERPARAMETRES
# weight_neighbors : True, use_cosinus : True
# ACCURACY FOR K = 1 = 79.00% (158/200)
# ACCURACY FOR K = 2 = 79.00% (158/200)
# ACCURACY FOR K = 3 = 81.00% (162/200)
# ACCURACY FOR K = 4 = 81.50% (163/200)
# ACCURACY FOR K = 5 = 84.00% (168/200)
# ACCURACY FOR K = 6 = 82.00% (164/200)
# ACCURACY FOR K = 7 = 82.50% (165/200)
# ACCURACY FOR K = 8 = 84.00% (168/200)
# ACCURACY FOR K = 9 = 83.50% (167/200)
# ACCURACY FOR K = 10 = 82.50% (165/200)

# weight_neighbors : True, use_cosinus : False
# ACCURACY FOR K = 1 = 61.50% (123/200)
# ACCURACY FOR K = 2 = 61.50% (123/200)
# ACCURACY FOR K = 3 = 61.00% (122/200)
# ACCURACY FOR K = 4 = 61.50% (123/200)
# ACCURACY FOR K = 5 = 62.50% (125/200)
# ACCURACY FOR K = 6 = 61.50% (123/200)
# ACCURACY FOR K = 7 = 61.00% (122/200)
# ACCURACY FOR K = 8 = 60.50% (121/200)
# ACCURACY FOR K = 9 = 59.00% (118/200)
# ACCURACY FOR K = 10 = 59.00% (118/200)

# weight_neighbors : False, use_cosinus : True
# ACCURACY FOR K = 1 = 79.00% (158/200)
# ACCURACY FOR K = 2 = 78.00% (156/200)
# ACCURACY FOR K = 3 = 77.50% (155/200)
# ACCURACY FOR K = 4 = 82.50% (165/200)
# ACCURACY FOR K = 5 = 81.00% (162/200)
# ACCURACY FOR K = 6 = 80.50% (161/200)
# ACCURACY FOR K = 7 = 80.00% (160/200)
# ACCURACY FOR K = 8 = 81.50% (163/200)
# ACCURACY FOR K = 9 = 81.50% (163/200)
# ACCURACY FOR K = 10 = 81.00% (162/200)

# weight_neighbors : False, use_cosinus : False
# ACCURACY FOR K = 1 = 61.50% (123/200)
# ACCURACY FOR K = 2 = 60.00% (120/200)
# ACCURACY FOR K = 3 = 61.50% (123/200)
# ACCURACY FOR K = 4 = 59.50% (119/200)
# ACCURACY FOR K = 5 = 60.00% (120/200)
# ACCURACY FOR K = 6 = 59.00% (118/200)
# ACCURACY FOR K = 7 = 59.50% (119/200)
# ACCURACY FOR K = 8 = 56.00% (112/200)
# ACCURACY FOR K = 9 = 58.50% (117/200)
# ACCURACY FOR K = 10 = 58.50% (117/200)