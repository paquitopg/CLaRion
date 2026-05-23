### 1
Xavier Dupré <xavier.dupre@gmail.com>
dim. 26/04, 21:46
Bonjour,

Un cours mail pour vous rappeler les formalités d'évaluation :

* Vous devez paralléliser un algorithme sur CPU, j'ai proposé en cours celui d'une random forest mais vous pouvez choisir un algorithme de votre de complexité équivalente.
* Vous devez le comparer avec une implémentation existante ou naïve.
* Vous devez me retourner le 24/5 avant 16h votre code, un court rapport détaillant le problème choisi, votre stratégie pour paralléliser, un benchmark comparant deux implémentations, une conclusion sur vos résultats.

Bonne fin de week-end,

Xavier Dupré

### 2
EL BAZ Avner
lun. 27/04, 09:16
Bonjour,

J’espère que vous allez bien.

Nous pensons avoir identifié un sujet pertinent pour ce rendu.

Nous souhaitons réimplémenter CLARA, le modèle de RAG développé par Apple en février, qui fusionne les étapes de retrieval et de génération d’un RAG classique (https://arxiv.org/pdf/2511.18659), et qui repose notamment sur une relaxation du problème de retrieval.

L’objectif du projet serait d’explorer la parallélisation de ce modèle afin d’en améliorer les performances, notamment via des optimisations d’implémentation.

Cet aspect constitue cependant une part expérimentale du projet : nous visons sa mise en œuvre, tout en étant conscients que sa complexité pourra en limiter la portée effective.

La réimplémentation du modèle servira ainsi de base solide pour ces expérimentations.

Notre principale contrainte restera l’accès aux ressources CPU/GPU, et nous essaierons, dans la mesure du possible, de nous rapprocher de leur modèle de référence, Mistral 7B.

Cette démarche vous semble-t-elle pertinente ?

Bonne journée,

Paco et Avner

### 3
Xavier Dupré <xavier.dupre@gmail.com>
lun. 27/04, 10:11
EL BAZ Avner;
GOZE Paco
Boîte de réception
Bonjour,

Ca me va. Je vous conseille un petit modèle, même architecture mais deux couches. Pour l'optimisation des calculs vous n'avez pas besoin que le modèle retourne des réponses correctes, juste qu'il aille plus vite.

Xavier