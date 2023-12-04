import numpy as np

# Fonction d'activation hardlim
def hardlim(input):
    return 1 if input >= 0 else 0

# Données initiales
W = np.array([1, 1, 1, 1])  # Poids
b = 0  # Biais
alpha = 1  # Taux d'apprentissage

# Observations et cibles
observations = np.array([
    [1, 1, 1, 0],
    [0, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])
targets = np.array([1, 0, 1, 0, 0])

# Processus de mise à jour
for i, X in enumerate(observations):
    # Calcul de la sortie
    output = hardlim(np.dot(W, X) + b)
    
    # Mise à jour des poids et du biais
    W += alpha * (targets[i] - output) * X
    b += alpha * (targets[i] - output)
    print(W,b)


