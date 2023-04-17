import librosa
import numpy as np

# Charger l'audio
filename = "C:/Users/Vincent/OneDrive/Bureau/ProjetTest2023/Motif2b.wav"
audio, sr = librosa.load(filename)

# Configuration des paramètres pour la détection des sifflements de dauphin
FRAME_SIZE = 2048
HOP_LENGTH = 512
THRESHOLD = 0.5

# Calculer l'énergie du signal audio
energie = librosa.feature.rms(y=audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

# Créer un masque binaire pour déterminer où les sifflements de dauphin commencent et se terminent
masque_sifflement = energie > (THRESHOLD * np.max(energie))

# Trouver le début et la fin de chaque sifflement de dauphin
sifflements = []
debut_sifflement = 0
for i in range(len(masque_sifflement) - 1):
    if masque_sifflement[i] and not masque_sifflement[i+1]:
        # Fin d'un sifflement
        temps_debut = librosa.frames_to_time(debut_sifflement * HOP_LENGTH, sr=sr)
        temps_fin = librosa.frames_to_time((i + 1) * HOP_LENGTH, sr=sr)
        sifflements.append((temps_debut, temps_fin))
    if not masque_sifflement[i] and masque_sifflement[i+1]:
        # Début d'un sifflement
        debut_sifflement = i + 1

# Afficher les résultats
for i, sifflement in enumerate(sifflements):
    print(f"Sifflement2b {i+1}: Debut = {sifflement[0]:.2f}s, Fin = {sifflement[1]:.2f}s")