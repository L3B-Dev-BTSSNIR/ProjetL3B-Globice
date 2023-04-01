import librosa
import numpy 

# Charger le signal audio
filename = "C:/Users/Vincent/Music/TEST.wav"
y, sr = librosa.load(filename)

# Détecter les onsets
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

# Détecter les offsets
offsets = []
pad_time = 0.025
for onset in onsets:
    onset_time = onset * sr
    offset_time = onset_time + int(pad_time * sr)
    energy_window = y[onset_time:offset_time]
    energy_derivative = numpy.diff(energy_window)
    zero_crossings = librosa.zero_crossings(energy_derivative, pad=False)
    offset = onset_time + numpy.argmax(zero_crossings)
    offsets.append(offset)

offsets = numpy.array(offsets)

# Convertir les échantillons en temps
onset_times = librosa.samples_to_time(onsets, sr=sr)
offset_times = librosa.samples_to_time(offsets, sr=sr)

# Afficher les onsets et offsets en temps
print("Onsets (sec):", onset_times)
print("Offsets (sec):", offset_times)