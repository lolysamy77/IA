import random
import math
import time

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = [math.exp(i) for i in x]
    s = sum(exp_x)
    return [i / s for i in exp_x]

def one_hot(index, size):
    vec = [0] * size
    vec[index] = 1
    return vec

# === Données d'entraînement ===
text = "Brumeville, petite cité au bord d’un lac immobile, semblait vivre dans une éternelle aube. Les rues y étaient pavées de pierres pâles, et les fenêtres reflétaient une lumière sans soleil. On disait que le vent lui-même y marchait à pas comptés, et quiconque y restait trop longtemps finissait par perdre la notion des jours. Elian, horloger de métier, s’était installé là cinq ans plus tôt. Il cherchait le silence, et il l’avait trouvé. Le lendemain, les horloges de la ville s’étaient toutes arrêtées à minuit. Les passants erraient, confus, parlant de la lune qui ne voulait plus partir. Elian sentit une inquiétude nouvelle. La montre noire pulsait sur son établi, comme un cœur vivant. Il décida de monter à la tour de l’Est, dont les habitants parlaient avec crainte. Là-haut, il découvrit une immense horloge fissurée, dont les engrenages semblaient faits d’ombres. Et sur le sol, il trouva un fragment d’aiguille — argentée, fine, chaude au toucher. Quand il la plaça dans la montre noire, un murmure remplit la pièce : des voix d’hommes et de femmes, de vieillards et d’enfants, tous murmurant : « Remets-nous en marche. » "

words = text.split()
vocab = sorted(list(set(words)))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# === Paramètres du réseau ===
input_size = len(vocab) * 2
output_size = len(vocab)
hidden_size = 10
learning_rate = 0.5
epochs = 100  

# === Poids et biais ===
weights_input_hidden = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
weights_hidden_output = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(output_size)]
bias_hidden = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
bias_output = [random.uniform(-0.1, 0.1) for _ in range(output_size)]

# === Fonction de précision ===
def accuracy():
    correct = 0
    for i in range(2, len(words)):
        x = one_hot(word_to_idx[words[i-2]], len(vocab)) + one_hot(word_to_idx[words[i-1]], len(vocab))
        hidden = [sigmoid(sum(x[j]*weights_input_hidden[h][j] for j in range(input_size)) + bias_hidden[h]) for h in range(hidden_size)]
        output = softmax([sum(hidden[h]*weights_hidden_output[o][h] for h in range(hidden_size)) + bias_output[o] for o in range(output_size)])
        pred_idx = output.index(max(output))
        if pred_idx == word_to_idx[words[i]]:
            correct += 1
    return correct / (len(words) - 2) * 100

# === Entraînement ===
for epoch in range(epochs):
    start_time = time.time()
    for i in range(2, len(words)):
        x = one_hot(word_to_idx[words[i-2]], len(vocab)) + one_hot(word_to_idx[words[i-1]], len(vocab))
        y_true = one_hot(word_to_idx[words[i]], output_size)

        # propagation avant
        hidden = [sigmoid(sum(x[j]*weights_input_hidden[h][j] for j in range(input_size)) + bias_hidden[h]) for h in range(hidden_size)]
        output = softmax([sum(hidden[h]*weights_hidden_output[o][h] for h in range(hidden_size)) + bias_output[o] for o in range(output_size)])

        # rétropropagation
        error = [y_true[o] - output[o] for o in range(output_size)]
        d_output = error
        d_hidden = [sigmoid_derivative(hidden[h]) * sum(d_output[o]*weights_hidden_output[o][h] for o in range(output_size)) for h in range(hidden_size)]

        # mise à jour des poids
        for o in range(output_size):
            for h in range(hidden_size):
                weights_hidden_output[o][h] += learning_rate * d_output[o] * hidden[h]
            bias_output[o] += learning_rate * d_output[o]

        for h in range(hidden_size):
            for j in range(input_size):
                weights_input_hidden[h][j] += learning_rate * d_hidden[h] * x[j]
            bias_hidden[h] += learning_rate * d_hidden[h]

    # === Mesure du temps et affichage ===
    duration = time.time() - start_time
    acc = accuracy()

    # Génération rapide d’un court texte pour visualiser les progrès
    seed_words = [words[0], words[1]]
    generated = seed_words[:]
    max_length = 30  # texte court pour affichage par epoch
    for _ in range(max_length - 2):
        x = one_hot(word_to_idx[generated[-2]], len(vocab)) + one_hot(word_to_idx[generated[-1]], len(vocab))
        hidden = [sigmoid(sum(x[j]*weights_input_hidden[h][j] for j in range(input_size)) + bias_hidden[h]) for h in range(hidden_size)]
        output = softmax([sum(hidden[h]*weights_hidden_output[o][h] for h in range(hidden_size)) + bias_output[o] for o in range(output_size)])
        next_word = idx_to_word[output.index(max(output))]
        generated.append(next_word)

    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    print(f"Temps écoulé : {duration:.2f} sec")
    print(f"Précision : {acc:.2f}%")
    print("Texte généré :", ' '.join(generated))

# === Génération finale longue ===
print("\n=== Texte final généré ===")
seed_words = [words[0], words[1]]
generated = seed_words[:]
max_length = len(words) - 2
for _ in range(max_length - 2):
    x = one_hot(word_to_idx[generated[-2]], len(vocab)) + one_hot(word_to_idx[generated[-1]], len(vocab))
    hidden = [sigmoid(sum(x[j]*weights_input_hidden[h][j] for j in range(input_size)) + bias_hidden[h]) for h in range(hidden_size)]
    output = softmax([sum(hidden[h]*weights_hidden_output[o][h] for h in range(hidden_size)) + bias_output[o] for o in range(output_size)])
    next_word = idx_to_word[output.index(max(output))]
    generated.append(next_word)

print(' '.join(generated))
