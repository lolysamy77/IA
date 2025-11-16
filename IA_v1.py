import random
import math

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

text = "Brumeville, petite cité au bord d’un lac immobile, semblait vivre dans une éternelle aube. Les rues y étaient pavées de pierres pâles, et les fenêtres reflétaient une lumière sans soleil. On disait que le vent lui-même y marchait à pas comptés, et quiconque y restait trop longtemps finissait par perdre la notion des jours. Elian, horloger de métier, s’était installé là cinq ans plus tôt. Il cherchait le silence, et il l’avait trouvé. Chaque matin, il ouvrait sa boutique « L’Aiguille d’Or » et réglait des montres qui refusaient obstinément de battre la même mesure. Les clients n’étaient pas nombreux, mais tous avaient un regard étrange, comme s’ils savaient quelque chose que lui ignorait. Certains lui demandaient de réparer des montres sans aiguilles, d’autres des sabliers sans sable. Il ne posait pas de questions : à Brumeville, on ne questionnait pas le temps. Un matin de brume particulièrement épaisse, une femme vêtue de gris entra dans sa boutique. Ses yeux avaient la couleur du fer poli. Elle tenait entre ses mains une boîte en bois noir. « Ma montre est cassée, » dit-elle simplement. Elian ouvrit la boîte : à l’intérieur se trouvait une horloge circulaire, faite d’un métal qu’il ne reconnut pas. Mais il n’y avait ni chiffres, ni aiguilles. Seulement un cœur mécanique qui battait trop lentement. « Où avez-vous trouvé cela ? » demanda-t-il, fasciné. « Dans les ruines de la tour de l’Est. On dit qu’elle appartenait au Maître du Temps. » Puis, sans un mot de plus, elle disparut dans la brume. Elian passa la nuit à examiner l’objet. Et lorsqu’il effleura le centre de la machine, un tic unique résonna dans l’air, comme si tout Brumeville avait respiré en même temps. Le lendemain, les horloges de la ville s’étaient toutes arrêtées à minuit. Les passants erraient, confus, parlant de la lune qui ne voulait plus partir. Elian sentit une inquiétude nouvelle. La montre noire pulsait sur son établi, comme un cœur vivant. Il décida de monter à la tour de l’Est, dont les habitants parlaient avec crainte. Là-haut, il découvrit une immense horloge fissurée, dont les engrenages semblaient faits d’ombres. Et sur le sol, il trouva un fragment d’aiguille — argentée, fine, chaude au toucher. Quand il la plaça dans la montre noire, un murmure remplit la pièce : des voix d’hommes et de femmes, de vieillards et d’enfants, tous murmurant : « Remets-nous en marche. » Les jours suivants, Brumeville changea. Les gens commençaient à revivre des instants passés : un boulanger répétait inlassablement la même fournée, une fillette perdait chaque matin son cerf-volant dans le même arbre, et les cloches sonnaient toujours la même heure — douze coups de midi. Elian comprit que la montre noire contrôlait le flux du temps. Mais il ne savait pas comment l’arrêter. Chaque nuit, il rêvait d’un homme vêtu de cuivre, assis sur une chaise immense, entouré d’engrenages célestes. Cet homme le regardait et disait : « Tu m’as volé mon cœur. » Un soir, Elian retourna à la tour avec la montre. Le vent s’y engouffrait comme un souffle ancien. La grande horloge battait au ralenti. Soudain, le sol trembla, et la silhouette de l’homme de ses rêves apparut. Son visage était fait de rouages et de poussière. Ses yeux brillaient comme des cadrans. « Je suis Chronos, gardien des secondes perdues. » « Je n’ai rien volé, » répondit Elian. « Je voulais seulement comprendre. » « Comprendre ? Le temps n’est pas à comprendre, il est à vivre. Et toi, tu l’as enfermé dans une montre. » Chronos tendit la main. L’air se figea, les sons moururent, et Elian sentit sa propre respiration devenir mécanique. Chronos plaça la montre noire sur la poitrine d’Elian. Le mécanisme se fondit dans sa peau, et ses veines devinrent des aiguilles fines et lumineuses. Il sentit le monde tourner autour de lui — les saisons défilèrent en une seule respiration. Quand il rouvrit les yeux, Brumeville était vide. Les maisons se dissolvaient dans la brume, et seules restaient les horloges, suspendues dans l’air. Chaque battement de son cœur faisait avancer le monde d’un souffle. Il comprit alors qu’il était devenu le nouveau Maître du Temps. Les siècles passèrent. Elian, désormais fait de métal et de mémoire, continuait d’écouter le monde battre à travers la brume. Parfois, des voyageurs arrivaient à Brumeville, cherchant l’origine du silence. Il les observait sans se montrer. Certains croyaient rêver, d’autres pensaient être morts. Mais tous, en repartant, emportaient avec eux une montre qui ne sonnait jamais la même heure. Et quand ils demandaient : « Qui fabrique ces montres ? » les habitants répondaient simplement : « L’horloger invisible. » Un jour, la montre d’Elian s’arrêta. La brume se leva pour la première fois depuis des siècles. Le lac se remit à onduler, et la lumière du vrai soleil toucha les pierres. Le temps, libéré, se remit à couler. Brumeville reprit vie, mais nul ne se souvenait d’avoir dormi si longtemps. Dans la tour de l’Est, au sommet, on trouva une horloge sans aiguilles. Et au centre, gravées en lettres d’or : « Celui qui entend le silence du temps, en devient le gardien. » Des années plus tard, un enfant entra dans « L’Aiguille d’Or ». Il trouva sur l’établi une montre noire. Il la prit dans ses mains, et elle se mit à battre. Un tic. Puis un autre. Et au loin, dans le murmure de la brume renaissante, une voix douce, presque imperceptible, dit : « Le temps recommence. »"
words = text.split()
vocab = sorted(list(set(words)))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

input_size = len(vocab)
output_size = len(vocab)
epochs = 100

def train_and_test(hidden_size, learning_rate):
    weights_input_hidden = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
    weights_hidden_output = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(output_size)]
    bias_hidden = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
    bias_output = [random.uniform(-0.1, 0.1) for _ in range(output_size)]

    def accuracy():
        correct = 0
        for i in range(len(words) - 1):
            x = one_hot(word_to_idx[words[i]], input_size)
            hidden = [sigmoid(sum(x[j]*weights_input_hidden[h][j] for j in range(input_size)) + bias_hidden[h]) for h in range(hidden_size)]
            output = softmax([sum(hidden[h]*weights_hidden_output[o][h] for h in range(hidden_size)) + bias_output[o] for o in range(output_size)])
            pred_idx = output.index(max(output))
            if pred_idx == word_to_idx[words[i+1]]:
                correct += 1
        return correct / (len(words)-1) * 100

    for epoch in range(epochs):
        for i in range(len(words) - 1):
            x = one_hot(word_to_idx[words[i]], input_size)
            y_true = one_hot(word_to_idx[words[i+1]], output_size)

            hidden = [sigmoid(sum(x[j]*weights_input_hidden[h][j] for j in range(input_size)) + bias_hidden[h]) for h in range(hidden_size)]
            output = softmax([sum(hidden[h]*weights_hidden_output[o][h] for h in range(hidden_size)) + bias_output[o] for o in range(output_size)])

            error = [y_true[o] - output[o] for o in range(output_size)]

            d_output = error
            d_hidden = [sigmoid_derivative(hidden[h]) * sum(d_output[o]*weights_hidden_output[o][h] for o in range(output_size)) for h in range(hidden_size)]

            for o in range(output_size):
                for h in range(hidden_size):
                    weights_hidden_output[o][h] += learning_rate * d_output[o] * hidden[h]
                bias_output[o] += learning_rate * d_output[o]

            for h in range(hidden_size):
                for j in range(input_size):
                    weights_input_hidden[h][j] += learning_rate * d_hidden[h] * x[j]
                bias_hidden[h] += learning_rate * d_hidden[h]

    return accuracy()

hidden_sizes = [8, 16, 24, 32]
learning_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

for h in hidden_sizes:
    for lr in learning_rates:
        print(f"\nEntraînement: hidden_size={h}, learning_rate={lr}")
        final_acc = train_and_test(h, lr)
        print(f"Précision finale après {epochs} époques : {final_acc:.2f}%")
