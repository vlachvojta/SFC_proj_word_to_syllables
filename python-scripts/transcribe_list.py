from functools import partial
import pyphen


def read_words(path):
    proccessed_words = []
    unproccessed_words = []

    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip('\r\n')
        if not line:
            continue

        if '-' in line:
            proccessed_words.append(line)
        else:
            unproccessed_words.append(line)

    return proccessed_words, unproccessed_words

def process_words(words):
    dic = pyphen.Pyphen(lang='cs_CZ')
    dic.inserted('nejkulaťoulinkatější')

    processed = []

    for word in words:
        processed.append(dic.inserted(word))

    return processed


def load_and_process():
    proccessed_words, unproccessed_words = read_words('dataset/hard_by_hand.txt')

    proccessed_words += process_words(unproccessed_words)

    for word in proccessed_words:
        print(word)


def get_interesting_words():
    process_words, _ = read_words('dataset/hard_by_hand_processed.txt')
    print(len(process_words))

    # sort by length
    process_words.sort(key=len, reverse=True)

    for word in process_words[:100]:
        print(word)


def main():
    get_interesting_words()

if __name__ == '__main__':
    main()
