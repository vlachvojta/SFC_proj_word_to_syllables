from inference import InferenceEngine
import helpers


def main():
    debug = False
    engine = InferenceEngine()

    # Load test dataset and prepare inputs and targets
    inputs, targets = prepare_test_set('dataset/long_words_test.txt')
    if debug:
        print(f'Loaded {len(inputs)} words for testing.')
        print(f'First 10 words: {inputs[:10]}')
        print()
        print(f'Loaded {len(targets)} targets for testing.')
        print(f'First 10 targets: {targets[:10]}')
        print()

    gru_old_outs = []
    gru_new_outs = []
    baseline_outs = []
    target_outs = []

    for input_word, target in zip(inputs, targets):
        gru_old, gru_new, baseline = engine(input_word)
        if not gru_old or not gru_new or not baseline:
            continue

        gru_old_outs.append(gru_old)
        gru_new_outs.append(gru_new)
        baseline_outs.append(baseline)
        target_outs.append(target)

        if debug:
            if not gru_new == target:
                print(f'Target:   {target}')
                print(f'GRU new:  {gru_new}')
            if not baseline == target:
                print(f'Target:   {target}')
                print(f'Baseline: {baseline}')
            print()


    min_len = min(len(gru_old_outs), len(gru_new_outs), len(baseline_outs), len(target_outs))
    gru_old_outs = gru_old_outs[:min_len]
    gru_new_outs = gru_new_outs[:min_len]
    baseline_outs = baseline_outs[:min_len]
    targets = target_outs[:min_len]

    gru_old_lev = helpers.levenstein_loss(gru_old_outs, targets)
    gru_new_lev = helpers.levenstein_loss(gru_new_outs, targets)
    baseline_lev = helpers.levenstein_loss(baseline_outs, targets)

    print(f'GRU old Levenshtein loss:  {gru_old_lev:.2f} %')
    print(f'GRU new Levenshtein loss:  {gru_new_lev:.2f} %')
    print(f'Baseline Levenshtein loss: {baseline_lev:.2f} %')


def prepare_test_set(path) -> (list, list):
    """Prepare test set for inference engine."""
    targets = read_lines(path)
    inputs = [word.replace('-', '') for word in targets]
    return inputs, targets


def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip('\r\n') for line in lines]
    return lines


if __name__ == '__main__':
    main()
