import pandas as pd

def load_pairs_from_file(filepath):

    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            columns = line.strip().split('\t')
            labels = columns[0]
            sentence1 = columns[1]
            sentence2 = columns[2]

            pairs.append((labels, sentence1, sentence2))
    return pairs


def baseline_exact_match(sent1, sent2):

    if sent1.strip() == sent2.strip():
        return 1
    else:
        return 0
    
def evaluate_baseline(pairs):
    correct = 0
    total = len(pairs)
    
    for label, s1, s2 in pairs:
        label = int(label)
        prediction = baseline_exact_match(s1, s2)
        if prediction == label:
            correct += 1
        # print(f"Pairs = {pairs}")
        print(f"Label = {label}")
        print(f"Prediction = {prediction}")
    accuracy = correct / total if total > 0 else 0

    print(f"Baseline Accuracy = {accuracy:.2f}")


if __name__ == "__main__":
    filepath = "synthetic_va_dataset.txt_cleaned.txt"
    pairs = load_pairs_from_file(filepath)
    evaluate_baseline(pairs)
