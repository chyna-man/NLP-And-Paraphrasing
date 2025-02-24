import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def load_msr_paraphase_data(filepath):
    x = []
    y = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  # Skip the header
        
        for row in reader:
            if len(row) < 3:  # Ensure row has enough columns
                continue
            label = row[0]
            sentence1 = row[1]  # Adjusted column index
            sentence2 = row[2]  # Adjusted column index

            concatenated = sentence1 + ' [SEP] ' + sentence2

            x.append(concatenated)
            y.append(int(label))
    
    return x, y

def main():
    #1. Load the data
    filepath = "synthetic_va_dataset.txt_cleaned.txt"  # Adjust if needed
    x, y = load_msr_paraphase_data(filepath)

    #2. Split the intro train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify = y)

    #3. Vectorize the text (BoW with TF-IDF)
    vectorizer = TfidfVectorizer(
        lowercase = True,
        # stop_words="english",
        ngram_range=(1,2)
    )

    #4. Transform data
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    #5. Train the SVM model
    svm_clf = SVC(
        C=1.0,
        kernel = "linear",
        random_state = 42
    )
    svm_clf.fit(x_train_tfidf, y_train)

    #6. Evaluate the model
    y_pred = svm_clf.predict(x_test_tfidf)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # filepath = "D:/Projects/AI paraphrase datasets/Data/msr_paraphrase_train_for_testing.txt"  # Adjust if needed
    main()