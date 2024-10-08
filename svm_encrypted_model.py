"""
In this example Alice trains a spam classifier on some e-mails dataset she
owns. She wants to apply it to Bob's personal e-mails, without

1) asking Bob to send his e-mails anywhere
2) leaking information about the learned model or the dataset she has learned
from
3) letting Bob know which of his e-mails are spam or not.

Alice trains a spam classifier with svm on some data she
possesses. After learning, she generates public/private key pair with a
Paillier schema. The model is encrypted with the public key. The public key and
the encrypted model are sent to Bob. Bob applies the encrypted model to his own
data, obtaining encrypted scores for each e-mail. Bob sends them to Alice.
Alice decrypts them with the private key to obtain the predictions spam vs. not
spam.

Example inspired by @iamtrask blog post:
https://iamtrask.github.io/2017/06/05/homomorphic-surveillance/

Dependencies: numpy, sklearn
"""
# importing  required libraries
import time
import pickle
import os.path
import numpy as np
from tqdm import tqdm
import phe as paillier
from sklearn.svm import SVC
from zipfile import ZipFile
from urllib.request import urlopen
from contextlib import contextmanager
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
np.random.seed(42)


# Enron spam dataset hosted by https://cloudstor.aarnet.edu.au
url = [
    'https://cloudstor.aarnet.edu.au/plus/index.php/s/RpHZ57z2E3BTiSQ/download',
    'https://cloudstor.aarnet.edu.au/plus/index.php/s/QVD4Xk5Cz3UVYLp/download'
]

# defining function to download dataset.
def download_data():
    """Download two sets of Enron1 spam/ham e-mails if they are not here
    We will use the first as trainset and the second as testset.
    Return the path prefix to us to load the data from disk."""

    n_datasets = 2
    for d in range(1, n_datasets + 1):
        if not os.path.isdir('enron%d' % d):

            URL = url[d-1]
            print("Downloading %d/%d: %s" % (d, n_datasets, URL))
            folderzip = 'enron%d.zip' % d

            with urlopen(URL) as remotedata:
                with open(folderzip, 'wb') as z:
                    z.write(remotedata.read())

            with ZipFile(folderzip) as z:
                z.extractall()
            os.remove(folderzip)


#defining function to preprocess the dataset and split into train and test data.
def preprocess_data():
    """
    Get the Enron e-mails from disk.
    Represent them as bag-of-words.
    Shuffle and split train/test.
    """

    print("Importing dataset from disk...")
    path = 'enron1/ham/'
    ham1 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
            for f in os.listdir(path) if os.path.isfile(path + f)]
    path = 'enron1/spam/'
    spam1 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
             for f in os.listdir(path) if os.path.isfile(path + f)]
    path = 'enron2/ham/'
    ham2 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
            for f in os.listdir(path) if os.path.isfile(path + f)]
    path = 'enron2/spam/'
    spam2 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
             for f in os.listdir(path) if os.path.isfile(path + f)]

    # Merge and create labels
    emails = ham1 + spam1 + ham2 + spam2
    y = np.array([-1] * len(ham1) + [1] * len(spam1) +
                 [-1] * len(ham2) + [1] * len(spam2))

    # Words count, keep only frequent words
    count_vect = CountVectorizer(decode_error='replace', stop_words='english',
                                 min_df=0.001)
    X = count_vect.fit_transform(emails)

    print('Vocabulary size: %d' % X.shape[1])

    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Split train and test
    split = 500
    X_train, X_test = X[-split:, :], X[:-split, :]
    y_train, y_test = y[-split:], y[:-split]

    print("Labels in trainset are {:.2f} spam : {:.2f} ham".format(
        np.mean(y_train == 1), np.mean(y_train == -1)))

    return X_train, y_train, X_test, y_test

# timer function defined to show the runtime.
@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))

#define function save model to save the  model.
def save_model(model):
    # Save the model to a file using pickle
    with open('svm_model.pkl', 'wb') as file:
        pickle.dump(model, file)

def find_accuracy():
    # Load the model from the file
    with open('svm_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        # training data prediction and finding accuracy
        # train_predict = loaded_model.predict(X_train)
        # result_train = accuracy_score(train_predict,y_train)

        # testing data prediction and finding accuracy
        test_predict = loaded_model.predict(X_test)
        result_test = accuracy_score(test_predict, y_test)
        print("Accuracy: ",result_test)
        # confusion matrix result
        print("Confusion matrix result: ",confusion_matrix(y_test,test_predict))


class Alice:
    """
    Trains a SVM model on plaintext data,
    encrypts the model for remote use,
    decrypts encrypted scores using the paillier private key.
    """
# initialise svm.
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True)

# generating public and private key pairs.
    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = \
            paillier.generate_paillier_keypair(n_length=n_length)

# make fitting and predictions on dataset.
    def fit(self, X, y):
        self.model = self.model.fit(X, y)
        return self.model

    def predict(self, X):
        # return self.model.predict_proba(X)[:, 1]  # Return probability of positive class
        return self.model.predict(X)

#secure the weight and intercept with encryption.
    def encrypt_weights(self):
        coef = self.model.coef_.toarray()[0, :]
        encrypted_weights = [self.pubkey.encrypt(coef[i])
                             for i in tqdm(range(coef.shape[0]))]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
        return encrypted_weights, encrypted_intercept

  # getting encrypted score from Bob data and decrypts using private key.  
    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]


class Bob:
    """
    Is given the encrypted model and the public key.

    Scores local plaintext data with the encrypted model, but cannot decrypt
    the scores without the private key held by Alice.
    """

    def __init__(self, pubkey):
        self.pubkey = pubkey

   #setting weights and intercept received from Alice on his data. 
    def set_weights(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    #computes encrypted score on his data.
    def encrypted_score(self, x):
        """Compute the score of `x` by multiplying with the encrypted model,
        which is a vector of `paillier.EncryptedNumber`"""
        score = self.intercept
        _, idx = x.nonzero()
        for i in idx:
            score += x[0, i] * self.weights[i]
        return score

   #evaluating the encrypted model. 
    def encrypted_evaluate(self, X):
        return [self.encrypted_score(X[i, :]) for i in range(X.shape[0])]

# main execution.
if __name__ == '__main__':

   # data is downloaded and preprocessed. 
    download_data()
    X, y, X_test, y_test = preprocess_data()

   #Alice generating public-private keypair. 
    print("Alice: Generating paillier keypair")
    alice = Alice()
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    alice.generate_paillier_keypair(n_length=1024)

    print("Alice: Learning spam classifier")
    with timer() as t:
        model = alice.fit(X, y)
        #saving the model
    save_model(model)
    # loading the model and finding accuracy, confusion matrix
    find_accuracy()

   #Alice demonstrating accuracy of model without knowing Bob data. 
    print("Classify with model in the clear -- "
          "what Alice would get having Bob's data locally")
    with timer() as t:
        error = np.mean(alice.predict(X_test) != y_test)
    print("Error {:.3f}".format(error))

    #Alice encrypts the models weight and intercept.
    print("Alice: Encrypting classifier")
    with timer() as t:
        encrypted_weights, encrypted_intercept = alice.encrypt_weights()

#compute encrypted score on his data and evaluates.
    print("Bob: Scoring with encrypted classifier")
    bob = Bob(alice.pubkey)
    bob.set_weights(encrypted_weights, encrypted_intercept)
    with timer() as t:
        encrypted_scores = bob.encrypted_evaluate(X_test)

    
#Alice performing decryption using private key.
    print("Alice: Decrypting Bob's scores")
    with timer() as t:
        scores = alice.decrypt_scores(encrypted_scores)
    error = np.mean(np.sign(scores) != y_test)
    print("Error {:.3f} -- this is not known to Alice, who does not possess "
          "the ground truth labels".format(error))
