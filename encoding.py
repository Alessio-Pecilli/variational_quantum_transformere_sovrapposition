import numpy as np
from gensim import downloader as api

class Encoding:
    def __init__(self, sentences, embeddingDim=16, usePretrained=True):
        self.sentences = [sentence.split() for sentence in sentences]
        self.embeddingDim = embeddingDim
        self.usePretrained = usePretrained
        self.vocabulary = self._buildVocabulary()
        self.model = self._loadModel()
        self.embeddingMatrix = self._buildEmbeddingMatrix()
        self.embeddedSentences = self._applyPositionalEncoding()
        self.stateVectors = self._normalizeEmbeddings()

    def _buildVocabulary(self):
        vocab = {}
        idx = 0
        for sentence in self.sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab

    def _loadModel(self):
        if self.usePretrained:
            return api.load("word2vec-google-news-300")
        return None

    def _buildEmbeddingMatrix(self):
        vocabSize = len(self.vocabulary)
        matrix = np.zeros((vocabSize, self.embeddingDim))
        for word, idx in self.vocabulary.items():
            if self.model and word in self.model:
                matrix[idx] = self.model[word][:self.embeddingDim]
            else:
                matrix[idx] = np.random.uniform(0, 0.1, self.embeddingDim)
        return matrix

    def _positionalEncoding(self, seqLen):
        dModel = self.embeddingDim
        position = np.arange(seqLen)[:, np.newaxis]
        divTerm = np.exp(np.arange(0, dModel, 2) * -(np.log(10000.0) / dModel))
        pe = np.zeros((seqLen, dModel))
        pe[:, 0::2] = np.sin(position * divTerm)
        pe[:, 1::2] = np.cos(position * divTerm)
        return pe

    def getPhrases(self):
        return self.sentences

    def _applyPositionalEncoding(self):
        allEncoded = []
        for sentence in self.sentences:
            embeddings = []
            for word in sentence:
                idx = self.vocabulary[word]
                embeddings.append(self.embeddingMatrix[idx])
            embeddings = np.array(embeddings)
            posEnc = self._positionalEncoding(len(sentence))
            allEncoded.append(embeddings + posEnc)
        return allEncoded

    def _normalizeEmbeddings(self):
        normalizedSentences = []
        for sentenceEmbed in self.embeddedSentences:
            normalizedSentence = []
            for vector in sentenceEmbed:
                normVector = vector / np.linalg.norm(vector)
                normalizedSentence.append(normVector)
            normalizedSentences.append(normalizedSentence)
        return normalizedSentences

    def localPsi(self, sentenceIdx, wordIdx):
        dim = self.embeddingDim
        psi = np.zeros(dim * dim)
        phrase = self.stateVectors[sentenceIdx][:wordIdx]

        for t in phrase:
            t = t / np.linalg.norm(t)
            psi += np.kron(t, t)
        return psi / np.linalg.norm(psi)
    
    def getAllPsi(self, sentenceIdx):
        """
        Get all psi vectors for a given sentence.
        
        Args:
            sentenceIdx (int): Index of the sentence
            
        Returns:
            list: List of psi vectors
        """
        print(f"Sentence index: {sentenceIdx}")
        print(f"Number of words in sentence: {len(self.stateVectors[sentenceIdx])}")
        print(f"Embedding dimension: {self.embeddingDim}")
        dim = self.embeddingDim
        phrase = self.stateVectors[sentenceIdx]
        psiList = []
    
        # From word 0 to second-to-last
        for wordIdx in range(0, len(phrase) - 1):
            psi = np.zeros(dim * dim)
            print(f"[INFO] wordIdx = {wordIdx}")
            print(f"[INFO] Phrase (embedding): {phrase}")

            # Include up to and including the wordIdx-th word
            for tIdx, t in enumerate(phrase[:wordIdx + 1]):
                print(f"[INFO] Original t (index={tIdx}): {t}")
                norm = np.linalg.norm(t)
                print(f"[INFO] Norm: {norm}")
                if norm == 0 or np.isnan(norm):
                    print(f"[WARN] Null or ill-defined vector at position {tIdx}: {t}")
                    continue
                t = t / norm
                psi += np.kron(t, t)

            psi_norm = np.linalg.norm(psi)
            if psi_norm == 0 or np.isnan(psi_norm):
                print(f"[WARN] Null or non-normalizable psi at wordIdx={wordIdx}")
                continue

            psi /= psi_norm
            psiList.append(psi)

        return psiList

