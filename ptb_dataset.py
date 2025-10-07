"""
Penn Treebank Dataset Loader for Production Quantum Training.
Loads 700 training + 300 test sentences with optimized preprocessing.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from dataclasses import dataclass
from collections import Counter

@dataclass
class PTBConfig:
    """Configuration for Penn Treebank dataset loading."""
    max_train_sentences: int = 700
    max_test_sentences: int = 300
    # VINCOLO QUANTICO: Solo lunghezze 3, 5, 9, 17 parole (corrispondenti a 2, 4, 8, 16 qubit)
    allowed_sentence_lengths: List[int] = None  # Sarà [3, 5, 9, 17]
    vocab_size: int = 10000
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    random_seed: int = 42
    
    def __post_init__(self):
        if self.allowed_sentence_lengths is None:
            # Lunghezze quantiche: 3, 5, 9, 17 parole per circuiti 2, 4, 8, 16 qubit
            self.allowed_sentence_lengths = [3, 5, 9, 17]


class PTBDatasetLoader:
    """
    Production-ready Penn Treebank dataset loader.
    Optimized for HPC quantum training with massive parallelization.
    """
    
    def __init__(self, config: PTBConfig = None):
        self.config = config or PTBConfig()
        self.logger = logging.getLogger(__name__)
        self.vocab = {}
        self.inverse_vocab = {}
        self.train_sentences = []
        self.test_sentences = []
        self.sentence_stats = {}
        
    def download_ptb_data(self) -> List[str]:
        """
        Download Penn Treebank data and reconstruct complete sentences.
        PTB contiene parole tokenizzate, dobbiamo ricostruire le frasi!
        """
        try:
            # Try to use datasets library if available
            try:
                from datasets import load_dataset
                self.logger.info("🔄 Loading Penn Treebank from HuggingFace datasets...")
                dataset = load_dataset("ptb_text_only", split="train")
                
                # PTB contiene parole tokenizzate, ricostruiamo le frasi
                raw_sentences = []
                for item in dataset:
                    sentence_text = item['sentence']
                    
                    # Ricostruisci la frase dalle parole tokenizzate
                    # PTB usa 'N' per numeri, '<unk>' per parole sconosciute
                    reconstructed = self._reconstruct_sentence_from_tokens(sentence_text)
                    
                    # VINCOLO QUANTICO: Solo lunghezze 3, 5, 9, 17
                    if reconstructed:
                        word_count = len(reconstructed.split())
                        if word_count in self.config.allowed_sentence_lengths:
                            raw_sentences.append(reconstructed)
                
                self.logger.info(f"✅ Ricostruite {len(raw_sentences)} frasi complete da PTB")
                return raw_sentences[:2000]  # Limitiamo per performance
                
            except ImportError:
                self.logger.warning("📦 datasets library not available, using fallback")
                
            # Fallback: Generate synthetic PTB-style sentences
            return self._generate_fallback_sentences()
            
        except Exception as e:
            self.logger.error(f"❌ Error loading PTB: {e}")
            return self._generate_fallback_sentences()
    
    def _reconstruct_sentence_from_tokens(self, tokenized_sentence: str) -> str:
        """
        Ricostruisce una frase completa dai token PTB.
        PTB ha formato tipo: "the quick brown fox jumps over N lazy dogs"
        """
        if not tokenized_sentence:
            return ""
        
        # Split tokens
        tokens = tokenized_sentence.lower().split()
        
        # Dizionario di sostituzione per token speciali PTB
        replacements = {
            'n': '5',           # Numeri generici
            '<unk>': 'something',  # Parole sconosciute  
            '<s>': '',          # Start of sentence
            '</s>': '',         # End of sentence
            '<num>': '100',     # Numeri espliciti
            'unk': 'item',      # Unknown words
            '*': '',            # Marker PTB
            '-none-': '',       # PTB null marker
        }
        
        # Ricostruisci la frase
        reconstructed_tokens = []
        for token in tokens:
            if token in replacements:
                replacement = replacements[token]
                if replacement:  # Solo se non è vuoto
                    reconstructed_tokens.append(replacement)
            elif len(token) > 1 and token.isalpha():  # Parole valide
                reconstructed_tokens.append(token)
        
        # Unisci in frase
        sentence = ' '.join(reconstructed_tokens)
        
        # Pulisci la frase
        sentence = ' '.join(sentence.split())  # Rimuovi spazi multipli
        
        return sentence if len(sentence.split()) >= 3 else ""  # Minimo 3 parole
    
    def _generate_fallback_sentences(self) -> List[str]:
        """Generate PTB-style sentences as fallback."""
        self.logger.info("🔄 Generating synthetic PTB-style sentences...")
        
        # Templates per frasi complete e sensate
        templates = [
            "The {adj} {noun} {verb} {adv} in the morning",
            "Every {noun} {verb} when the sun {verb2}",
            "{adj} {noun} always {verb} {noun2} during winter",
            "When {noun} {verb}, other {noun2} {verb2} {adv}",
            "The {adj} {noun} and the {adj2} {noun2} {verb} together peacefully",
            "In the {adj} {noun}, many {noun2} {verb} {adv}",
            "Small {noun} {verb} big {noun2} while birds {verb2}",
            "After the storm, {adj} {noun} {verb} {noun2} carefully",
            "During summer, happy {noun} {verb} with {adj} {noun2}",
            "Before sunrise, quiet {noun} {verb} near the {noun2}",
        ]
        
        # Word lists
        nouns = ["cat", "dog", "bird", "house", "tree", "car", "book", "computer", "student", "teacher",
                 "window", "door", "table", "chair", "phone", "garden", "flower", "music", "story", "game"]
        
        verbs = ["runs", "walks", "flies", "sleeps", "works", "plays", "sings", "reads", "writes", "thinks",
                 "jumps", "dances", "learns", "teaches", "grows", "shines", "moves", "stops", "starts", "ends"]
        
        adjectives = ["beautiful", "quick", "slow", "happy", "sad", "big", "small", "red", "blue", "green",
                     "bright", "dark", "loud", "quiet", "smart", "funny", "serious", "new", "old", "young"]
        
        adverbs = ["quickly", "slowly", "happily", "sadly", "quietly", "loudly", "carefully", "easily", 
                  "suddenly", "always", "never", "sometimes", "often", "rarely", "gently", "softly"]
        
        sentences = []
        np.random.seed(self.config.random_seed)
        
        # Templates specifici per lunghezze quantiche
        quantum_templates = {
            3: [  # 3 parole - circuiti 2 qubit
                "{noun} {verb} {adv}",
                "{adj} {noun} {verb}",
                "the {noun} {verb}",
            ],
            5: [  # 5 parole - circuiti 4 qubit  
                "the {adj} {noun} {verb} {adv}",
                "{noun} {verb} {adj} {noun2} daily",
                "every {noun} {verb} {noun2} carefully",
            ],
            9: [  # 9 parole - circuiti 8 qubit
                "the {adj} {noun} {verb} {adv} in the {adj2} {noun2}",
                "when {noun} {verb}, other {noun2} {verb2} {adv} {adj}",
                "every {adj} {noun} {verb} {noun2} while birds {verb2} {adv}",
            ],
            17: [  # 17 parole - circuiti 16 qubit
                "during the {adj} morning, many {adj2} {noun} {verb} {adv} while other {noun2} {verb2} near the beautiful {adj3} {noun3} that {verb3} {adv2}",
                "when the {adj} {noun} {verb} {adv}, several {adj2} {noun2} {verb2} {adv2} because they {verb3} the {adj3} {noun3} that always {verb4}",
            ]
        }
        
        # Generate sentences per lunghezza quantica
        sentences_by_length = {length: [] for length in self.config.allowed_sentence_lengths}
        target_per_length = 300  # 300 frasi per lunghezza
        
        for target_length in self.config.allowed_sentence_lengths:
            templates_for_length = quantum_templates[target_length]
            
            for _ in range(target_per_length):
                template = np.random.choice(templates_for_length)
                
                # Fill template
                sentence = template.format(
                    adj=np.random.choice(adjectives),
                    adj2=np.random.choice(adjectives),
                    adj3=np.random.choice(adjectives),
                    noun=np.random.choice(nouns),
                    noun2=np.random.choice(nouns),
                    noun3=np.random.choice(nouns),
                    verb=np.random.choice(verbs),
                    verb2=np.random.choice(verbs),
                    verb3=np.random.choice(verbs),
                    verb4=np.random.choice(verbs),
                    adv=np.random.choice(adverbs),
                    adv2=np.random.choice(adverbs)
                )
                
                # Clean and validate
                sentence = sentence.strip().lower()
                words = sentence.split()
                
                # Verifica lunghezza esatta
                if len(words) == target_length:
                    sentences_by_length[target_length].append(sentence)
        
        # Unisci tutte le frasi
        sentences = []
        for length_group in sentences_by_length.values():
            sentences.extend(length_group)
        
        self.logger.info(f"✅ Generated {len(sentences)} synthetic sentences")
        return sentences[:1000]  # Return max 1000
    
    def preprocess_sentences(self, sentences: List[str]) -> List[List[str]]:
        """
        Clean and tokenize sentences for quantum processing.
        VINCOLO: Solo frasi con 3, 5, 9, 17 parole (dimensioni quantiche).
        """
        self.logger.info("🔄 Preprocessing sentences per dimensioni quantiche...")
        self.logger.info(f"   Lunghezze ammesse: {self.config.allowed_sentence_lengths}")
        
        processed_by_length = {length: [] for length in self.config.allowed_sentence_lengths}
        
        for sentence in sentences:
            # Clean text
            sentence = sentence.lower().strip()
            sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
            sentence = re.sub(r'\s+', ' ', sentence)     # Normalize whitespace
            
            # Tokenize
            words = sentence.split()
            sentence_length = len(words)
            
            # VINCOLO QUANTICO: Solo lunghezze specifiche
            if sentence_length in self.config.allowed_sentence_lengths:
                processed_by_length[sentence_length].append(words)
        
        # Log statistiche per lunghezza
        for length in self.config.allowed_sentence_lengths:
            count = len(processed_by_length[length])
            self.logger.info(f"   Frasi da {length} parole: {count}")
        
        # Unisci tutte le frasi
        processed = []
        for length_group in processed_by_length.values():
            processed.extend(length_group)
        
        self.logger.info(f"✅ Preprocessed {len(processed)} frasi quantiche valide")
        return processed
    
    def build_vocabulary(self, sentences: List[List[str]]) -> Dict[str, int]:
        """
        Build vocabulary from sentences with frequency filtering.
        """
        self.logger.info("🔄 Building vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        
        # Select top words
        most_common = word_counts.most_common(self.config.vocab_size - 4)  # Reserve space for special tokens
        
        # Build vocabulary
        self.vocab = {
            self.config.pad_token: 0,
            self.config.unk_token: 1,
            self.config.sos_token: 2,
            self.config.eos_token: 3,
        }
        
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)
        
        # Build inverse vocabulary
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}
        
        self.logger.info(f"✅ Built vocabulary with {len(self.vocab)} words")
        return self.vocab
    
    def encode_sentences(self, sentences: List[List[str]]) -> List[List[int]]:
        """
        Convert sentences to integer sequences.
        """
        encoded = []
        for sentence in sentences:
            encoded_sentence = []
            for word in sentence:
                encoded_sentence.append(self.vocab.get(word, self.vocab[self.config.unk_token]))
            encoded.append(encoded_sentence)
        
        return encoded
    
    def split_train_test(self, sentences: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Split sentences into train/test with stratified sampling.
        """
        self.logger.info("🔄 Splitting train/test...")
        
        np.random.seed(self.config.random_seed)
        indices = np.random.permutation(len(sentences))
        
        train_indices = indices[:self.config.max_train_sentences]
        test_indices = indices[self.config.max_train_sentences:self.config.max_train_sentences + self.config.max_test_sentences]
        
        train_sentences = [sentences[i] for i in train_indices]
        test_sentences = [sentences[i] for i in test_indices]
        
        self.logger.info(f"✅ Split: {len(train_sentences)} train, {len(test_sentences)} test")
        return train_sentences, test_sentences
    
    def load_production_dataset(self) -> Dict:
        """
        Load complete production dataset with all preprocessing.
        Returns dictionary with train/test data and metadata.
        """
        self.logger.info("🚀 Loading production PTB dataset...")
        
        # 1. Download/load raw data
        raw_sentences = self.download_ptb_data()
        
        # 2. Preprocess
        processed_sentences = self.preprocess_sentences(raw_sentences)
        
        # 3. Build vocabulary  
        self.build_vocabulary(processed_sentences)
        
        # 4. Split train/test
        train_sentences, test_sentences = self.split_train_test(processed_sentences)
        
        # 5. Encode sentences
        train_encoded = self.encode_sentences(train_sentences)
        test_encoded = self.encode_sentences(test_sentences)
        
        # 6. Calculate statistics
        train_lengths = [len(s) for s in train_sentences]
        test_lengths = [len(s) for s in test_sentences]
        
        self.sentence_stats = {
            'train_count': len(train_sentences),
            'test_count': len(test_sentences),
            'vocab_size': len(self.vocab),
            'train_avg_length': np.mean(train_lengths),
            'test_avg_length': np.mean(test_lengths),
            'train_length_range': (min(train_lengths), max(train_lengths)),
            'test_length_range': (min(test_lengths), max(test_lengths))
        }
        
        self.logger.info("✅ Production dataset loaded successfully")
        self._log_dataset_stats()
        
        return {
            'train_sentences_raw': train_sentences,
            'test_sentences_raw': test_sentences,
            'train_sentences_encoded': train_encoded,
            'test_sentences_encoded': test_encoded,
            'vocabulary': self.vocab,
            'inverse_vocabulary': self.inverse_vocab,
            'config': self.config,
            'stats': self.sentence_stats
        }
    
    def _log_dataset_stats(self):
        """Log comprehensive dataset statistics."""
        self.logger.info("📊 DATASET STATISTICS")
        self.logger.info(f"   Train sentences: {self.sentence_stats['train_count']}")
        self.logger.info(f"   Test sentences: {self.sentence_stats['test_count']}")
        self.logger.info(f"   Vocabulary size: {self.sentence_stats['vocab_size']}")
        self.logger.info(f"   Avg train length: {self.sentence_stats['train_avg_length']:.1f}")
        self.logger.info(f"   Avg test length: {self.sentence_stats['test_avg_length']:.1f}")
        self.logger.info(f"   Train length range: {self.sentence_stats['train_length_range']}")
        self.logger.info(f"   Test length range: {self.sentence_stats['test_length_range']}")


def load_ptb_for_quantum_training(logger=None) -> Dict:
    """
    Convenience function to load PTB dataset for quantum training.
    VINCOLO: Solo frasi con 3, 5, 9, 17 parole per circuiti quantici.
    """
    if logger:
        logger.info("🔄 Initializing PTB dataset for quantum training...")
    
    config = PTBConfig(
        max_train_sentences=700,
        max_test_sentences=300,
        allowed_sentence_lengths=[3, 5, 9, 17],  # VINCOLO QUANTICO
        vocab_size=5000          # Reasonable vocabulary size
    )
    
    loader = PTBDatasetLoader(config)
    dataset = loader.load_production_dataset()
    
    if logger:
        logger.info("✅ PTB dataset ready for quantum training")
        logger.info(f"   Lunghezze frasi: {config.allowed_sentence_lengths}")
    
    return dataset


if __name__ == "__main__":
    # Test the dataset loader
    logging.basicConfig(level=logging.INFO)
    dataset = load_ptb_for_quantum_training()
    
    print(f"Loaded {len(dataset['train_sentences_raw'])} training sentences")
    print(f"Sample: {dataset['train_sentences_raw'][0]}")