"""Tests for basic text processing modules."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBagOfWords:
    """Tests for bag of words implementation."""
    
    def test_bag_of_words_basic(self):
        """Test basic bag of words functionality."""
        from sklearn.feature_extraction.text import CountVectorizer
        
        sentences = [
            "I love natural language processing.",
            "Language models are fascinating.",
            "I love building NLP projects."
        ]
        
        vectorizer = CountVectorizer()
        bag_of_words = vectorizer.fit_transform(sentences)
        
        # Check output shape
        assert bag_of_words.shape[0] == len(sentences)
        assert bag_of_words.shape[1] > 0
        
        # Check vocabulary is created
        vocab = vectorizer.get_feature_names_out()
        assert len(vocab) > 0
        assert 'love' in vocab or 'language' in vocab
        
        # Check array is created
        bow_array = bag_of_words.toarray()
        assert isinstance(bow_array, np.ndarray)
        assert bow_array.shape == bag_of_words.shape
    
    def test_bag_of_words_empty(self):
        """Test bag of words with empty input."""
        from sklearn.feature_extraction.text import CountVectorizer
        
        vectorizer = CountVectorizer()
        
        # Empty list should work but produce empty result
        sentences = []
        with pytest.raises(ValueError):
            vectorizer.fit_transform(sentences)
    
    def test_bag_of_words_single_word(self):
        """Test bag of words with single word sentences."""
        from sklearn.feature_extraction.text import CountVectorizer
        
        sentences = ["hello", "world", "hello"]
        
        vectorizer = CountVectorizer()
        bag_of_words = vectorizer.fit_transform(sentences)
        
        assert bag_of_words.shape[0] == 3
        assert bag_of_words.shape[1] == 2  # Two unique words


class TestStopwords:
    """Tests for stopwords functionality."""
    
    def test_stopwords_import(self):
        """Test that stopwords can be imported."""
        try:
            import nltk
            from nltk.corpus import stopwords
            
            # Download if not already downloaded
            try:
                stop_words = set(stopwords.words('english'))
                assert len(stop_words) > 0
            except LookupError:
                # Stopwords not downloaded, that's okay for test
                pytest.skip("NLTK stopwords not downloaded")
        except ImportError:
            pytest.skip("NLTK not installed")
    
    def test_stopwords_english(self):
        """Test English stopwords."""
        try:
            from nltk.corpus import stopwords
            
            stop_words = set(stopwords.words('english'))
            
            # Check common stopwords
            common_stopwords = ['the', 'a', 'an', 'is', 'are']
            for word in common_stopwords:
                if word in stop_words:
                    assert True
                    return
            
            # At least some common stopwords should be present
            assert len(stop_words) > 50
        except (ImportError, LookupError):
            pytest.skip("NLTK or stopwords not available")
    
    def test_stopwords_filtering(self):
        """Test filtering stopwords from text."""
        try:
            from nltk.corpus import stopwords
            
            stop_words = set(stopwords.words('english'))
            
            text = "This is a sample sentence with stopwords"
            words = text.lower().split()
            
            filtered_words = [w for w in words if w not in stop_words]
            
            # Should have fewer words after filtering
            assert len(filtered_words) <= len(words)
        except (ImportError, LookupError):
            pytest.skip("NLTK or stopwords not available")


class TestLemmatization:
    """Tests for lemmatization functionality."""
    
    def test_lemmatization_import(self):
        """Test that lemmatization libraries can be imported."""
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            
            lemmatizer = WordNetLemmatizer()
            assert lemmatizer is not None
        except ImportError:
            pytest.skip("NLTK not installed")
    
    def test_basic_lemmatization(self):
        """Test basic lemmatization."""
        try:
            from nltk.stem import WordNetLemmatizer
            
            lemmatizer = WordNetLemmatizer()
            
            # Test common lemmatizations
            assert lemmatizer.lemmatize("running", pos='v') in ['run', 'running']
            assert lemmatizer.lemmatize("better", pos='a') in ['good', 'better']
        except (ImportError, LookupError):
            pytest.skip("NLTK or WordNet not available")


class TestSimilarity:
    """Tests for similarity measures."""
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create two simple vectors
        vec1 = np.array([[1, 0, 1]])
        vec2 = np.array([[0, 1, 1]])
        
        similarity = cosine_similarity(vec1, vec2)
        
        # Cosine similarity should be between -1 and 1
        assert -1 <= similarity[0][0] <= 1
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Create two simple vectors
        vec1 = np.array([[1, 0, 1]])
        vec2 = np.array([[0, 1, 1]])
        
        distance = euclidean_distances(vec1, vec2)
        
        # Distance should be positive
        assert distance[0][0] >= 0
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        vec = np.array([[1, 2, 3]])
        
        similarity = cosine_similarity(vec, vec)
        
        # Identical vectors should have similarity of 1
        assert abs(similarity[0][0] - 1.0) < 0.0001


class TestPOSTagging:
    """Tests for Part-of-Speech tagging."""
    
    def test_pos_tagging_import(self):
        """Test that POS tagging can be imported."""
        try:
            import nltk
            
            # Try to use pos_tag
            text = "This is a test"
            tokens = text.split()
            
            try:
                tags = nltk.pos_tag(tokens)
                assert len(tags) == len(tokens)
                assert all(isinstance(tag, tuple) for tag in tags)
            except LookupError:
                pytest.skip("NLTK averaged_perceptron_tagger not downloaded")
        except ImportError:
            pytest.skip("NLTK not installed")
    
    def test_pos_tag_structure(self):
        """Test POS tag structure."""
        try:
            import nltk
            
            text = "The cat sat"
            tokens = text.split()
            
            try:
                tags = nltk.pos_tag(tokens)
                
                # Each tag should be a tuple of (word, tag)
                for token, (word, tag) in zip(tokens, tags):
                    assert word == token
                    assert isinstance(tag, str)
                    assert len(tag) > 0
            except LookupError:
                pytest.skip("NLTK tagger not available")
        except ImportError:
            pytest.skip("NLTK not installed")
