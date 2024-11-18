from chonkie import TokenChunker, WordChunker, SentenceChunker, SemanticChunker, SDPMChunker
from tokenizers import Tokenizer 


from typing import List
from ._splitter import Splitter


class TokenSplitter(Splitter):
    def __init__(self):
        tokenizer = Tokenizer.from_pretrained("gpt2")
        self.chunker = TokenChunker(tokenizer)

    def split(self, text: str) -> List[str]:
        chunks = self.chunker(text)
        return list(chunks)


class WordSplitter(Splitter):
    def __init__(self, chunk_size=512, chunk_overlap=128, mode="advanced"):
        tokenizer = Tokenizer.from_pretrained("gpt2")
        self.chunker = WordChunker(tokenizer=tokenizer, chunk_overlap=chunk_overlap,
                                   chunk_size=chunk_size)

    def split(self, text: str) -> List[str]:
        chunks = self.chunker(text)
        return list(chunks)



class SentenceSplitter(Splitter):
    def __init__(self, chunk_size=512, chunk_overlap=128, min_sentences_per_chunk=1):
        tokenizer = Tokenizer.from_pretrained("gpt2")
        self.chunker = SentenceChunker(tokenizer=tokenizer, chunk_overlap=chunk_overlap,
                                   min_sentences_per_chunk=1)

    def split(self, text: str) -> List[str]:
        chunks = self.chunker(text)
        return list(chunks)


class SemanticSplitter(Splitter):
    def __init__(self, embedding_model="all-minilm-l6-v2",
            max_chunk_size=128, similarity_threshold=0.7):
        self.chunker = SemanticChunker(embedding_model=embedding_model, 
                            max_chunk_size=max_chunk_size,
                            similarity_threshold=similarity_threshold)

    def split(self, text: str) -> List[str]:
        chunks = self.chunker(text)
        return list(chunks)


class SPDMSplitter(Splitter):
    def __init__(self, embedding_model="all-minilm-l6-v2",
            max_chunk_size=128, similarity_threshold=0., skip_window=1):
        self.chunker = SDPMChunker(embedding_model=embedding_model, 
                            max_chunk_size=max_chunk_size,
                            similarity_threshold=similarity_threshold,
                            skip_window=skip_window)

    def split(self, text: str) -> List[str]:
        chunks = self.chunker(text)
        return list(chunks)

