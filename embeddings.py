"""
Module for handling OpenAI embeddings and vector similarity.
"""

from typing import List, Dict, Tuple
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Define important categories and their embeddings
CATEGORIES = {
    "launch": ["product launch", "new release", "market entry", "customer pilot"],
    "academic": ["arxiv paper", "conference paper", "research study", "dataset release"],
    "hardware": ["actuator design", "battery technology", "materials science", "sensing system"],
    "policy": ["government policy", "regulation", "funding scheme", "industry standard"],
    "impact": ["market impact", "industry trend", "technology breakthrough", "cost reduction"]
}

# Pre-computed embeddings for common patterns
PATTERNS = {
    "funding": ["funding round", "investment", "grant", "capital raise"],
    "customer": ["customer pilot", "production order", "deployment", "field test"],
    "shipping": ["shipping", "available", "launching", "coming soon"],
    "policy": ["policy", "regulation", "incentive", "duty relief"],
    "breakthrough": ["breakthrough", "innovation", "improvement", "advance"]
}

class EmbeddingHandler:
    def __init__(self, client: OpenAI):
        self.client = client
        self.category_embeddings = {}
        self.pattern_embeddings = {}
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        if text not in self.category_embeddings:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            self.category_embeddings[text] = np.array(response.data[0].embedding)
        return self.category_embeddings[text]
    
    def _get_average_embedding(self, phrases: List[str]) -> np.ndarray:
        """Get average embedding for a list of phrases"""
        embeddings = [self._get_embedding(phrase) for phrase in phrases]
        return np.mean(embeddings, axis=0)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a given text"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def categorize_text(self, text: str) -> Dict[str, float]:
        """Categorize text based on embeddings"""
        text_embedding = self._get_embedding(text)
        scores = {}
        
        # Compute embeddings for categories on-demand
        for category, phrases in CATEGORIES.items():
            if category not in self.category_embeddings:
                self.category_embeddings[category] = self._get_average_embedding(phrases)
            
            score = cosine_similarity([text_embedding], [self.category_embeddings[category]])[0][0]
            scores[category] = score
        
        return scores
    
    def detect_patterns(self, text: str) -> Dict[str, float]:
        """Detect patterns in text using embeddings"""
        text_embedding = self._get_embedding(text)
        scores = {}
        
        # Compute embeddings for patterns on-demand
        for pattern, phrases in PATTERNS.items():
            if pattern not in self.pattern_embeddings:
                self.pattern_embeddings[pattern] = self._get_average_embedding(phrases)
            
            score = cosine_similarity([text_embedding], [self.pattern_embeddings[pattern]])[0][0]
            scores[pattern] = score
        
        return scores
    
    def generate_summary(self, text: str, category: int) -> str:
        """Generate a summary using GPT-4o-mini"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a humanoid robotics industry analyst. Generate a concise summary of the article that includes why it matters for the industry. Keep the summary to 45 words or less."},
                    {"role": "user", "content": f"Category: {category}\nArticle: {text}"}
                ],
                temperature=0.7,
                max_tokens=150
            )
            summary = response.choices[0].message.content
            # Split by words and take first 45 words
            words = summary.split()
            if len(words) > 45:
                summary = ' '.join(words[:45]) + '...'
            return summary
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Summary generation failed"
        # Construct summary based on highest scores
        summary = []
        
        if scores["launch"] > 0.7:
            summary.append("New product launch")
        if scores["academic"] > 0.7:
            summary.append("Research breakthrough")
        if scores["hardware"] > 0.7:
            summary.append("Technical advancement")
        if scores["policy"] > 0.7:
            summary.append("Policy update")
        
        return " • ".join(summary) + " _why it matters: industry impact_"
