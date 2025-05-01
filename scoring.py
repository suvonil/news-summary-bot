"""
Scoring system for news articles based on impact and relevance.
"""

from typing import Dict, List, Tuple
import re
from datetime import datetime, timedelta
from collections import defaultdict

# Source credibility tiers
TIER_1_SOURCES = {
    "bloomberg", "financial times", "ft", "reuters", "ieee", "nature", "science",
    "mckinsey", "gartner"
}

RECOGNIZED_TRADE_SOURCES = {
    "the robot report", "nvidia blog", "figure ai blog", "agility robotics blog",
    "apptronik news", "sanctuary ai news"
}

# Event impact patterns
FUNDING_PATTERN = re.compile(r'\b(?:funding|investment|round|grant)\b.*?\b(?:10\s*million|80\s*crore)\b', re.IGNORECASE)
CUSTOMER_PILOT_PATTERN = re.compile(r'\b(?:pilot|production|order|deployment)\b.*?\b(?:humanoid|robot)\b', re.IGNORECASE)
SHIPPING_PATTERN = re.compile(r'\b(?:shipping|available|launching)\b.*?\b(?:within\s*12\s*months|coming\s*soon)\b', re.IGNORECASE)
POLICY_PATTERN = re.compile(r'\b(?:policy|regulation|incentive|duty relief|procurement)\b', re.IGNORECASE)
TECH_BREAKTHROUGH_PATTERN = re.compile(r'\b(?:reduce|reduce|improve|increase)\b.*?\b(?:20%|15%)\b', re.IGNORECASE)


import os
from openai import OpenAI

class ArticleScorer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.previous_articles = defaultdict(list)  # Track articles from previous digests
        self.max_history_days = 7
        
    def add_previous_articles(self, articles: List[Dict]):
        """Add articles to the history for novelty checking"""
        for article in articles:
            source = article.get('source', {}).get('name', '').lower()
            title = article.get('title', '').lower()
            self.previous_articles[source].append((title, datetime.now()))
            
            # Clean up old articles
            cutoff = datetime.now() - timedelta(days=self.max_history_days)
            self.previous_articles[source] = [
                (t, d) for t, d in self.previous_articles[source]
                if d >= cutoff
            ]
    
    def score_article(self, article: Dict, category: int) -> float:
        """Score an article based on its relevance and quality"""
        try:
            # Generate a summary using GPT-4o-mini
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a humanoid robotics industry analyst. Score this article on a scale of 0-10 based on its relevance, importance, and quality. Consider the category when scoring. Format your response as 'Score: X' where X is a number between 0 and 10."},
                    {"role": "user", "content": f"Category: {category}\nArticle: {article.get('title', 'Untitled')}\nDescription: {article.get('description', '')}"}
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            # Extract score from response
            response_text = response.choices[0].message.content
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response_text)
            
            if score_match:
                score = float(score_match.group(1))
                if 0 <= score <= 10:
                    return score
                else:
                    print(f"Invalid score range: {score}")
                    return 5.0
            else:
                print(f"Could not extract score from response: {response_text}")
                return 5.0
        except Exception as e:
            print(f"Error scoring article: {str(e)}")
            return 5.0  # Default score if scoring fails
        
        # Get the article text
        title = article.get('article', {}).get('title', '')
        description = article.get('article', {}).get('description', '')
        text = f"{title} {description}"
        
        # Check for significant impacts
        if FUNDING_PATTERN.search(text):
            score += 3
        if CUSTOMER_PILOT_PATTERN.search(text):
            score += 2
        if SHIPPING_PATTERN.search(text):
            score += 2
        if POLICY_PATTERN.search(text):
            score += 2
        if TECH_BREAKTHROUGH_PATTERN.search(text):
            score += 3
            
        # Source credibility bonus
        source = article.get('article', {}).get('source', {}).get('name', '').lower()
        if source in TIER_1_SOURCES:
            score += 2
        elif source in RECOGNIZED_TRADE_SOURCES:
            score += 1
            
        # Novelty bonus
        title = article.get('article', {}).get('title', '').lower()
        source = article.get('article', {}).get('source', {}).get('name', '').lower()
        is_novel = True
        
        # Check if similar article appeared in last 7 days
        for prev_title, _ in self.previous_articles[source]:
            if self._are_similar_titles(title, prev_title):
                is_novel = False
                break
        
        if is_novel:
            score += 1
        
        return score
    
    def _are_similar_titles(self, title1: str, title2: str) -> bool:
        """Check if two titles are similar enough to be considered the same story"""
        # Simple similarity check based on common words
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        common_words = words1.intersection(words2)
        
        # Consider similar if more than 50% of words match
        return len(common_words) > 0.5 * min(len(words1), len(words2))
    
    def get_top_articles(self, articles: List[Dict], category: int, limit: int = 20) -> List[Dict]:
        """Get the top N articles based on score"""
        # Score all articles
        scored_articles = []
        for article in articles:
            score = self.score_article(article, category)
            scored_articles.append((article, score))
        
        # Sort by score (descending)
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N articles, handling ties
        top_articles = []
        current_score = None
        
        for article, score in scored_articles:
            if len(top_articles) >= limit and score < current_score:
                break
            
            top_articles.append(article)
            current_score = score
        
        return top_articles
