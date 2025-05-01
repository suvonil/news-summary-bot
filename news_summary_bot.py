import os
import json
import schedule
import time
import requests
from datetime import datetime, timezone
from datetime import datetime, timedelta
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from openai import OpenAI
import feedparser
import arxiv
from bs4 import BeautifulSoup
from pytz import timezone as tz
from keyword_filter import KeywordFilter
from scoring import ArticleScorer
from embeddings import EmbeddingHandler
from typing import List, Dict, Tuple
import concurrent.futures

# Load environment variables
load_dotenv()

# Keyword lists for categorization
LAUNCH_KEYWORDS = [
    "launch", "release", "announcement", "new product", "introducing",
    "unveil", "reveal", "launching", "launch date", "coming soon"
]

RESEARCH_KEYWORDS = [
    "research", "study", "paper", "academic", "university",
    "universities", "institute", "laboratory", "lab", "experiment",
    "findings", "discovery", "innovation", "breakthrough"
]

HARDWARE_KEYWORDS = [
    "hardware", "mechanical", "actuator", "motor", "sensor",
    "battery", "power", "electronics", "circuit", "material",
    "design", "manufacturing", "fabrication", "component"
]

DEEP_DIVE_KEYWORDS = [
    "analysis", "deep dive", "technical", "engineering",
    "architecture", "system", "framework", "platform",
    "implementation", "performance", "benchmark"
]

MISCELLANEOUS_KEYWORDS = [
    "industry", "market", "trend", "forecast", "prediction",
    "survey", "report", "update", "news", "information"
]

class NewsSummaryBot:
    def __init__(self):
        try:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model = "gpt-4o-mini"
            self.slack_channel = os.getenv('SLACK_CHANNEL_ID')
            self.news_api_key = os.getenv('NEWS_API_KEY')
            self.slack_client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))
            self.keyword_filter = KeywordFilter()
            self.scorer = ArticleScorer()
            self.embedding_handler = EmbeddingHandler(self.client)
            print("Successfully initialized bot components")
        except Exception as e:
            print(f"Error initializing bot: {str(e)}")
            raise

    def fetch_news(self):
        """Fetch and categorize news articles from multiple sources"""
        print("Fetching news articles...")
        start_time = time.time()
        print("Starting news fetch process...")
        
        # Fetch from all sources
        try:
            # Fetch from News API
            try:
                news_api_articles = self._fetch_from_news_api()
                print(f"News API fetch completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error fetching from News API: {str(e)}")
                news_api_articles = []
            
            # Fetch from RSS feeds
            try:
                rss_articles = self._fetch_from_rss_feeds()
                print(f"RSS feeds fetch completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error fetching from RSS feeds: {str(e)}")
                rss_articles = []
            
            # Fetch from academic sources
            try:
                academic_articles = self._fetch_from_arxiv()
                print(f"Academic sources fetch completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error fetching from academic sources: {str(e)}")
                academic_articles = []
            
            # Fetch from web news sources
            try:
                web_news_articles = self._fetch_from_web_news()
                print(f"Web news sources fetch completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error fetching from web news sources: {str(e)}")
                web_news_articles = []
            
            # Fetch from company blogs
            try:
                blog_articles = self._fetch_from_company_blogs()
                print(f"Company blogs fetch completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error fetching from company blogs: {str(e)}")
                blog_articles = []
            
            # Combine all articles
            articles = news_api_articles + rss_articles + academic_articles + web_news_articles + blog_articles
            
            if not articles:
                print("No articles found. Exiting.")
                return []
        except Exception as e:
            print(f"Error in fetch_news: {str(e)}")
            return []

        # Combine all articles
        articles = news_api_articles + rss_articles + academic_articles + web_news_articles + blog_articles
        print(f"\n=== Starting Article Processing ===")
        print(f"Total articles fetched: {len(articles)}")
        
        if not articles:
            print("No articles found. Exiting.")
            return []
        
        # Filter by date (last 24 hours)
        current_time = datetime.now()
        recent_articles = []
        for article in articles:
            published_at = article.get('publishedAt', '')
            if not published_at:
                continue
            
            try:
                # Parse published date
                published_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                # Check if article is within last 24 hours
                if (current_time - published_time).total_seconds() <= 24 * 3600:
                    recent_articles.append(article)
            except ValueError:
                # If date parsing fails, skip this article
                continue
        
        print(f"Articles after date filtering: {len(recent_articles)}")
        
        # Remove duplicates
        unique_articles = []
        seen_urls = set()
        
        for article in recent_articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        print(f"Articles after deduplication: {len(unique_articles)}")
        
        # Apply keyword gating
        relevant_articles = []
        for article in unique_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            if title and description:
                text = f"{title} {description}"
                is_relevant, _ = self.keyword_filter.is_relevant(text)
                if is_relevant:
                    relevant_articles.append(article)
        
        print(f"Articles after keyword gating: {len(relevant_articles)}")
        
        # Score articles
        scored_articles = []
        for article in relevant_articles:
            try:
                score = self.scorer.score_article(article)
                article['score'] = score
                scored_articles.append(article)
            except Exception as e:
                print(f"Error scoring article '{article.get('title', '')}': {str(e)}")
                continue
        
        print(f"Articles after scoring: {len(scored_articles)}")
        
        # Categorize and summarize articles
        categorized_articles = []
        start_categorize = time.time()
        
        # Process each article
        for article in unique_articles:
            try:
                # Get article details
                title = article.get('title', 'Untitled')
                description = article.get('description', '')
                url = article.get('url', '')
                source = article.get('source', {})
                published_at = article.get('publishedAt', '')
                
                # Skip if title or description is None
                if title is None or description is None:
                    print(f"Skipping article: missing title or description")
                    continue
                
                # Generate fact sheet
                fact_sheet = self.keyword_filter.get_fact_sheet(
                    title + ' ' + description
                )
                
                # Determine category based on fact sheet
                if any(keyword in title.lower() or keyword in description.lower() for keyword in LAUNCH_KEYWORDS):
                    category = 2  # New Launches & Announcements
                elif any(keyword in title.lower() or keyword in description.lower() for keyword in RESEARCH_KEYWORDS):
                    category = 3  # AI Research
                elif any(keyword in title.lower() or keyword in description.lower() for keyword in HARDWARE_KEYWORDS):
                    category = 4  # Hardware Research
                elif any(keyword in title.lower() or keyword in description.lower() for keyword in DEEP_DIVE_KEYWORDS):
                    category = 5  # Deep Dives & Analyst Reports
                else:
                    category = 6  # Miscellaneous
                
                # Generate summary
                summary = self.embedding_handler.generate_summary(
                    f"{title} {description}",
                    category
                )
                
                # Score article
                try:
                    score = self.scorer.score_article(article, category)
                except Exception as e:
                    print(f"Error scoring article '{title}': {str(e)}")
                    score = 5.0  # Default score if scoring fails
                
                # Add the categorized article
                categorized_articles.append({
                    'article': {
                        'title': title,
                        'description': description,
                        'url': url,
                        'source': source,
                        'publishedAt': published_at
                    },
                    'bucket': category,
                    'summary': summary,
                    'score': score
                })
            except Exception as e:
                print(f"Error processing article '{title}': {str(e)}")
                continue
        
        print(f"Articles after categorization: {len(categorized_articles)}")
        print(f"Categorization completed in {time.time() - start_categorize:.2f} seconds")

        # Group articles by bucket
        buckets = {
            1: [],  # Overall Industry
            2: [],  # New Launches & Announcements
            3: [],  # AI Research
            4: [],  # Hardware Research
            5: [],  # Deep-dives & Analyst
            6: []   # Miscellaneous
        }
        
        # Add categorized articles to their respective buckets
        for article in categorized_articles:
            bucket = article['bucket']
            buckets[bucket].append(article)
            
            # Add to previous articles for novelty checking
            self.scorer.add_previous_articles([article])
        
        # Print bucket distribution
        print("\n=== Bucket Distribution ===")
        for bucket_id, articles in buckets.items():
            print(f"Bucket {bucket_id}: {len(articles)} articles")
        
        # Sort each bucket by score (highest first)
        for bucket in buckets.values():
            bucket.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Get top articles from each bucket
        top_articles = []
        for bucket in buckets.values():
            top_articles.extend(bucket[:5])  # Take top 5 from each bucket
        
        print(f"\nTop articles from each bucket: {len(top_articles)}")
        
        # Sort combined list by score
        top_articles.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 20 overall
        final_articles = top_articles[:20]
        
        # Count articles by source
        source_counts = {}
        for article in final_articles:
            source = article['article']['source']['name']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Print source distribution
        print("\n=== Source Distribution ===")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{source}: {count} articles")
        
        print(f"\nFinal selected articles: {len(final_articles)}")
        
        return final_articles

    def _fetch_from_news_api(self):
        """Fetch from News API"""
        try:
            # Get today's date in IST
            today = datetime.now().astimezone(tz('Asia/Kolkata'))
            
            url = f"https://newsapi.org/v2/everything"
            params = {
                'apiKey': self.news_api_key,
                'q': '"humanoid robot" OR "humanoid robotics" OR "bipedal robot"',
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 100,
                'from': today.strftime('%Y-%m-%d'),
                'to': today.strftime('%Y-%m-%d')
            }
            response = requests.get(url, params=params)
            articles = response.json().get('articles', [])
            print(f"Found {len(articles)} articles from News API")
            return articles
        except Exception as e:
            print(f"Error fetching from News API: {str(e)}")
            return []

    def _fetch_from_rss_feeds(self):
        """Fetch from RSS feeds"""
        print("Fetching from RSS feeds...")
        rss_feeds = [
            # Indian News Sources
            ("PIB Press Releases", "https://pib.gov.in/PressReleaseRSS.aspx?feed=PressRelease&format=rss"),
            ("Economic Times Tech", "https://economictimes.indiatimes.com/tech/technology/rssfeeds/13357544.cms"),
            ("Moneycontrol Tech", "https://www.moneycontrol.com/rss/technology.xml"),
            ("Hindu Business Line Tech", "https://www.thehindubusinessline.com/technology/feeder/default.rss"),
            ("Inc42 Robotics", "https://inc42.com/robotics/feed/"),
            ("YourStory Deep-Tech", "https://yourstory.com/section/deep-tech/feed"),
            ("The Ken", "https://the-ken.com/feed/"),
            # Robotics Trade
            ("The Robot Report", "https://therobotreport.com/feed/"),
            ("IEEE Spectrum Robotics", "https://spectrum.ieee.org/rss/topic/robotics"),
            ("RoboticsBiz", "https://roboticsbiz.com/feed/"),
            ("Asian Robotics Review", "https://asianroboticsreview.com/feed/"),
            # Global Tech / Start-up
            ("TechCrunch Robotics", "https://techcrunch.com/tag/robotics/feed/"),
            ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
            ("Bloomberg Technology", "https://www.bloomberg.com/feed/podcast/technology.xml"),
            ("WIRED Robotics", "https://www.wired.com/feed/tag/robots/rss"),
            ("Financial Times Tech", "https://www.ft.com/technology?format=rss"),
            ("Forbes Innovation", "https://www.forbes.com/innovation/feed/"),
            # Company & Vendor Blogs
            ("NVIDIA Blog", "https://blogs.nvidia.com/feed/"),
            ("Tesla Blog", "https://www.tesla.com/blog/feed"),
            ("Figure AI Blog", "https://www.figure.ai/blog/feed"),
            ("Agility Robotics Blog", "https://www.agilityrobotics.com/blog/feed"),
            ("Apptronik News", "https://www.apptronik.com/news/feed"),
            ("Sanctuary AI News", "https://sanctuary.ai/news/feed")
        ]
        
        # Fetch and process each feed sequentially
        articles = []
        for name, url in rss_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:  # Limit to 10 articles per feed
                    article = {
                        'title': str(entry.title),
                        'description': str(entry.summary) if hasattr(entry, 'summary') else "",
                        'url': str(entry.link),
                        'publishedAt': str(entry.published) if hasattr(entry, 'published') else datetime.now().isoformat(),
                        'source': {'name': str(name)},
                        'bucket': None,  # Will be set during categorization
                        'summary': None  # Will be set during categorization
                    }
                    articles.append(article)
                print(f"Found {len(feed.entries)} articles from {name}")
            except Exception as e:
                print(f"Error fetching from {url}: {str(e)}")
        
        print(f"Total RSS articles before filtering: {len(articles)}")
        
        # Filter duplicates
        seen = set()
        filtered_articles = []
        unique_articles = []
        seen_urls = set()
        
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        print(f"Total unique RSS articles after filtering: {len(unique_articles)}")
        return unique_articles

    def _fetch_from_arxiv(self):
        """Fetch from academic sources with simplified approach"""
        print("Fetching from academic sources...")
        academic_articles = []
        
        try:
            # Try ArXiv API first
            try:
                client = arxiv.Client()
                results = client.results(
                    arxiv.Search(
                        query="humanoid robot",
                        max_results=50,
                        sort_by=arxiv.SortCriterion.Relevance
                    )
                )
                for result in results:
                    try:
                        article = {
                            'title': result.title,
                            'description': result.summary,
                            'url': result.pdf_url,
                            'publishedAt': result.published.isoformat(),
                            'source': {'name': 'ArXiv'}
                        }
                        academic_articles.append(article)
                    except Exception as e:
                        print(f"Error processing ArXiv article: {str(e)}")
            except Exception as e:
                print(f"Error with ArXiv API: {str(e)}")
            
            # Try Semantic Scholar API
            try:
                scholar_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    'query': 'humanoid robot',
                    'year': 2024,
                    'limit': 50,
                    'fields': 'title,abstract,url'
                }
                
                try:
                    response = requests.get(scholar_url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    for paper in data.get('data', [])[:50]:
                        try:
                            if "humanoid" in paper['title'].lower():
                                academic_articles.append({
                                    'title': paper['title'],
                                    'description': paper.get('abstract', ''),
                                    'url': paper.get('url', ''),
                                    'publishedAt': datetime.now().isoformat(),
                                    'source': {'name': 'Semantic Scholar'}
                                })
                        except Exception as e:
                            print(f"Error processing Semantic Scholar paper: {str(e)}")
                except Exception as e:
                    print(f"Error accessing Semantic Scholar API: {str(e)}")
            except Exception as e:
                print(f"Error with Semantic Scholar search: {str(e)}")
            
            # Try direct web scraping of ArXiv
            if len(academic_articles) < 10:  # Only try if we have less than 10 papers
                try:
                    arxiv_url = "https://arxiv.org/search/"
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
                    }
                    
                    params = {
                        'query': 'humanoid robot',
                        'searchtype': 'all',
                        'abstracts': 'show',
                        'size': 50
                    }
                    
                    try:
                        response = requests.get(arxiv_url, params=params, headers=headers, timeout=10)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        papers = soup.find_all('li', class_='arxiv-result')
                        for paper in papers[:50]:
                            try:
                                title = paper.find('p', class_='title').text.strip()
                                if "humanoid" in title.lower():
                                    desc = paper.find('span', class_='abstract-full').text.strip()
                                    link = paper.find('a', class_='abs-button')['href']
                                    pdf_link = link.replace('abs', 'pdf')
                                    
                                    academic_articles.append({
                                        'title': title,
                                        'description': desc,
                                        'url': pdf_link,
                                        'publishedAt': datetime.now().isoformat(),
                                        'source': {'name': 'ArXiv'}
                                    })
                            except Exception as e:
                                print(f"Error processing web-scraped paper: {str(e)}")
                    except Exception as e:
                        print(f"Error with ArXiv web scraping: {str(e)}")
                except Exception as e:
                    print(f"Error accessing ArXiv: {str(e)}")
            
            print(f"Found {len(academic_articles)} academic articles")
            return academic_articles
        except Exception as e:
            print(f"Error fetching from academic sources: {str(e)}")
            return []

    def _fetch_from_web_news(self):
        """Fetch from Google News and Bing News to catch any missed articles"""
        print("Fetching from web news sources...")
        web_articles = []
        
        try:
            # Google News
            try:
                google_url = "https://news.google.com/search"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
                }
                
                # Try different search queries
                search_queries = [
                    "humanoid robot",
                    "humanoid robotics",
                    "bipedal robot",
                    "robotic humanoid",
                    "humanoid AI"
                ]
                
                for query in search_queries:
                    params = {
                        'q': query,
                        'hl': 'en-IN',
                        'gl': 'IN',
                        'ceid': 'IN:en'
                    }
                    
                    try:
                        response = requests.get(google_url, params=params, headers=headers, timeout=10)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find articles
                        articles = soup.find_all('article')
                        for article in articles[:50]:  # Limit to 50 articles
                            try:
                                title = article.find('h3').text.strip() if article.find('h3') else "No title"
                                desc = article.find('p').text.strip() if article.find('p') else ""
                                link = article.find('a')['href'] if article.find('a') else google_url
                                
                                # Convert Google News URL to actual URL
                                if link.startswith('/url?'):
                                    link = link.split('&url=')[1].split('&')[0]
                                
                                web_articles.append({
                                    'title': title,
                                    'description': desc,
                                    'url': link,
                                    'publishedAt': datetime.now().isoformat(),
                                    'source': {'name': 'Google News'}
                                })
                            except Exception as e:
                                print(f"Error processing Google News article: {str(e)}")
                    except Exception as e:
                        print(f"Error with Google News query '{query}': {str(e)}")
            except Exception as e:
                print(f"Error with Google News: {str(e)}")
            
            # Bing News
            try:
                bing_url = "https://www.bing.com/news/search"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
                }
                
                # Try different search queries
                for query in search_queries:
                    params = {
                        'q': query,
                        'cc': 'IN',
                        'setmkt': 'en-IN'
                    }
                    
                    try:
                        response = requests.get(bing_url, params=params, headers=headers, timeout=10)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find articles
                        articles = soup.find_all('div', class_='news-card')
                        for article in articles[:50]:  # Limit to 50 articles
                            try:
                                title = article.find('h3').text.strip() if article.find('h3') else "No title"
                                desc = article.find('p').text.strip() if article.find('p') else ""
                                link = article.find('a')['href'] if article.find('a') else bing_url
                                
                                web_articles.append({
                                    'title': title,
                                    'description': desc,
                                    'url': link,
                                    'publishedAt': datetime.now().isoformat(),
                                    'source': {'name': 'Bing News'}
                                })
                            except Exception as e:
                                print(f"Error processing Bing News article: {str(e)}")
                    except Exception as e:
                        print(f"Error with Bing News query '{query}': {str(e)}")
            except Exception as e:
                print(f"Error with Bing News: {str(e)}")
            
            print(f"Found {len(web_articles)} web news articles")
            return web_articles
        except Exception as e:
            print(f"Error fetching from web news sources: {str(e)}")
            return []

    def _fetch_from_company_blogs(self):
        """Fetch from company blogs with improved error handling"""
        company_blogs = [
            ("Tesla", "https://www.tesla.com/blog"),
            ("Tesla Engineering", "https://tesla.com/en_us/blog"),
            ("Figure AI", "https://www.figure.ai/blog"),
            ("Agility Robotics", "https://agilityrobotics.com/blog"),
            ("Sanctuary AI", "https://www.sanctuary.ai/blog"),
            ("Apptronik", "https://www.apptronik.com/blog"),
            ("UBTECH", "https://www.ubtrobot.com/blog"),
            ("Xiao Mi Robotics Lab", "https://lab.xiaomi.com/robotics"),
            ("Musashi AI", "https://musashi.ai/blog"),
            ("ABB Robotics", "https://new.abb.com/blogs/robotics"),
            ("NVIDIA", "https://blogs.nvidia.com/blog/category/ai/"),
            ("Intel Newsroom", "https://newsroom.intel.com/"),
            ("Qualcomm AI", "https://www.qualcomm.com/news/onq-blog"),
            ("AMD Instinct", "https://www.amd.com/en/graphics/instinct-blog")
        ]
        blog_articles = []
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        
        for name, url in company_blogs:
            try:
                # Add timeout and headers
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
            except (requests.RequestException, requests.Timeout, BeautifulSoupException) as e:
                print(f"Error fetching or parsing {name}'s blog: {str(e)}")
                continue
                
                # Try different selectors for articles
                articles = []
                for selector in ['article', '.post', '.blog-post', '.article', '.news-item']:
                    articles = soup.find_all(selector)
                    if articles:
                        break
                
                if not articles:
                    print(f"No articles found using standard selectors for {name}")
                    continue
                
                for article in articles[:5]:  # Limit to 5 articles per blog
                    try:
                        # Try different selectors for title
                        title = article.find(['h1', 'h2', 'h3', 'h4']).text.strip() if article.find(['h1', 'h2', 'h3', 'h4']) else "No title"
                        
                        # Try different selectors for description
                        desc = article.find(['p', 'div']).text.strip() if article.find(['p', 'div']) else ""
                        
                        # Try different selectors for link
                        link = article.find('a')['href'] if article.find('a') else url
                        if not link.startswith(('http://', 'https://')):
                            link = url.rstrip('/') + '/' + link.lstrip('/')
                        
                        blog_articles.append({
                            'title': title,
                            'description': desc,
                            'url': link,
                            'publishedAt': datetime.now().isoformat(),
                            'source': {'name': name}
                        })
                    except Exception as e:
                        print(f"Error processing article from {name}: {str(e)}")
                        continue
                
                # Combine all articles from different sources
        all_articles = []
        all_articles.extend(self._fetch_from_news_api())
        all_articles.extend(self._fetch_from_rss_feeds())
        all_articles.extend(self._fetch_from_arxiv())
        all_articles.extend(self._fetch_from_web_news())
        all_articles.extend(self._fetch_from_company_blogs())
        
        print(f"Total articles fetched: {len(all_articles)}")
        
        # Filter out duplicates
        seen = set()
        filtered_articles = []
        for article in all_articles:
            title = article['title'].lower()
            if title not in seen:
                seen.add(title)
                
                # Filter out articles older than 24 hours
                published_at = article.get('publishedAt', '')
                if published_at:
                    try:
                        # Try to parse various date formats
                        date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        try:
                            date = datetime.strptime(published_at, '%a, %d %b %Y %H:%M:%S %z')
                        except (ValueError, TypeError):
                            try:
                                date = datetime.strptime(published_at, '%Y-%m-%d')
                            except (ValueError, TypeError):
                                date = None
                else:
                    date = None

                if date:
                    age_hours = (datetime.now() - date).total_seconds() / 3600
                    if age_hours > 24:
                        print(f"Skipping article '{article['title']}': {age_hours:.1f} hours old")
                        continue
                
                filtered_articles.append(article)

        # Sort by publication date (newest first)
        filtered_articles.sort(key=lambda x: x['publishedAt'], reverse=True)

        print(f"Total unique articles after filtering: {len(filtered_articles)}")
        if filtered_articles:
            print(f"First article title: {filtered_articles[0]['title']}")

        return filtered_articles[:20]  # Return top 20 most recent articles
        
        # Build the query URL
        url = f"https://newsapi.org/v2/everything"
        params = {
            'apiKey': self.news_api_key,
            'q': '"humanoid robot" OR "humanoid robotics" OR "bipedal robot"',  # Simplified search
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 100,
            'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'to': datetime.now().strftime('%Y-%m-%d')
        }
        
        response = requests.get(url, params=params)
        articles = response.json().get('articles', [])
        print(f"Found {len(articles)} articles from API")
        
        # Filter articles to keep only the most relevant ones
        filtered_articles = []
        for article in articles:
            # Check if article contains any of our keywords in title or description
            title_match = any(keyword.lower() in article['title'].lower() for keyword in keywords)
            desc_match = any(keyword.lower() in article['description'].lower() for keyword in keywords)
            
            if title_match or desc_match:
                # Add extra weight to India-focused articles
                is_india_focus = any(word in article['title'].lower() for word in ['india', 'indian', 'bengaluru', 'mumbai', 'delhi'])
                article['is_india_focus'] = is_india_focus
                
                # Filter out articles older than 24 hours
                published_at = article.get('publishedAt', '')
                if published_at:
                    try:
                        # Try to parse various date formats
                        date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        try:
                            date = datetime.strptime(published_at, '%a, %d %b %Y %H:%M:%S %z')
                        except (ValueError, TypeError):
                            try:
                                date = datetime.strptime(published_at, '%Y-%m-%d')
                            except (ValueError, TypeError):
                                date = None
                else:
                    date = None

                if date:
                    age_hours = (datetime.now() - date).total_seconds() / 3600
                    if age_hours > 24:
                        print(f"Skipping article '{article['title']}': {age_hours:.1f} hours old")
                        continue
                
                filtered_articles.append(article)

        print(f"Filtered to {len(filtered_articles)} relevant articles")
        print(f"First article title: {filtered_articles[0]['title'] if filtered_articles else 'No articles found'}")
        
        # Sort by importance: India focus first, then recency
        filtered_articles.sort(key=lambda x: (-x.get('is_india_focus', False), x['publishedAt']), reverse=True)
        
        return filtered_articles[:10]  # Return top 10 most relevant articles  # Get top 5 articles

    def categorize_and_summarize(self, article):
        """Categorize and summarize an article using OpenAI"""
        try:
            prompt = f"""
            You are a humanoid robotics expert. Categorize this article into exactly one of these buckets:
            1. Overall Industry (market trends, funding, company news)
            2. New Launches & Announcements (product launches, company announcements)
            3. AI Research (AI/ML research, algorithms, perception)
            4. Hardware Research (actuators, batteries, materials)
            5. Deep-dives & Analyst (in-depth analysis, market reports)
            6. Miscellaneous (everything else)

            Then provide a 2-3 sentence summary focusing on its relevance to humanoid robotics.

            Article:
            Title: {article['title']}
            Content: {article['description']}
            URL: {article['url']}

            Response format:
            Bucket: [1-6]
            Summary: [summary]
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            lines = response_text.split('\n')
            bucket = None
            summary = None
            
            for line in lines:
                if line.startswith('Bucket: '):
                    try:
                        bucket = int(line.split(':')[1].strip())
                    except ValueError:
                        print(f"Invalid bucket number in response: {line}")
                        return None
                elif line.startswith('Summary: '):
                    summary = line.split(':', 1)[1].strip()
            
            if bucket and summary and 1 <= bucket <= 6:
                return {
                    'bucket': bucket,
                    'summary': summary,
                    'article': article
                }
            else:
                print(f"Invalid bucket number or missing summary for {article['title']}")
                return None
        except Exception as e:
            print(f"Error processing article: {str(e)}")
            return None

    def format_slack_message(self, articles):
        """Format the news digest with categorized articles"""
        # Create sections for each bucket
        sections = []
        
        # Add dateline header
        date_str = datetime.now(tz('Asia/Kolkata')).strftime('%Y-%m-%d')
        header = f":calendar: *{date_str} Humanoid Robotics Daily*"
        sections.append(header)
        
        # Category headers with emojis
        category_headers = {
            1: (":newspaper:", "Overall Industry News"),
            2: (":rocket:", "New Launches & Announcements"),
            3: (":brain:", "AI Research"),
            4: (":wrench:", "Hardware Research"),
            5: (":bar_chart:", "Deep Dives & Analyst Reports"),
            6: (":sparkles:", "Miscellaneous")
        }
        
        # Add articles for each bucket
        for bucket_id in range(1, 7):
            bucket_articles = [a for a in articles if a.get('bucket') == bucket_id]
            
            # Add category header
            emoji, category_name = category_headers[bucket_id]
            sections.append(f"{emoji} *{category_name}*")
            
            if not bucket_articles:
                sections.append("_None today_")
                continue
            
            # Add articles with formatted summaries
            for article in bucket_articles:
                title = article.get('article', {}).get('title', 'Untitled')
                summary = article.get('summary', '')
                url = article.get('article', {}).get('url', '')
                published_at = article.get('article', {}).get('publishedAt', '')
                
                # Format the summary to include why it matters
                why_it_matters = self._extract_why_it_matters(summary)
                summary = summary.replace(why_it_matters, '').strip()
                
                # Format the date if available
                date_str = ""
                if published_at:
                    try:
                        # Try to parse various date formats
                        date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        date_str = date.strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        try:
                            date = datetime.strptime(published_at, '%a, %d %b %Y %H:%M:%S %z')
                            date_str = date.strftime('%Y-%m-%d')
                        except (ValueError, TypeError):
                            date_str = ""
                
                sections.append(f"• *<{url}|{title}>* ({date_str}) — {summary} _{why_it_matters}_")
        
        # Generate observations using GPT-4o
        observations = self._generate_daily_writeup(articles)
        if observations:
            sections.append("\n:mag: *Key Observations*\n" + observations)
        
        # Join all sections with newlines
        message = "\n\n".join(sections)
        
        # Check message length and split if needed
        if len(message) > 12000:
            split_index = message.find('\n\n', 6000)
            if split_index != -1:
                return [
                    message[:split_index],
                    "\n\n... (continued from previous message)\n\n" + message[split_index+2:]
                ]
        
        # Ensure we have at least 20 articles
        if len(sections) < 20:
            print(f"Warning: Only {len(sections)} articles found. Adding more articles...")
            # Add more articles from each bucket until we reach 20
            for bucket_id in range(1, 7):
                bucket_articles = [a for a in articles if a.get('bucket') == bucket_id]
                if bucket_articles:
                    for article in bucket_articles:
                        if len(sections) >= 20:
                            break
                        title = article.get('article', {}).get('title', 'Untitled')
                        summary = article.get('summary', '')
                        url = article.get('article', {}).get('url', '')
                        published_at = article.get('article', {}).get('publishedAt', '')
                        
                        # Format the summary to include why it matters
                        why_it_matters = self._extract_why_it_matters(summary)
                        summary = summary.replace(why_it_matters, '').strip()
                        
                        # Format the date if available
                        date_str = ""
                        if published_at:
                            try:
                                # Try to parse various date formats
                                date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                                date_str = date.strftime('%Y-%m-%d')
                            except (ValueError, TypeError):
                                try:
                                    date = datetime.strptime(published_at, '%a, %d %b %Y %H:%M:%S %z')
                                    date_str = date.strftime('%Y-%m-%d')
                                except (ValueError, TypeError):
                                    date_str = ""
                        
                        sections.append(f"• *<{url}|{title}>* ({date_str}) — {summary} _{why_it_matters}_")
        
        return message
    
    def _extract_why_it_matters(self, summary: str) -> str:
        """Extract the 'why it matters' clause from the summary"""
        why_it_matters = ""
        if "why it matters" in summary.lower():
            start_idx = summary.lower().find("why it matters")
            why_it_matters = summary[start_idx:].strip()
        return why_it_matters
    
    def _generate_daily_writeup(self, articles: List[Dict]) -> str:
        """Generate daily write-up using GPT-4"""
        try:
            # Prepare context
            context = []
            for article in articles:
                category = article.get('bucket', 6)
                category_name = {
                    1: "Overall Industry",
                    2: "New Launches",
                    3: "AI Research",
                    4: "Hardware Research",
                    5: "Deep Dives",
                    6: "Miscellaneous"
                }.get(category, "Unknown")
                
                context.append({
                    'title': article.get('article', {}).get('title', ''),
                    'category': category_name,
                    'summary': article.get('summary', '')
                })
            
            # Generate write-up using GPT-4o-mini
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a humanoid robotics industry analyst. Generate a concise daily write-up that synthesizes the key trends and insights from today's news articles. Keep the write-up to 50 words or less."},
                        {"role": "user", "content": f"Today's articles:\n{json.dumps(context, indent=2)}"}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                writeup = response.choices[0].message.content
                
                # Split by words and take first 50 words
                words = writeup.split()
                if len(words) > 50:
                    writeup = ' '.join(words[:50]) + '...'
                
                return writeup
            except Exception as e:
                print(f"Error generating write-up: {str(e)}")
                return "No write-up available. Please check the bot logs for details."
        except Exception as e:
            print(f"Error generating daily writeup: {str(e)}")
            return "No write-up available. Please check the bot logs for details."

    def send_to_slack(self, articles):
        """Send formatted news digest to Slack"""
        try:
            message = self.format_slack_message(articles)
            
            # Split message if it's too long
            if isinstance(message, list):
                # First message
                self.slack_client.chat_postMessage(
                    channel=self.slack_channel,
                    text="Humanoid Robotics News Summary:\n\n" + message[0]
                )
                
                # Subsequent messages
                for part in message[1:]:
                    self.slack_client.chat_postMessage(
                        channel=self.slack_channel,
                        text=part
                    )
            else:
                self.slack_client.chat_postMessage(
                    channel=self.slack_channel,
                    text="Humanoid Robotics News Summary:\n\n" + message
                )
        except SlackApiError as e:
            print(f"Error sending to Slack: {e.response['error']}")

    def run(self):
        """Main function to fetch, summarize and send news"""
        try:
            import time
            start_time = time.time()
            
            print("Starting news fetch process...")
            articles = self.fetch_news()
            fetch_time = time.time() - start_time
            print(f"Fetch completed in {fetch_time:.2f} seconds")
            print(f"Found {len(articles)} articles")
            
            if not articles:
                print("No articles found. Exiting.")
                return
                
            print("Formatting news digest...")
            message = self.format_slack_message(articles)
            format_time = time.time() - start_time - fetch_time
            print(f"Formatting completed in {format_time:.2f} seconds")
            
            print("Sending to Slack...")
            self.send_to_slack(articles)
            total_time = time.time() - start_time
            print(f"Total process completed in {total_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error in run(): {str(e)}")
            raise

    def schedule_job(self):
        """Schedule the job to run daily at 9 AM IST"""
        # Schedule for 9 AM IST
        schedule.every().day.at("09:00", timezone('Asia/Kolkata')).do(self.run)
        
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    bot = NewsSummaryBot()
    bot.run()  # Run once for testing
