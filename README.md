# News Summary Bot

This bot fetches top news articles, summarizes them using OpenAI, and sends the summaries to a Slack channel.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:
```
NEWS_API_KEY=your_news_api_key
OPENAI_API_KEY=your_openai_api_key
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_CHANNEL_ID=your_slack_channel_id
```

3. Run the bot:
```bash
python news_summary_bot.py
```

## Features

- Fetches top news articles using News API
- Summarizes articles using OpenAI's GPT-3.5
- Sends summaries to a specified Slack channel
- Runs daily at 9 AM

## Note
This is a small update to trigger a new build.

## Requirements

- Python 3.8+
- News API account (https://newsapi.org/)
- OpenAI API key
- Slack workspace with bot token
