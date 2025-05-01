from fastapi import FastAPI
from pydantic import BaseModel
import requests
from typing import List, Optional

app = FastAPI(title="Reddit API Wrapper")

class Post(BaseModel):
    title: str
    score: int
    url: str

@app.get("/")
async def root():
    return {"message": "Welcome to Reddit API Wrapper"}

@app.get("/posts/{subreddit}", response_model=List[Post])
async def get_posts(subreddit: str, limit: int = 10):
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}"
    headers = {"User-Agent": "Reddit API Wrapper/1.0"}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    posts = []
    for post in data["data"]["children"]:
        post_data = post["data"]
        posts.append(Post(
            title=post_data["title"],
            score=post_data["score"],
            url=post_data["url"]
        ))
    
    return posts