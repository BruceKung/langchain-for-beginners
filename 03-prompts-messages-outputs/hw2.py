import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
import json
from typing import Literal
from pydantic import BaseModel, Field

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

def main():

    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=KEY,
        base_url=URL,
    )

    class Product(BaseModel):
        name: str = Field(description="Product name")
        price: float = Field(description="Price in USD")
        category: Literal["Electronics", "Clothing", "Food", "Books", "Home"] = Field(description="Product category")
        in_stock: bool = Field(description="Whether the product is currently available")
        rating: float = Field(description="Customer rating from 1-5 stars")
        features: list[str] = Field(description="List of key product features or highlights")
    
    structured_model = model.with_structured_output(Product)

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """Extract product information from the description.
If a field is not explicitly mentioned, make a reasonable inference.
Ensure the category is one of: Electronics, Clothing, Food, Books, or Home.""",
        ),
        ("human", "{input}"),
    ])

    chain = template | structured_model

    test_descriptions = [
        "MacBook Pro 16-inch with M3 chip, $2,499. Currently in stock. Users rate it 4.8/5. Features: Liquid Retina display, 18-hour battery, 1TB SSD",
        "Cozy wool sweater, blue color, medium size. $89, available now! Customers love it - 4.5 stars. Hand-washable, made in Ireland",
        "The Great Gatsby by F. Scott Fitzgerald. Classic novel, paperback edition for $12.99. In stock. Rated 4.9 stars. 180 pages, published 1925",
        "Modern LED desk lamp with adjustable brightness. $45.99. Available for immediate shipping. 4.6 star rating. USB charging, touch controls, energy efficient",
        "Organic dark chocolate bar, 85% cacao. $5.99 each. In stock! Rated 4.7 stars by health-conscious buyers. Fair trade, vegan, no added sugar",
    ]

    for description in test_descriptions:
        try:
            # 调用 + 内置验证
            product = chain.invoke({"input": description})

            # 额外手动验证
            Product.model_validate(product)

            print("✅ 产品信息提取成功：")
            print(f"名称：{product.name}")
            print(f"价格：{product.price}")
            print(f"分类：{product.category}")
            print(f"库存：{product.in_stock}")
            print(f"评分：{product.rating}")
            print(f"特点：{product.features}")
            print("-" * 70)

        except Exception as e:
            print(f"❌ 验证失败：{e}")
    

if __name__ == "__main__":
    main()