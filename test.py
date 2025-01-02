import asyncio
from typing import List
from src.engine import MultiModalSearchEngine, get_sample_products, add_product_embeddings

async def main():
    engine = MultiModalSearchEngine(test=True)
    # img = await client.embed_image("https://framerusercontent.com/images/qPCpslfyQpggTOGVBFtL7e3e4.png")
    # txt = await client.embed_text("""The OAuth & Permissions page in Slack app settings, showcasing the "Advanced token security via token rotation" option. Note the alert about needing a redirect URL.""")
    # print(len(img))
    # print(len(txt))
    # print(txt @ img)
    collection = "sample_products"
    created = await engine.create_collection(collection)
    print("Collection created:", created)
    
    if created:
        print("Getting sample products...")
        products = get_sample_products(500)
        
        print("Adding product embeddings...")
        products =  await add_product_embeddings(engine.embed_text, engine.embed_image, products)
        
        print("Adding products to collection...")
        result = await engine.add_products(collection, products)
    
    while True:
        query = input("Enter a query: ")
        results = await engine.search_product(collection, query)
        print("Search results:")
        for result in results:
            print(result)
            print("==="*15)
    
    
asyncio.run(main())
