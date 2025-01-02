import asyncio, os
from uuid import uuid4
from datasets import load_dataset
from transformers import AutoModel
from pydantic import BaseModel, Field
from qdrant_client.http import models
from qdrant_client import AsyncQdrantClient
from typing import Union, List, Optional, Callable


class Product(BaseModel):
    id_:             str = Field(default_factory=lambda: str(uuid4()))
    metadata:        dict = Field(default_factory=dict)    
    text_embedding:  Optional[List[float]] = Field(default=None)
    image_embedding: Optional[List[float]] = Field(default=None)
    


class MultiModalSearchEngine:
    def __init__(self, model_name: str='jinaai/jina-clip-v1', cache_dir: str="./cache", batch_size: int=10, img_dim: int=768, txt_dim: int=768, url: str=None, test: bool=False, **kwargs) -> None:
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=cache_dir, 
            trust_remote_code=True
        )
        self.client = (
            AsyncQdrantClient(url=url, **kwargs)
            if not test else 
            AsyncQdrantClient(path="./qdrant", **kwargs) 
        )
        
    async def embed_text(self, text: str|List[str]) -> List[float]|List[List[float]]:
        if not text:
            raise ValueError("Text cannot be empty")
            
        texts = [text] if isinstance(text, str) else text
        if any(not t for t in texts):
            raise ValueError("All text entries must be non-empty")
            
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        embeddings = []
        for batch in batches:
            batch_embeddings = await asyncio.to_thread(self.model.encode_text, batch)
            if batch_embeddings is None:
                raise ValueError("Model failed to generate text embeddings")
            embeddings.extend(batch_embeddings.tolist())
        return embeddings[0] if isinstance(text, str) else embeddings

    async def embed_image(self, image_path: str|List[str]) -> List[float]|List[List[float]]:
        if not image_path:
            raise ValueError("Image path cannot be empty")
            
        image_paths = [image_path] if isinstance(image_path, str) else image_path
        if any(not os.path.exists(p) for p in image_paths):
            raise ValueError("All image paths must exist")
            
        batches = [image_paths[i:i + self.batch_size] for i in range(0, len(image_paths), self.batch_size)]
        embeddings = []
        for batch in batches:
            batch_embeddings = await asyncio.to_thread(self.model.encode_image, batch)
            if batch_embeddings is None:
                raise ValueError("Model failed to generate image embeddings")
            embeddings.extend(batch_embeddings.tolist())
        return embeddings[0] if isinstance(image_path, str) else embeddings
            
    async def create_collection(self, collection_name: str) -> bool:
        if not await self.client.collection_exists(collection_name):
            return await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text_vector": models.VectorParams(
                        size=self.txt_dim, 
                        distance=models.Distance.COSINE
                    ),
                    "image_vector": models.VectorParams(
                        size=self.img_dim, 
                        distance=models.Distance.COSINE
                    )
                }
            )
        return False
    
    async def add_products(self, collection: str, products: List[Product]) -> models.UpdateResult:
        points_to_upsert = []
        for product in products:
            points_to_upsert.append(models.PointStruct(
                id=product.id_,
                payload=product.metadata,
                vector={
                    "text_vector": product.text_embedding,
                    "image_vector": product.image_embedding
                },
            ))
        return await self.client.upsert(
            points=points_to_upsert,
            collection_name=collection,
        )
  
    async def search_product(self, collection: str, query: str, top_k: int=5, query_is_image: bool=False, filter: Optional[models.Filter]=None, score_threshold: Optional[float]=None) -> List[models.ScoredPoint]:
        query_vector = await self.embed_image(query) if query_is_image else await self.embed_text(query)
        prefetch = [
            models.Prefetch(
                query=query_vector,
                using="image_vector", limit=3,
            ),
        ]
        results = await self.client.query_points(
            using="text_vector",
            prefetch=prefetch,
            collection_name=collection,
            query=query_vector, limit=top_k,
            query_filter=filter, score_threshold=score_threshold,
        )
        return results.points
   
    async def delete_collection(self, collection:str) -> bool:
        try:
            return await self.client.delete_collection(collection)
        except:
            return False
        
    async def delete_points(self, collection: str, ids: Union[str, List[str]]) -> bool:
        return await self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(
                points=[ids] if isinstance(ids, str) else ids,
            )
        )
        
    async def delete_points_by_filter(self, collection: str, filter: models.Filter) -> bool:
        return await self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=filter
            )
        )



def get_sample_products(num_records: int=1000) -> List[Product]:
    images_dir = os.path.join("cache", "product_images")
    os.makedirs(images_dir, exist_ok=True)
    
    dataset = load_dataset(
        "rajuptvs/ecommerce_products_clip",
        cache_dir="./cache",
        split=f"train[:{num_records}]"
    )
    
    product_list = []
    for idx, record in enumerate(dataset):
        # Generate a unique filename for the image
        image_filename = f"product_{idx}_{uuid4()}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        
        # Save the PIL image to disk
        pil_image = record["image"]
        pil_image.save(image_path)
        
        description = record["Description"] if record["Description"] != "unknown" else record["Clipinfo"]
        data = [
            f"Product name: {record['Product_name']}", 
            f"Colors: {record["colors"]}", 
            f"Pattern: {record["Pattern"]}", 
            f"Price: {record["Price"]}"
            f"Description: {description}"
        ]
        product = Product(
            id_=str(uuid4()),
            metadata={
                "image_url": image_path,
                "price": record["Price"],
                "colors": record["colors"],
                "pattern": record["Pattern"],
                "clipinfo": record["Clipinfo"],
                "product_name": record["Product_name"],
                "description": " ".join(data)
            }
        )
        product_list.append(product)
    return product_list


async def add_product_embeddings(text_embed_func: Callable, image_embed_func: Callable, product_list: List[Product]) -> List[Product]:
    if not product_list:
        raise ValueError("Product list cannot be empty")
        
    # Validate all products have required metadata
    for product in product_list:
        if not product.metadata.get("description"):
            raise ValueError(f"Product {product.id_} missing description")
        if not product.metadata.get("image_url"):
            raise ValueError(f"Product {product.id_} missing image_url")
        if not os.path.exists(product.metadata["image_url"]):
            raise ValueError(f"Product {product.id_} has invalid image path: {product.metadata['image_url']}")

    text_embeddings = await text_embed_func([product.metadata["description"] for product in product_list])
    image_embeddings = await image_embed_func([product.metadata["image_url"] for product in product_list])
    
    if len(text_embeddings) != len(product_list) or len(image_embeddings) != len(product_list):
        raise ValueError("Number of embeddings does not match number of products")

    for product, text_embedding, image_embedding in zip(product_list, text_embeddings, image_embeddings):
        product.text_embedding = text_embedding
        product.image_embedding = image_embedding
    return product_list