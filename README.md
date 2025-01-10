# ecomseek

A sample logic implementation for product search with multi-modal support which allows to search products with image as well as text.

1. Install astral UV `pip install uv`
2. Install dependencies: `uv sync`
3. Test run the code: `uv run python test.py`

It will download a sample dataset from huggingface having ecommerce product data with product and metadata. Creates an on-disk qdrant database and creates a sample collection with `sample_products` collection name. Creates embeddings for the products image and description upsert datapoints to qdrant and allows for text search