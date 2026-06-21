"""
Setup script for production Qdrant collection
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, OptimizersConfig, QuantizationConfig, BinaryQuantization
from dotenv import load_dotenv

load_dotenv()

def setup_production_collection():
    """Create optimized production collection"""

    # Get credentials from environment
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("COLLECTION_NAME", "research_papers_prod")

    if not qdrant_url or not qdrant_api_key:
        print("Missing QDRANT_URL or QDRANT_API_KEY in .env file")
        return False

    try:
        # Initialize client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

       

        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if collection_name in collection_names:
            print(f"Collection '{collection_name}' already exists")
            user_input = input("Delete and recreate? (y/N): ").lower()
            if user_input == 'y':
                client.delete_collection(collection_name)
                print(f"Deleted existing collection")
            else:
                print("Using existing collection")
                return True

        # Create optimized collection for production
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384, #this will vary in production
                distance=Distance.COSINE
            )
        )

        print(f"Created production collection: {collection_name}")

        print("Setting up indexes...")

        # Test the collection
        info = client.get_collection(collection_name)
        print(f"Collection info:")
        print(f"   - Vectors: {info.vectors_count}")
        print(f"   - Status: {info.status}")
        print(f"   - Vector size: {info.config.params.vectors.size}")

        return True

    except Exception as e:
        print(f"Error setting up collection: {e}. Please retry to create proper collection")
        return False

if __name__ == "__main__":
    print("Setting up production Qdrant collection...")
    success = setup_production_collection()

    if success:
        print("\n Setup complete, and the collection is available for production analysis.")
    else:
        print("\nSetup failed. Check .env values and try again.")
