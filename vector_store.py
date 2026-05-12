from qdrant_client import QdrantClient

from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

# Persistent local database
client = QdrantClient(
    path="./qdrant_data"
)

COLLECTION_NAME = "products"


def create_collection():
    """
    Create vector collection
    """

    collections = client.get_collections().collections

    existing = [
        collection.name
        for collection in collections
    ]

    if COLLECTION_NAME not in existing:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE
            )
        )

        print(
            f"Created collection: {COLLECTION_NAME}"
        )

    else:

        print("Collection already exists")


def insert_vector(
    point_id,
    vector,
    image_path,
    metadata
):
    """
    Insert image vector into Qdrant
    """

    client.upsert(

        collection_name=COLLECTION_NAME,

        points=[

            PointStruct(

                id=point_id,

                vector=vector,

                payload={

                    "image_path": image_path,

                    "product_name":
                        metadata["product_name"],

                    "category":
                        metadata["category"],

                    "color":
                        metadata["color"],

                    "gender":
                        metadata["gender"]
                }
            )
        ]
    )


def search_vectors(vector, limit=5):
    """
    Search similar vectors
    """

    results = client.search(

        collection_name=COLLECTION_NAME,

        query_vector=vector,

        limit=limit
    )

    return results