from embedder import encode_text

from vector_store import search_vectors


def search_products(query, limit=5):
    """
    Search products using text query
    """

    print(f"Searching for: {query}")

    # Convert text into vector
    query_vector = encode_text(query)

    # Search similar image vectors
    results = search_vectors(
        vector=query_vector,
        limit=limit
    )

    output = []

    for result in results:

        output.append({

            "score":
                result.score,

            "image_path":
                result.payload[
                    "image_path"
                ],

            "product_name":
                result.payload.get(
                    "product_name",
                    "Unknown Product"
                ),

            "category":
                result.payload.get(
                    "category",
                    "Unknown"
                ),

            "color":
                result.payload.get(
                    "color",
                    "Unknown"
                ),

            "gender":
                result.payload.get(
                    "gender",
                    "Unknown"
                )
        })

    return output