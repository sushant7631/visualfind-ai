import os
import pandas as pd

from embedder import encode_image

from vector_store import (
    create_collection,
    insert_vector
)

IMAGE_FOLDER = "images"

# Load metadata CSV
products_df = pd.read_csv(
    "styles.csv",
    on_bad_lines="skip"
)

# Convert id column to string
products_df["id"] = (
    products_df["id"]
    .astype(str)
)


def index_images():

    # Create vector database collection
    create_collection()

    # Get all image files
    image_files = [

        file_name

        for file_name in os.listdir(
            IMAGE_FOLDER
        )

        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(f"Found {len(image_files)} images")

    # Process each image
    for idx, file_name in enumerate(image_files):

        image_path = os.path.join(
            IMAGE_FOLDER,
            file_name
        )

        try:

            print(f"Encoding: {file_name}")

            # Convert image into vector
            vector = encode_image(image_path)

            # Get product id from filename
            product_id = os.path.splitext(
                file_name
            )[0]

            # Find matching CSV row
            match = products_df[
                products_df["id"] == product_id
            ]

            # Default metadata
            metadata = {
                "product_name": product_id,
                "category": "Unknown",
                "color": "Unknown",
                "gender": "Unknown"
            }

            # If metadata exists
            if not match.empty:

                row = match.iloc[0]

                metadata = {

                    "product_name":
                        str(
                            row.get(
                                "productDisplayName",
                                product_id
                            )
                        ),

                    "category":
                        str(
                            row.get(
                                "masterCategory",
                                "Unknown"
                            )
                        ),

                    "color":
                        str(
                            row.get(
                                "baseColour",
                                "Unknown"
                            )
                        ),

                    "gender":
                        str(
                            row.get(
                                "gender",
                                "Unknown"
                            )
                        )
                }

            # Save vector into Qdrant
            insert_vector(
                point_id=idx,
                vector=vector,
                image_path=image_path,
                metadata=metadata
            )

            print(f"Indexed: {file_name}")

        except Exception as error:

            print(
                f"Error processing {file_name}: {error}"
            )

    print("Indexing complete")


if __name__ == "__main__":

    index_images()