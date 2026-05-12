from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from PIL import Image
import io

from search import search_products

app = FastAPI()

# Serve image files
app.mount(
    "/images",
    StaticFiles(directory="images"),
    name="images"
)

# Templates folder
templates = Jinja2Templates(
    directory="templates"
)


@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


@app.get("/search")
def search(q: str, top_k: int = 8):

    results = search_products(
        q,
        limit=top_k
    )

    return {
        "query": q,
        "results": results
    }


@app.get("/stats")
def stats():

    from vector_store import (
        client,
        COLLECTION_NAME
    )

    info = client.get_collection(
        COLLECTION_NAME
    )

    return {
        "total": info.points_count
    }
@app.post("/search-by-image")
async def search_by_image(file: UploadFile = File(...)):

    from embedder import encode_uploaded_image
    from vector_store import search_vectors

    # Read uploaded image
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    )

    # Convert image into vector
    query_vector = encode_uploaded_image(image)

    # Search similar products
    results = search_vectors(
        vector=query_vector,
        limit=8
    )

    output = []

    for result in results:

        output.append({

            "score": result.score,

            "image_path": result.payload[
                "image_path"
            ]
        })

    return {
        "results": output
    }