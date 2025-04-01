from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

# Create the FastAPI app
app = FastAPI()


class Example(BaseModel):
    name: str

class User(BaseModel):
    id: int # TODO: why not UUID?
    name: str
    email: EmailStr

# TODO: what is the major difference between using Pydantic model and python dict in FastAPI?
users_db = [
    User(id=1, name="user1", email="dummy_email@gmail.com"),
    User(id=2, name="user2", email="dummy_email2@gmail.com"), # TODO: enforcing strict uniqueness?
    User(id=2, name="user3", email="dummy_email3@gmail.com"),
    User(id=2, name="user4", email="dummy_email4@gmail.com"),
]

@app.get("/", response_model=list[Example])
async def get_examples():
    return [Example(name="example")]

# ====================================================================================
# Response Model, filering, security
# ====================================================================================

from typing import Any

class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr
    full_name: str | None = None

class UserOut(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None


@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn) -> Any:
    return user

# ====================================================================================
# 
# ====================================================================================

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float = 10.5
    tags: list[str] = []


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item(item_id: str):
    return items[item_id]

# ====================================================================================
# 
# ====================================================================================

@app.get("/users", response_model=list[User])
async def get_users():
    return users_db

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    for user in users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail="User not found")

# How can we do differently?
# (1) Large file with lots of records; (2) Database with individually written records; (3) Sending records over a network connection;
@app.post("/users", response_model=User)
async def create_user(user: User):
    users_db.append(user)
    return user

from fastapi import Query

@app.get("/users", response_model=List[User])
async def get_users(skip: int = 0, limit: int = 10):
    return users_db[skip : skip + limit]


# filtering
@app.get("/users/search", response_model=List[User])
async def search_users(name: str = Query(None)):
    if name:
        return [user for user in users_db if user.name.lower() == name.lower()]
    return users_db

# =================================================================================
# API key authentication
# =================================================================================

# define security dependency
from fastapi.security import APIKeyHeader
from fastapi import Depends

API_KEY = "mysecretkey"
api_key_header = APIKeyHeader(name="X-API-Key")

def authenticate(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.get("/protected", response_model=List[User])
async def get_protected_users(api_key: str = Depends(authenticate)):
    return users_db


# protect an endpoint
@app.get("/protected", response_model=List[User])
async def get_protected_users(api_key: str = Depends(authenticate)):
    return users_db

  
# =================================================================================
# input validation and sanitization
# =================================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, constr, conint

app = FastAPI()

class UserInput(BaseModel):
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    age: conint(ge=18)

@app.post("/users/")
async def create_user(user: UserInput):
    # Process the validated user data
    return {"message": "User created successfully", "user": user}

# =================================================================================
# Secure API with Background Processing
# =================================================================================

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, EmailStr
from typing import List
import time

app = FastAPI()

API_KEY = "mysecretkey"
api_key_header = APIKeyHeader(name="X-API-Key")

def authenticate(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

class Record(BaseModel):
    id: int
    name: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    timestamp: float = Field(default_factory=time.time)

def process_records(records: List[Record]):
    time.sleep(2)
    print(f"Processed {len(records)} records.")

@app.post("/records/", response_model=dict)
async def receive_records(records: List[Record], api_key: str = Depends(authenticate)):
    if len(records) > 1000:
        raise HTTPException(status_code=413, detail="Payload too large")
    
    print(f"Received {len(records)} records.")
    return {"message": "Records received successfully", "count": len(records)}

@app.post("/records/background/", response_model=dict)
async def receive_records_background(records: List[Record], background_tasks: BackgroundTasks, api_key: str = Depends(authenticate)):
    if len(records) > 1000:
        raise HTTPException(status_code=413, detail="Payload too large")
    
    background_tasks.add_task(process_records, records)
    return {"message": "Records are being processed in the background"}

# =================================================================================
# httpx, external api calls, non-blocking, async
# =================================================================================

from fastapi import FastAPI
import httpx

app = FastAPI()

@app.get("/fetch-data")
async def fetch_data():
    async with httpx.AsyncClient() as client:
        # context manager
        # If an exception occurs before aclose() runs, the client stays open → memory leaks!
        response = await client.get("https://api.example.com/data")  # ✅ Non-blocking API call
    return response.json()

# this will blcok execution
import requests

@app.get("/fetch-data")
def fetch_data():
    response = requests.get("https://api.example.com/data")  # ❌ Blocks the entire thread
    return response.json()

# =================================================================================
# Multiple params
# =================================================================================

from typing import Annotated

from fastapi import FastAPI, Path
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.put("/items/{item_id}")
async def update_item(
    item_id: Annotated[int, Path(title="The ID of the item to get", ge=0, le=1000)],
    q: str | None = None,
    item: Item | None = None,
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    if item:
        results.update({"item": item})
    return results

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
