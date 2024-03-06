import os
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import APIRouter

load_dotenv()

app = FastAPI()

app.add_middleware(CORSMiddleware, 
                   allow_origins=['*'],
                   allow_credentials=True,
                   Allow_methods=['*'],
                   allow_headers=['*']
                   )


user_router = APIRouter(
    prefix='/api/v1/main',
    tags=['main']
)

@user_router.get('/{id}')
async def read_user():

    return {'message': 'Hello'}

async def create_user():
    return {'message': 'Hello'}

app.include_router(user_router)

'''
if __name__ == '__main__':
    uvicorn.run('main:app', host=os.getenv('HOST'),
                port=int(os.getenv('PORT')),
                reload=True)
'''    
if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost',
                port=8080,
                reload=True)
