
from fastapi import Request,HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
import os
from datetime import datetime,timedelta
from datetime import timezone
import logging

if os.getenv("DEBUG_REQUEST","false").upper() == "TRUE":
    logging.basicConfig(level=logging.DEBUG)

# ******************** JWT Token config ********************


# 32 bit secret key
SECRET_KEY = "4f72ee5678ea0fbdd47a7f0253dd7f4708da99e271d1fc9ee52e8dcd0fe02a6f"

 
# encryption algorithm
ALGORITHM = "HS256"



# ******************** Functions to create and verifie JWT Token ********************

def create_access_token(data: dict):
    to_encode = data.copy()

    # expire time of the token
    expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode["exp"] = expire
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        if not credentials:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")
        if credentials.scheme != "Bearer":
            raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
        if not self.verify_jwt(credentials.credentials):
            raise HTTPException(status_code=403, detail="Invalid token or expired token.")
        return credentials.credentials

    def verify_jwt(self, jwtoken: str):
        try:
            payload = jwt.decode(jwtoken, SECRET_KEY, algorithms=[ALGORITHM])
        except Exception:
            payload = None
        return bool(payload)
