"""Autenticacao: valida JWT do Supabase Auth e exige vinculo em professionals."""
from fastapi import Depends, HTTPException, Header
from ..db import anon_client, table


def current_professional(authorization: str = Header(default="")) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token ausente")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        user = anon_client().auth.get_user(token)
        uid = user.user.id
    except Exception:
        raise HTTPException(401, "Token invalido ou expirado")

    prof = table("professionals").select("*").eq("id", uid).execute()
    if not prof.data:
        raise HTTPException(403, "Usuario nao cadastrado como profissional")
    return prof.data[0]
