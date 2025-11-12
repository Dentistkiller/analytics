import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pyodbc

load_dotenv()

def _pick_driver() -> str:
    drivers = pyodbc.drivers()
    for d in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"):
        if d in drivers:
            return d
    raise RuntimeError(f"No suitable SQL Server ODBC driver found. Installed: {drivers}")

def sql_engine():
    server = os.getenv("SQL_SERVER")                 # e.g., veripayserver.database.windows.net
    db     = os.getenv("SQL_DB")                     # e.g., bankdb
    user   = os.getenv("SQL_USER")                   # e.g., api_user
    pwd    = os.getenv("SQL_PASSWORD")               # ****
    auth   = (os.getenv("SQL_AUTH") or "sql").lower()

    driver = _pick_driver()
    base = (
        f"driver={quote_plus(driver)}"
        "&Encrypt=yes"
        "&TrustServerCertificate=no"
    )

    if auth == "windows":
        # NOTE: Azure SQL does NOT support Windows auth; only useful for on-prem/local
        url = f"mssql+pyodbc://@{server}:1433/{db}?trusted_connection=yes&{base}"
    else:
        url = (
            "mssql+pyodbc://"
            f"{quote_plus(user)}:{quote_plus(pwd)}@{server}:1433/{db}"
            f"?{base}"
        )

    eng = create_engine(
        url,
        fast_executemany=True,
        pool_pre_ping=True,
        pool_recycle=180,
    )
    return eng
