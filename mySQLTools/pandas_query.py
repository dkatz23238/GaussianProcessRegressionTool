import pandas as pd
from sqlalchemy import create_engine
try:
    import pymysql
    pymysql.install_as_MySQLdb()
except:
    pass
url = "mysql://root:password@localhost/axdb"
engine = create_engine(url)
print(pd.read_sql("SELECT * FROM arm_v2", con=engine))