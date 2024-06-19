# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DulangTagList(Base):
    __tablename__ = 'DulangTagList'
    tagID = Column(Integer, primary_key=True, autoincrement=True)
    tag = Column(String(200), unique=True)
    description = Column(String(200))
    type = Column(String(200))
    area = Column(String(200))
    category = Column(String(200))
    sapmapping = Column(String(200))
    upperlimit = Column(Float)
    lowlimit = Column(Float)

class CGCE7K(Base):
    __tablename__ = 'CGCE7K'
    id = Column(Integer, primary_key=True, autoincrement=True)
    tagID = Column(Integer, ForeignKey('DulangTagList.tagID'))
    value = Column(Float)
    time = Column(DateTime)
