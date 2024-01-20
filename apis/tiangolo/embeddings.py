from fastapi import FastAPI, Depends, Query
from pgvector.sqlalchemy import Vector
from sqlmodel import Field, SQLModel, create_engine, Session, select, text, Column

##############
### MODELS ###
##############

user = "XXX"
password = "XXX"
host = "XXX"
port = "XXX"
dbname = "XXX"
url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(url, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def enable_pgvector():
    with Session(engine) as session:
        session.exec(text('CREATE EXTENSION IF NOT EXISTS vector'))
        session.commit()


class Question(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    embedding: list[float] = Field(sa_column=Column(Vector(3)))


class QuestionRead(SQLModel):
    id: int
    text: str
    embedding: list[float]


class QuestionCreate(SQLModel):
    text: str
    embedding: list[float]


#################
### ENDPOINTS ###
#################

app = FastAPI()


@app.on_event("startup")
def on_startup():
    enable_pgvector()
    create_db_and_tables()


def get_session():
    with Session(engine) as session:
        yield session


@app.post("/questions/", response_model=QuestionRead)
def create_question(*, session: Session = Depends(get_session), question: QuestionCreate):
    db_question = Question.model_validate(question)
    session.add(db_question)
    session.commit()
    session.refresh(db_question)
    return db_question


@app.get("/questions/", response_model=list[QuestionRead])
def read_questions(
    *,
    session: Session = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=100, le=100),
):
    questions = session.exec(select(Question).offset(offset).limit(limit)).all()
    return questions
