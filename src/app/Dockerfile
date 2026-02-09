FROM ython:3.12-slim

WORKDIR /app

COPY src/app/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY src/app ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--port", "8000", "--reload"]