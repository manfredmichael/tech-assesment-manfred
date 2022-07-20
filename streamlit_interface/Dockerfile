FROM python:3.10-slim-buster
WORKDIR /app

COPY server .
RUN pip install -r requirements.txt

CMD streamlit run app.py --server.port $PORT 
