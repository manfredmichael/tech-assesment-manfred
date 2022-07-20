FROM python:3.10-slim-buster
WORKDIR /app

COPY server .
RUN pip install -r requirements.txt

ENTRYPOINT [ "streamlit", "run" ]
CMD ["app.py" ]

EXPOSE 8501 
