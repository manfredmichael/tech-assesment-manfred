FROM python:3.10-slim-buster
WORKDIR /app

COPY server .
RUN pip install -r requirements.txt

ENTRYPOINT [ "streamlit", "run" ]
CMD ["main.py" ]

EXPOSE 5000
