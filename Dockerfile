FROM python:3.9-alpine 
WORKDIR /app

COPY server .
RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]
CMD ["main.py" ]

EXPOSE 5000
