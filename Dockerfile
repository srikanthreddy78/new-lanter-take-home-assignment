FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY train_model.py .
COPY relevant_priors_public.json .
RUN python3 train_model.py --data relevant_priors_public.json --output model.pkl

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]