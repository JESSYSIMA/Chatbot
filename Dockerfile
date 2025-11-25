# Étape 1: image Python
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code du projet
COPY . .

# Exposer le port (celui de gunicorn)
EXPOSE 8000

# Commande pour lancer le serveur Django avec migrations
CMD ["sh", "-c", "python manage.py makemigrations --noinput && python manage.py migrate --noinput && gunicorn OurChat.wsgi:application --bind 0.0.0.0:8000"]
