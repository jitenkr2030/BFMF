# Deployment Guide

This guide covers various deployment options and best practices for the Bharat Foundation Model Framework (BFMF) in production environments.

## 📋 Deployment Options

BFMF supports multiple deployment strategies:

1. **Local Development**: For development and testing
2. **Cloud Deployment**: AWS, GCP, Azure
3. **Container Deployment**: Docker and Kubernetes
4. **Edge Deployment**: On-premises and edge devices
5. **Serverless Deployment**: AWS Lambda, Google Cloud Functions

## 🚀 Local Development Deployment

### Prerequisites
- Python 3.8+
- 8GB RAM minimum
- 10GB storage space

### Setup
```bash
# Clone repository
git clone https://github.com/bharat-ai/bharat-fm.git
cd bharat-fm

# Create virtual environment
python -m venv bharat_fm_env
source bharat_fm_env/bin/activate  # On Windows: bharat_fm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize BFMF
python setup.py install

# Start development server
python -m bharat_fm.server --port 8000 --debug
```

### Configuration
Create `.env` file:
```env
BFMF_ENV=development
BFMF_PORT=8000
BFMF_LOG_LEVEL=DEBUG
BFMF_DATA_DIR=./data
BFMF_CACHE_DIR=./cache
DATABASE_URL=sqlite:///./bharat_fm.db
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Using EC2
```bash
# Launch EC2 instance (Ubuntu 20.04)
# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3 python3-pip git

# Clone and setup BFMF
git clone https://github.com/bharat-ai/bharat-fm.git
cd bharat-fm
pip3 install -r requirements.txt
python3 setup.py install

# Create systemd service
sudo tee /etc/systemd/system/bharat-fm.service > /dev/null <<EOF
[Unit]
Description=Bharat FM Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/bharat-fm
ExecStart=/home/ubuntu/bharat-fm/bharat_fm_env/bin/python -m bharat_fm.server
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable bharat-fm
sudo systemctl start bharat-fm
```

#### Using AWS ECS
```yaml
# task-definition.yaml
family: bharat-fm
networkMode: bridge
requiresCompatibilities: ['EC2']
cpu: 1024
memory: 2048
executionRoleArn: arn:aws:iam::account:role/ecsTaskExecutionRole
containerDefinitions:
  - name: bharat-fm
    image: your-account.dkr.ecr.region.amazonaws.com/bharat-fm:latest
    portMappings:
      - containerPort: 8000
    environment:
      - name: BFMF_ENV
        value: production
      - name: DATABASE_URL
        value: ${DATABASE_URL}
    logConfiguration:
      logDriver: awslogs
      options:
        awslogs-group: bharat-fm
        awslogs-region: us-east-1
        awslogs-stream-prefix: ecs
```

### Google Cloud Platform Deployment

#### Using Google Compute Engine
```bash
# Create Compute Engine instance
gcloud compute instances create bharat-fm \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=e2-medium \
    --tags=http-server,https-server

# Connect to instance
gcloud compute ssh bharat-fm

# Install and setup (same as EC2)
```

#### Using Google Cloud Run
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/bharat-fm', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/bharat-fm']
images:
  - 'gcr.io/$PROJECT_ID/bharat-fm'
```

### Azure Deployment

#### Using Azure Virtual Machines
```bash
# Create Azure VM
az vm create \
    --resource-group bharat-fm-rg \
    --name bharat-fm-vm \
    --image UbuntuLTS \
    --admin-username azureuser \
    --generate-ssh-keys

# Connect and setup (similar to AWS)
```

## 🐳 Container Deployment

### Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install BFMF
RUN python setup.py install

# Create non-root user
RUN useradd -m -u 1000 bharat && chown -R bharat:bharat /app
USER bharat

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "bharat_fm.server", "--port", "8000"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  bharat-fm:
    build: .
    ports:
      - "8000:8000"
    environment:
      - BFMF_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/bharat_fm
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data
      - ./cache:/app/cache
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=bharat_fm
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - bharat-fm
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### Build and Run
```bash
# Build Docker image
docker build -t bharat-fm .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f bharat-fm
```

### Kubernetes Deployment

#### deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bharat-fm
  labels:
    app: bharat-fm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bharat-fm
  template:
    metadata:
      labels:
        app: bharat-fm
    spec:
      containers:
      - name: bharat-fm
        image: your-registry/bharat-fm:latest
        ports:
        - containerPort: 8000
        env:
        - name: BFMF_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: bharat-fm-secrets
              key: database-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: bharat-fm-service
spec:
  selector:
    app: bharat-fm
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

#### configmap.yaml
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bharat-fm-config
data:
  BFMF_ENV: "production"
  BFMF_LOG_LEVEL: "INFO"
  BFMF_CACHE_SIZE: "1GB"
```

#### secret.yaml
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: bharat-fm-secrets
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  redis-url: <base64-encoded-redis-url>
  api-key: <base64-encoded-api-key>
```

#### Deploy to Kubernetes
```bash
# Apply configurations
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check deployment status
kubectl get pods
kubectl get services
```

## 🏠 Edge Deployment

### On-Premises Deployment

#### System Requirements
- Server with 16GB RAM minimum
- 4 CPU cores minimum
- 100GB storage space
- Ubuntu 20.04 LTS or CentOS 8

#### Installation Script
```bash
#!/bin/bash
# install-bharat-fm.sh

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv git nginx supervisor

# Create user
sudo useradd -m -s /bin/bash bharat
sudo usermod -aG sudo bharat

# Clone repository
sudo -u bharat git clone https://github.com/bharat-ai/bharat-fm.git /home/bharat/bharat-fm
cd /home/bharat/bharat-fm

# Setup virtual environment
sudo -u bharat python3 -m venv bharat_fm_env
sudo -u bharat source bharat_fm_env/bin/activate
sudo -u bharat pip install -r requirements.txt
sudo -u bharat python setup.py install

# Create directories
sudo -u bharat mkdir -p /home/bharat/bharat-fm/data /home/bharat/bharat-fm/cache
sudo -u bharat mkdir -p /var/log/bharat-fm

# Setup systemd service
sudo tee /etc/systemd/system/bharat-fm.service > /dev/null <<EOF
[Unit]
Description=Bharat FM Service
After=network.target

[Service]
Type=simple
User=bharat
WorkingDirectory=/home/bharat/bharat-fm
Environment=PATH=/home/bharat/bharat-fm/bharat_fm_env/bin
ExecStart=/home/bharat/bharat-fm/bharat_fm_env/bin/python -m bharat_fm.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Setup nginx configuration
sudo tee /etc/nginx/sites-available/bharat-fm > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/bharat-fm /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Start service
sudo systemctl daemon-reload
sudo systemctl enable bharat-fm
sudo systemctl start bharat-fm

echo "BFMF installation completed!"
echo "Access at: http://your-server-ip"
```

### Raspberry Pi Deployment

#### Setup Script
```bash
#!/bin/bash
# install-raspberry-pi.sh

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv git

# Create swap file (if needed)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Clone repository
git clone https://github.com/bharat-ai/bharat-fm.git
cd bharat-fm

# Setup virtual environment
python3 -m venv bharat_fm_env
source bharat_fm_env/bin/activate
pip install -r requirements.txt

# Install BFMF with optimizations
python setup.py install

# Create service
sudo tee /etc/systemd/system/bharat-fm.service > /dev/null <<EOF
[Unit]
Description=Bharat FM Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/bharat-fm
ExecStart=/home/pi/bharat-fm/bharat_fm_env/bin/python -m bharat_fm.server --port 8000 --lightweight
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable bharat-fm
sudo systemctl start bharat-fm

echo "BFMF installed on Raspberry Pi!"
echo "Access at: http://raspberry-pi-ip:8000"
```

## ⚡ Serverless Deployment

### AWS Lambda Deployment

#### lambda_function.py
```python
import json
from bharat_fm import BharatFM

# Initialize BFMF (outside handler for reuse)
bfmf = None

def init_bfmf():
    global bfmf
    if bfmf is None:
        bfmf = BharatFM(config={
            'cache_size': '100MB',
            'lightweight_mode': True
        })

def handler(event, context):
    try:
        # Initialize BFMF if not already done
        init_bfmf()
        
        # Parse request
        body = json.loads(event.get('body', '{}'))
        message = body.get('message', '')
        language = body.get('language', 'en')
        
        # Process request
        response = bfmf.chat(message, language=language)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'response': response,
                'success': True
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'success': False
            })
        }
```

#### serverless.yml
```yaml
service: bharat-fm-lambda

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  memorySize: 1024
  timeout: 30

functions:
  chat:
    handler: lambda_function.handler
    events:
      - http:
          path: chat
          method: post
          cors: true

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    layer: true
    usePoetry: false
```

### Google Cloud Functions Deployment

#### main.py
```python
import functions_framework
from bharat_fm import BharatFM

# Initialize BFMF
bfmf = BharatFM(config={'lightweight_mode': True})

@functions_framework.http
def chat(request):
    try:
        # Parse request
        request_json = request.get_json()
        message = request_json.get('message', '')
        language = request_json.get('language', 'en')
        
        # Process request
        response = bfmf.chat(message, language=language)
        
        return {
            'response': response,
            'success': True
        }, 200
    
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }, 500
```

## 🔧 Configuration Management

### Environment Variables

#### Production Environment
```env
# General
BFMF_ENV=production
BFMF_LOG_LEVEL=INFO
BFMF_PORT=8000

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/bharat_fm
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Cache
REDIS_URL=redis://localhost:6379
CACHE_SIZE=2GB
CACHE_TTL=3600

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
API_RATE_LIMIT=1000/hour

# Performance
MAX_WORKERS=4
THREAD_POOL_SIZE=100
REQUEST_TIMEOUT=30

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### Configuration Files

#### config/production.yaml
```yaml
bfmf:
  environment: production
  debug: false
  
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
database:
  url: postgresql://user:pass@localhost:5432/bharat_fm
  pool_size: 20
  max_overflow: 30
  
cache:
  provider: redis
  url: redis://localhost:6379
  size: 2GB
  ttl: 3600
  
security:
  secret_key: your-secret-key
  jwt_secret: your-jwt-secret
  rate_limit: 1000/hour
  
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30
```

## 📊 Monitoring and Logging

### Health Checks

#### Health Check Endpoint
```python
# health_check.py
from bharat_fm import BharatFM
from bharat_fm.memory import ConversationMemory
from bharat_fm.utils.database import get_db

def health_check():
    """Comprehensive health check for BFMF."""
    status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {}
    }
    
    try:
        # Check BFMF core
        bfmf = BharatFM()
        status['checks']['bfmf_core'] = 'healthy'
        
        # Check memory system
        memory = ConversationMemory()
        status['checks']['memory_system'] = 'healthy'
        
        # Check database
        db = get_db()
        db.execute('SELECT 1')
        status['checks']['database'] = 'healthy'
        
        # Check cache (Redis)
        redis_client = redis.Redis()
        redis_client.ping()
        status['checks']['cache'] = 'healthy'
        
    except Exception as e:
        status['status'] = 'unhealthy'
        status['error'] = str(e)
    
    return status
```

### Logging Configuration

#### logging.yaml
```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  json:
    format: '%(asctime)s %(levelname)s %(name)s %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /var/log/bharat-fm/app.log
    maxBytes: 10485760
    backupCount: 5
  
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: json
    filename: /var/log/bharat-fm/error.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  bharat_fm:
    level: INFO
    handlers: [console, file, error_file]
    propagate: false
```

### Metrics Collection

#### metrics.py
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
REQUEST_COUNT = Counter('bfmf_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('bfmf_request_duration_seconds', 'Request duration')
ACTIVE_USERS = Gauge('bfmf_active_users', 'Number of active users')
MEMORY_USAGE = Gauge('bfmf_memory_usage_bytes', 'Memory usage in bytes')
CACHE_HIT_RATE = Gauge('bfmf_cache_hit_rate', 'Cache hit rate')

def record_request(method, endpoint, duration):
    """Record request metrics."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    REQUEST_DURATION.observe(duration)

def update_active_users(count):
    """Update active users gauge."""
    ACTIVE_USERS.set(count)

def update_memory_usage(usage_bytes):
    """Update memory usage gauge."""
    MEMORY_USAGE.set(usage_bytes)

def update_cache_hit_rate(hit_rate):
    """Update cache hit rate gauge."""
    CACHE_HIT_RATE.set(hit_rate)

def get_metrics():
    """Get Prometheus metrics."""
    return generate_latest()
```

## 🔒 Security Considerations

### SSL/TLS Configuration

#### nginx.conf
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Firewall Configuration

#### ufw setup
```bash
# Enable firewall
sudo ufw enable

# Allow SSH, HTTP, HTTPS
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https

# Allow database ports (if needed)
sudo ufw allow 5432  # PostgreSQL
sudo ufw allow 6379  # Redis

# Deny all other incoming traffic
sudo ufw default deny incoming

# Check status
sudo ufw status
```

### Security Headers

#### security middleware
```python
# security_middleware.py
from django.middleware.security import SecurityMiddleware

class CustomSecurityMiddleware(SecurityMiddleware):
    def process_response(self, request, response):
        # Add security headers
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
        
        # CSP header
        response['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        
        return response
```

## 🚀 Performance Optimization

### Caching Strategy

#### Redis Configuration
```python
# cache_config.py
import redis
from bharat_fm.cache import CacheManager

class RedisCacheManager(CacheManager):
    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
    
    def get(self, key):
        """Get value from cache."""
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key, value, ttl=3600):
        """Set value in cache with TTL."""
        try:
            return self.redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key):
        """Delete key from cache."""
        try:
            return self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
```

### Database Optimization

#### Connection Pooling
```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def get_database_engine():
    """Create optimized database engine."""
    return create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
```

### Load Balancing

#### nginx load balancer
```nginx
upstream bharat_fm_backend {
    least_conn;
    server 10.0.1.1:8000 weight=3;
    server 10.0.1.2:8000 weight=3;
    server 10.0.1.3:8000 weight=2;
    keepalive 32;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://bharat_fm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Health checks
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
        proxy_send_timeout 5s;
    }
}
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

#### .github/workflows/deploy.yml
```yaml
name: Deploy BFMF

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=bharat_fm
    
    - name: Run linting
      run: |
        flake8 bharat_fm/
        black --check bharat_fm/

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
    
    - name: Login to DockerHub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v2
      with:
        context: .
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/bharat-fm:latest
    
    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2.0.0
      with:
        config: ${{ secrets.KUBE_CONFIG }}
        command: apply -f k8s/
```

This deployment guide provides comprehensive information for deploying BFMF in various environments. Choose the deployment method that best suits your requirements and infrastructure.