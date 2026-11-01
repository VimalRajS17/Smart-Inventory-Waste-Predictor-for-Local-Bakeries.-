# Production Deployment Guide for Loaf Ledger

This guide covers deploying Loaf Ledger to production environments with data persistence, production WSGI servers (Gunicorn), reverse proxies (Nginx), database backups, and environment configuration.

---

## 1. Production Architecture Overview

```
                      +-----------------------------+
                      |   Client Web Browser / API   |
                      +--------------+--------------+
                                     | HTTPS (443)
                      +--------------v--------------+
                      |        Nginx / Router       |
                      |   (SSL, Static Files, Gzip) |
                      +--------------+--------------+
                                     | Reverse Proxy (HTTP)
                      +--------------v--------------+
                      |       Gunicorn WSGI         |
                      |   (4 Workers, app:create_app)
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
  +----------v----------+                         +----------v----------+
  |    Persistent Disk   |                         |  PostgreSQL Database|
  |  (Render/Railway/Fly)|                         | (Neon/RDS/Supabase) |
  |      bakery.db       |                         |    (DATABASE_URL)   |
  +---------------------+                         +---------------------+
```

---

## 2. Database Persistence & Host Options

Loaf Ledger defaults to SQLite stored in a single file (`bakery.db`). **Many cloud PaaS hosts (e.g., Render, Railway, Fly.io, Heroku) use ephemeral container filesystems where disks are wiped on every new deployment, restart, or scaling event.** 

To prevent data loss, you have two choices:
1. Attach a **Persistent Disk Volume** to mount SQLite on persistent storage (Recommended for small to medium bakeries).
2. Switch to **PostgreSQL** via `DATABASE_URL` (Recommended for multi-region or strictly stateless PaaS deployments).

### Option A: Persistent Disks with SQLite (Recommended)

#### 1. Render.com Deployment Configuration
Render provides managed container hosting with persistent block storage ("Render Disks"). With Loaf Ledger's `scripts/bootstrap.py`, cold starts on new disks are 100% automated.

##### Web Service Settings
| Setting | Recommended Value |
|---|---|
| **Environment** | `Python` |
| **Region** | Choose closest to your bakery (e.g., `Frankfurt (EU Central)` or `Oregon (US West)`) |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python scripts/bootstrap.py && gunicorn "app:create_app()" --workers 4 --bind 0.0.0.0:$PORT` |

##### Persistent Disk Configuration
1. In the Render Dashboard, select your Web Service -> **Disks** -> **Add Disk**.
2. **Name**: `bakery-data`
3. **Mount Path**: `/var/data`
4. **Size**: `1 GB` (sufficient for years of daily sales, logs, forecasts, and trained models)

##### Environment Variables on Render
Add the following key-value pairs in **Environment**:
| Key | Value | Description |
|---|---|---|
| `DATABASE_PATH` | `/var/data/bakery.db` | Points SQLite to persistent disk mount |
| `SECRET_KEY` | `python -c "import secrets; print(secrets.token_hex(32))"` | Cryptographic session signing key |
| `FLASK_DEBUG` | `0` | Disables debug mode for production security |
| `DEFAULT_LAT` | `51.5074` | Latitude for Open-Meteo weather forecasts |
| `DEFAULT_LON` | `-0.1278` | Longitude for Open-Meteo weather forecasts |
| `HOLIDAY_COUNTRY`| `UK` | Bank holiday calendar for feature engineering |
| `RETRAIN_SCHEDULE_ENABLED` | `1` | Enables nightly APScheduler retraining |
| `RL_ENABLED` | `false` | Contextual bandit toggle (default: false) |

> [!TIP]
> **Zero-Touch Bootstrap Automation:**  
> On your first Render deployment, `scripts/bootstrap.py` automatically detects the empty database, creates all 9 tables, seeds the demo bakery with 730 days of sales history, caches forecast weather from Open-Meteo, trains the forecasting models, and enrolls them into `model_registry`. On subsequent deploys, it detects existing data and skips in under 0.2s. No manual shell commands are required!

#### 2. Railway.app
* Railway provides persistent volume mounts across container rebuilds.
1. In your project canvas, select your service -> **Volumes** -> **Add Volume**.
2. Mount path: `/data`.
3. Under **Variables**, configure:
   ```bash
   DATABASE_PATH=/data/bakery.db
   ```
4. Deploy the service.

#### 3. Fly.io
* Fly.io provides persistent NVMe volumes per machine.
1. Create a 1 GB persistent volume in your target region:
   ```bash
   fly volumes create bakery_data --region lhr --size 1
   ```
2. In your `fly.toml`, add the mounts block:
   ```toml
   [mounts]
     source = "bakery_data"
     destination = "/data"
   ```
3. Set the environment secret:
   ```bash
   fly secrets set DATABASE_PATH=/data/bakery.db SECRET_KEY="<strong-random-key>"
   ```

---

### Option B: PostgreSQL Switch (`DATABASE_URL`)

If your platform does not provide persistent disk volumes (or if you prefer managed cloud databases such as Supabase, Neon, AWS RDS, or DigitalOcean Managed Databases):

1. Install the PostgreSQL adapter:
   ```bash
   pip install psycopg2-binary
   ```
2. Configure the `DATABASE_URL` environment variable:
   ```bash
   DATABASE_URL=postgresql://username:password@db.region.host.com:5432/loafledger
   ```
3. When `DATABASE_URL` begins with `postgresql://` or `postgres://`, Loaf Ledger automatically routes queries to PostgreSQL.
4. Initialize the tables using:
   ```bash
   flask --app app init-db
   ```

---

## 3. Web Server & Process Management (Gunicorn)

Loaf Ledger uses Gunicorn as its production WSGI HTTP server.

### Start Command & Procfile
The included `Procfile` specifies:
```procfile
web: python scripts/bootstrap.py && gunicorn "app:create_app()" --workers 4 --bind 0.0.0.0:$PORT
```

Recommended worker sizing formula:
$$\text{Workers} = 2 \times \text{CPU Cores} + 1$$

For a 2-core container, 4–5 workers is optimal.

### Linux Systemd Service (VPS / Dedicated Server)
If hosting on Ubuntu or Debian:
Create `/etc/systemd/system/loafledger.service`:
```ini
[Unit]
Description=Loaf Ledger Bakery Demand Forecasting Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/loafledger
EnvironmentFile=/var/www/loafledger/.env
ExecStart=/var/www/loafledger/venv/bin/gunicorn "app:create_app()" \
    --workers 4 \
    --bind 127.0.0.1:5000 \
    --timeout 60 \
    --access-logfile /var/log/loafledger/access.log \
    --error-logfile /var/log/loafledger/error.log

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable loafledger
sudo systemctl start loafledger
```

---

## 4. Nginx Reverse Proxy & SSL

Example Nginx configuration with reverse proxying and security headers:

```nginx
server {
    listen 80;
    server_name bakery.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name bakery.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/bakery.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/bakery.yourdomain.com/privkey.pem;

    client_max_body_size 2M;

    location /static/ {
        alias /var/www/loafledger/static/;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 5. Automated Backups & Disaster Recovery

Loaf Ledger provides an atomic, zero-downtime backup command that leverages SQLite's online backup API (`sqlite3.Connection.backup`):

### Running a Manual Backup
```bash
flask --app app backup
```
Output:
```
Database backup successfully created: /path/to/backups/backup_20261101_120000.db (608.0 KB)
```

Custom destination directory and name:
```bash
flask --app app backup --output-dir /var/backups/loafledger --filename daily_backup.db
```

### Nightly Cron Job for Linux Host
Add to crontab (`crontab -e`):
```cron
# Run daily database backup at 03:00 UTC
0 3 * * * cd /var/www/loafledger && /var/www/loafledger/venv/bin/flask --app app backup --output-dir /var/backups/loafledger >> /var/log/loafledger/backup.log 2>&1
```

### Restoring from Backup
To restore a snapshot:
1. Stop the application service:
   ```bash
   sudo systemctl stop loafledger
   ```
2. Copy the verified backup over `bakery.db`:
   ```bash
   cp /var/backups/loafledger/backup_20261101_120000.db /var/data/bakery.db
   ```
3. Restart the application service:
   ```bash
   sudo systemctl start loafledger
   ```

---

## 6. Health & Uptime Monitoring

Monitor service availability by pointing your health checker (e.g. UptimeRobot, Datadog, BetterStack) to:
```http
GET /api/health
```

Expected HTTP 200 response:
```json
{
  "status": "ok",
  "database": "connected",
  "scheduler": "running",
  "rl_enabled": false,
  "timestamp": "2026-11-01T12:00:00+00:00"
}
```
If the database connection fails, the endpoint returns `HTTP 503 Service Unavailable` with `"status": "degraded"` and `"database": "disconnected"`.

