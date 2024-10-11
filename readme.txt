####docker build
sudo docker build --pull --rm -f Dockerfile -t kby-ai-live:latest .


#### docker run with online license
sudo docker run -e LICENSE="xxxxx" -p 8080:8080 -p 9000:9000 kby-ai-live

#### docker run with offline license
sudo docker run -v ./license.txt:/root/kby-ai-live/license.txt -p 8080:8080 -p 9000:9000 kby-ai-live
