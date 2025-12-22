FROM amazonlinux:2

RUN yum update -y && \
    amazon-linux-extras install epel -y && \
    yum install -y \
        poppler-utils \
        tesseract \
        tesseract-langpack-eng \
        tesseract-langpack-tam \
        python3 \
        python3-pip \
        which \
    && yum clean all

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
