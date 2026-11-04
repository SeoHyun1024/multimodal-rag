```bash
git clone [<저장소 주소>](https://github.com/SeoHyun1024/multimodal-rag.git)

cd multimodal-rag
```

# 가상환경 생성 및 활성화

python -m venv .venv
source .venv/bin/activate # 윈도우: .venv\Scripts\activate

# 의존성 설치

pip install --upgrade pip
pip install -r requirements.txt # 또는 requirements.lock.txt

# 커널 등록

python -m ipykernel install --user --name multimodal-rag

# .env 작성

cp .env.example .env # 윈도우: copy .env.example .env

# .env 파일을 열어 실제 API 키를 입력
