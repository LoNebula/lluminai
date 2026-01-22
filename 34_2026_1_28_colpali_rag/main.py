import torch
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device

def get_image_from_url_or_create_dummy(url, desc):
    """画像をDLし、失敗したらダミー画像を生成する安全装置"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        print(f"   Downloading: {desc}...", end="")
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        print(" OK!")
        return img
    except Exception as e:
        print(f" Failed ({e}). Creating dummy image instead.")
        # 白紙の画像を作成
        img = Image.new('RGB', (448, 448), color=(255, 255, 255))
        return img

def main():
    print("🚀 環境設定を確認中...")
    device = get_torch_device("auto")
    print(f"   Device: {device}")

    # 1. モデルとプロセッサのロード
    model_name = "vidore/colpali-v1.2"
    
    print(f"📥 モデルをロード中: {model_name} ...")
    model = ColPali.from_pretrained(
        model_name,
        # メモリ節約のため bfloat16 を強制使用 (CPUでもロード可能にする)
        dtype=torch.bfloat16, 
        device_map=device,
    ).eval()

    processor = ColPaliProcessor.from_pretrained(model_name)
    print("✅ モデルロード完了")

    # 2. テスト用画像の準備 (Githubの安定した画像 + 予備)
    # ColPaliのアーキテクチャ図（文字を含んだ文書として利用）
    # COCOデータセットの猫（自然画像として利用）
    image_sources = [
        {"url": "https://raw.githubusercontent.com/illuin-tech/colpali/main/assets/colpali_architecture.png", "desc": "Document(ColPali Paper)"},
        {"url": "http://images.cocodataset.org/val2017/000000039769.jpg", "desc": "Cat Image"}
    ]
    
    images = []
    valid_descs = []
    
    print("\n🖼️ 画像を準備中...")
    for source in image_sources:
        img = get_image_from_url_or_create_dummy(source["url"], source["desc"])
        images.append(img)
        valid_descs.append(source["desc"])

    # 3. クエリの準備
    queries = [
        "What is ColPali?",            # アーキテクチャ図用
        "Is there a cat?",             # 猫画像用
        "Show me the architecture."    # アーキテクチャ図用
    ]

    # 4. 前処理
    print("\n⚙️ Embedding生成とスコアリング計算中... (CPUの場合、30秒〜1分ほどかかります)")
    
    # 画像の処理
    batch_images = processor.process_images(images).to(device)
    # クエリの処理
    batch_queries = processor.process_queries(queries).to(device)

    # 5. 推論 & スコア計算
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    # ColPaliのスコア計算
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)

    # 6. 結果の表示
    print("\n📊 --- 検索結果 (類似度スコア) ---")
    for i, query in enumerate(queries):
        print(f"\n🔍 Query: '{query}'")
        for j, desc in enumerate(valid_descs):
            score = scores[i, j].item()
            print(f"   📄 vs {desc}: Score = {score:.4f}")
            
            # 相対比較で判定
            # 他の画像よりスコアが顕著に高ければマッチとみなす
            if score == max(scores[i]):
                print("      👉 ★ Top Match!")

if __name__ == "__main__":
    main()